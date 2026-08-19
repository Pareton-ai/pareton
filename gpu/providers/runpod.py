"""Runpod GPU cloud provider (REST API v2).

Implements Pareton's Provider protocol against ``https://api.runpod.io/v2``.
Uses ``gpu.ssh`` for the data plane. Host-local persistent mount at
``/workspace`` (deleted with the pod). Requires **direct** SSH
(``22/tcp`` + ``ssh.direct``) because bootstrap/rsync cannot use the
interactive-only proxy.

Bench needs a Docker host (``docker pull`` / ``docker run``). Stock Runpod
shared pods do not expose privileged / DinD; provision fails fast after SSH
is up if ``docker info`` cannot run, so the orchestrator can fall through.
Keep ``runpod`` out of the default ``PARETON_GPU_PROVIDERS`` list until the
account/image is known to support a Docker host (e.g. bare metal). Injectable
``transport`` for offline tests.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from gpu.errors import DestroyError, ProvisionError
from gpu.keys import ensure_durable_keypair
from gpu.registry import _state_dir
from gpu.ssh import exec as ssh_exec
from gpu.types import Offer, Pod, PodSpec, SshTarget

logger = logging.getLogger(__name__)

API_BASE = "https://api.runpod.io"
RUNPOD_DASHBOARD = "https://www.console.runpod.io/pods"
READY_TIMEOUT_S = 720
READY_POLL_S = 10
HTTP_TIMEOUT_S = 60.0
# Official image starts sshd when startSsh=true; nested Docker is not guaranteed.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# Container disk holds Docker's data root; engine image pulls need headroom.
MIN_CONTAINER_DISK_GB = 200

Transport = Callable[..., Any]


def _normalize_gpu_type(raw: str) -> str:
    t = (raw or "").strip()
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "GeForce "):
        if t.upper().startswith(prefix.upper()):
            return t[len(prefix) :]
    return t


def _gpu_type_matches(have: str, want: str) -> bool:
    """Exact or whole-token match (H200 vs 'H200 NVL'); not bare substring."""
    h = (have or "").upper().strip()
    w = (want or "").upper().strip()
    if not w:
        return True
    if h == w:
        return True
    tokens = [t for t in re.split(r"[^A-Z0-9]+", h) if t]
    return w in tokens


def _key_body(pub_key: str) -> str:
    parts = pub_key.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else pub_key.strip()


def _volume_gib() -> int:
    try:
        import config as _cfg

        return max(10, int(getattr(_cfg, "GPU_VOLUME_GIB", 250)))
    except Exception:  # noqa: BLE001
        return 250


def _container_disk_gb() -> int:
    """Ephemeral disk for the OS + Docker data root (engine image pulls)."""
    return max(MIN_CONTAINER_DISK_GB, _volume_gib())


def _runpod_image() -> str:
    try:
        import config as _cfg

        return str(getattr(_cfg, "RUNPOD_IMAGE", "") or DEFAULT_IMAGE)
    except Exception:  # noqa: BLE001
        return DEFAULT_IMAGE


def _runpod_cloud() -> str:
    try:
        import config as _cfg

        return str(getattr(_cfg, "RUNPOD_CLOUD", "ANY") or "ANY").strip().upper()
    except Exception:  # noqa: BLE001
        return "ANY"


def _as_item_list(data: Any, *keys: str) -> list[Any]:
    """Normalize list-or-wrapped-object JSON without calling ``.get`` on a list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            items = data.get(key)
            if isinstance(items, list):
                return items
    return []


def _default_transport(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict | None = None,
    params: dict | None = None,
    timeout: float = HTTP_TIMEOUT_S,
) -> requests.Response:
    return requests.request(
        method,
        url,
        headers=headers,
        json=json,
        params=params,
        timeout=timeout,
    )


def _parse_direct_ssh(info: dict[str, Any]) -> SshTarget | None:
    ssh = info.get("ssh") if isinstance(info.get("ssh"), dict) else {}
    direct = ssh.get("direct") if isinstance(ssh, dict) else None
    if not isinstance(direct, dict):
        return None
    host = str(direct.get("host") or "").strip()
    user = str(direct.get("username") or direct.get("user") or "").strip()
    try:
        port = int(direct.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    if not host or not user or port <= 0:
        return None
    return SshTarget(host=host, port=port, user=user)


class RunpodProvider:
    name = "runpod"

    def __init__(
        self,
        api_key: str,
        *,
        state_dir: Path | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] | None = None,
        ready_timeout_s: int = READY_TIMEOUT_S,
    ) -> None:
        if not api_key:
            raise ProvisionError("PARETON_RUNPOD_API_KEY is not set")
        self._api_key = api_key
        self._state_dir = _state_dir(state_dir)
        self._transport = transport or _default_transport
        self._sleep = sleep or time.sleep
        self._ready_timeout_s = ready_timeout_s
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._key_path = ensure_durable_keypair(self._state_dir)[0]

    def _req(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        timeout: float = HTTP_TIMEOUT_S,
        raise_for_status: bool = True,
    ) -> Any:
        url = path if path.startswith("http") else f"{API_BASE}{path}"
        resp = self._transport(
            method,
            url,
            headers=self._headers,
            json=json,
            params=params,
            timeout=timeout,
        )
        if raise_for_status and not getattr(resp, "ok", False):
            status = getattr(resp, "status_code", "?")
            text = getattr(resp, "text", "") or ""
            raise ProvisionError(
                f"Runpod {method} {path} failed HTTP {status}: {text[:300]}"
            )
        if not getattr(resp, "content", b""):
            return None
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return None

    def _get(self, path: str, **kwargs: Any) -> Any:
        return self._req("GET", path, **kwargs)

    def _post(self, path: str, json: dict | None = None, **kwargs: Any) -> Any:
        return self._req("POST", path, json=json, **kwargs)

    def _put(self, path: str, json: dict | None = None, **kwargs: Any) -> Any:
        return self._req("PUT", path, json=json, **kwargs)

    def _delete(self, path: str, **kwargs: Any) -> Any:
        return self._req("DELETE", path, **kwargs)

    def _ensure_ssh_key(self, ssh_public_key: str) -> None:
        """Merge Pareton's durable pubkey into the account key set.

        Runpod ``PUT /v2/account/ssh-keys`` is a full replace. Read first and
        keep any keys the operator already registered in the console.
        """
        local_body = _key_body(ssh_public_key)
        current = self._get("/v2/account/ssh-keys") or {}
        existing = list(current.get("keys") or []) if isinstance(current, dict) else []
        kept: list[str] = []
        seen: set[str] = set()
        for key in existing:
            line = str(key or "").strip()
            if not line:
                continue
            body = _key_body(line)
            if body in seen:
                continue
            seen.add(body)
            kept.append(line)
        if local_body in seen:
            logger.info("SSH key already registered with Runpod")
            return
        kept.append(ssh_public_key.strip())
        self._put("/v2/account/ssh-keys", json={"keys": kept})
        logger.info(
            "Registered Pareton SSH key with Runpod (merged; %d keys)", len(kept)
        )

    def search(self, spec: PodSpec) -> list[Offer]:
        want = max(1, int(spec.gpu_count or 1))
        cloud_pref = _runpod_cloud()
        params: dict[str, str] = {
            "include": "AVAILABILITY",
            "product": "POD",
            "count": str(want),
        }
        if cloud_pref in ("SECURE", "COMMUNITY"):
            params["cloud"] = cloud_pref
        resp = self._get("/v2/catalog/gpus", params=params)
        items = _as_item_list(resp, "gpus")
        out: list[Offer] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            gpu_id = str(item.get("id") or "")
            if not gpu_id:
                continue
            gpu_type = _normalize_gpu_type(gpu_id)
            if spec.gpu_type and not _gpu_type_matches(gpu_type, spec.gpu_type):
                continue
            availability = str(item.get("availability") or "").upper()
            if availability == "NONE":
                continue
            max_count = (
                item.get("maxCount") if isinstance(item.get("maxCount"), dict) else {}
            )
            price = item.get("price") if isinstance(item.get("price"), dict) else {}
            clouds: list[tuple[str, float]] = []
            for cloud_name, flag_key, price_key in (
                ("COMMUNITY", "community", "community"),
                ("SECURE", "secure", "secure"),
            ):
                if cloud_pref not in ("ANY", cloud_name):
                    continue
                if not bool(item.get(flag_key)):
                    continue
                try:
                    max_n = int(max_count.get(price_key) or 0)
                except (TypeError, ValueError):
                    max_n = 0
                if max_n and want > max_n:
                    continue
                try:
                    usd = float(price.get(price_key) or 0)
                except (TypeError, ValueError):
                    continue
                if usd <= 0:
                    continue
                # Catalog price is per-GPU; Pareton caps are for the whole pod.
                hourly_cents = round(usd * 100 * want)
                if hourly_cents > spec.max_hourly_cents:
                    continue
                clouds.append((cloud_name, usd))
            if not clouds:
                continue
            # Prefer cheaper cloud first (community before secure when ANY).
            clouds.sort(key=lambda c: c[1])
            cloud_name, usd = clouds[0]
            hourly_cents = round(usd * 100 * want)
            data_centers = []
            for dc in item.get("dataCenters") or []:
                if not isinstance(dc, dict):
                    continue
                dc_id = str(dc.get("id") or "")
                dc_avail = str(dc.get("availability") or "").upper()
                if dc_id and dc_avail != "NONE":
                    data_centers.append(dc_id)
            out.append(
                Offer(
                    provider=self.name,
                    instance_id=f"{cloud_name}:{gpu_id}",
                    description=str(item.get("name") or gpu_id),
                    hourly_price_cents=hourly_cents,
                    gpu_count=want,
                    gpu_type=gpu_type,
                    raw={
                        "gpu_id": gpu_id,
                        "cloud": cloud_name,
                        "data_center_ids": data_centers,
                        "availability": availability,
                    },
                )
            )
        out.sort(key=lambda o: (o.hourly_price_cents, o.gpu_type))
        return out

    def _require_docker_host(self, pod: Pod) -> None:
        """Fail closed if the pod cannot run nested Docker for bench engines."""
        check = ssh_exec(
            pod,
            "docker info >/dev/null 2>&1",
            timeout_s=60.0,
            check=False,
            state_dir=self._state_dir,
        )
        if check.exit_code == 0:
            return
        # Mirror bootstrap's install path once; still fails without privileged/DinD.
        install = ssh_exec(
            pod,
            "command -v docker >/dev/null 2>&1 || "
            "curl -fsSL https://get.docker.com | sh; "
            "mkdir -p /etc/docker && "
            'printf \'{"data-root":"/workspace/docker"}\\n\' > /etc/docker/daemon.json; '
            "(command -v systemctl >/dev/null && systemctl restart docker) || "
            "(service docker restart) || "
            "(dockerd --data-root /workspace/docker >/tmp/dockerd.log 2>&1 &) ; "
            "for i in 1 2 3 4 5 6 7 8 9 10; do "
            "docker info >/dev/null 2>&1 && exit 0; sleep 3; done; exit 1",
            timeout_s=300.0,
            check=False,
            state_dir=self._state_dir,
        )
        if install.exit_code == 0:
            logger.info("Docker host ready on Runpod pod %s", pod.pod_id)
            return
        raise ProvisionError(
            f"Runpod pod {pod.pod_id} cannot run Docker "
            "(stock shared pods lack privileged/DinD; use a Docker-capable "
            "image/account or omit runpod from PARETON_GPU_PROVIDERS)"
        )

    def provision(self, offer: Offer, *, name: str, ssh_public_key: str) -> Pod:
        raw = offer.raw or {}
        gpu_id = str(raw.get("gpu_id") or "")
        cloud = str(raw.get("cloud") or "SECURE")
        if not gpu_id and ":" in offer.instance_id:
            cloud, gpu_id = offer.instance_id.split(":", 1)
        if not gpu_id:
            raise ProvisionError("Runpod offer missing gpu_id")

        self._ensure_ssh_key(ssh_public_key)
        volume_gib = _volume_gib()
        disk_gb = _container_disk_gb()
        body: dict[str, Any] = {
            "name": name,
            "image": _runpod_image(),
            "gpu": {"id": gpu_id, "count": int(offer.gpu_count or 1)},
            "cloud": cloud,
            "startSsh": True,
            "startJupyter": False,
            "ports": ["22/tcp"],
            "disk": disk_gb,
            "mounts": {
                "persistent": {
                    "size": volume_gib,
                    "path": "/workspace",
                }
            },
        }
        dc_ids = [str(x) for x in (raw.get("data_center_ids") or []) if str(x)]
        if dc_ids:
            body["dataCenterIds"] = dc_ids[:5]

        pod_id: str | None = None
        try:
            created = self._post("/v2/pods", json=body)
            if not isinstance(created, dict):
                raise ProvisionError("Runpod create returned non-object")
            pod_id = str(created.get("id") or "")
            if not pod_id:
                raise ProvisionError("Runpod create returned no pod id")
            logger.info(
                "Runpod pod %s created (gpu=%sx%s cloud=%s disk=%dG volume=%dG)",
                pod_id,
                offer.gpu_count,
                gpu_id,
                cloud,
                disk_gb,
                volume_gib,
            )
            logger.info("Dashboard: %s/%s", RUNPOD_DASHBOARD, pod_id)
            pod = Pod(
                provider=self.name,
                pod_id=pod_id,
                name=name,
                ssh=SshTarget(host="", port=22, user="root"),
                key_path=self._key_path,
                hourly_price_cents=offer.hourly_price_cents,
                created_utc=datetime.now(timezone.utc),
                ttl_hours=0.0,
                raw={
                    "volume_uid": "",
                    "volume_name": name,
                    "gpu_id": gpu_id,
                    "cloud": cloud,
                    "dashboard": f"{RUNPOD_DASHBOARD}/{pod_id}",
                },
            )
            ready = self._wait_ready(pod)
            self._require_docker_host(ready)
            return ready
        except Exception:
            if pod_id:
                try:
                    self._teardown_pod(pod_id)
                except Exception:
                    logger.exception("abort cleanup failed for Runpod pod %s", pod_id)
            raise

    def _wait_ready(self, pod: Pod, timeout_s: int | None = None) -> Pod:
        timeout = self._ready_timeout_s if timeout_s is None else timeout_s
        started = time.monotonic()
        deadline = started + timeout
        last_log = started - 60.0
        last_status = ""
        while time.monotonic() < deadline:
            info = self._get(f"/v2/pods/{pod.pod_id}") or {}
            if not isinstance(info, dict):
                raise ProvisionError(f"Runpod pod {pod.pod_id} info: expected object")
            status = str(info.get("status") or "").upper()
            last_status = status
            now = time.monotonic()
            if now - last_log >= 60.0:
                logger.info(
                    "Runpod pod %s status=%s elapsed=%ds remaining=%ds",
                    pod.pod_id,
                    status or "?",
                    int(now - started),
                    max(0, int(deadline - now)),
                )
                last_log = now
            if status in ("ERROR", "EXITED", "TERMINATED"):
                raise ProvisionError(f"Runpod pod {pod.pod_id} entered {status}")
            if status == "RUNNING":
                ssh = _parse_direct_ssh(info)
                if ssh is None:
                    self._sleep(READY_POLL_S)
                    continue
                pod.ssh = ssh
                pod.raw = {
                    **(pod.raw or {}),
                    "data_center_id": info.get("dataCenterId"),
                    "status": status,
                }
                return pod
            self._sleep(READY_POLL_S)
        raise ProvisionError(
            f"Runpod pod {pod.pod_id} not ready after {timeout}s "
            f"(last status={last_status or '?'}; need RUNNING + ssh.direct)"
        )

    def destroy(self, pod: Pod) -> None:
        try:
            self._teardown_pod(pod.pod_id)
        except DestroyError:
            raise
        except Exception as exc:
            raise DestroyError(str(exc)) from exc

    def _teardown_pod(self, pod_id: str) -> None:
        try:
            self._delete(f"/v2/pods/{pod_id}")
            logger.info("Runpod pod %s terminated", pod_id)
        except ProvisionError as exc:
            if "404" in str(exc):
                logger.info("Runpod pod %s already gone", pod_id)
                return
            raise DestroyError(str(exc)) from exc

    def list_pods(self) -> list[Pod]:
        data = self._get("/v2/pods")
        items = _as_item_list(data, "pods")
        out: list[Pod] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("id") or "")
            if not uid:
                continue
            status = str(item.get("status") or "").upper()
            if status in ("TERMINATED",):
                continue
            ssh = _parse_direct_ssh(item) or SshTarget(host="", port=22, user="root")
            out.append(
                Pod(
                    provider=self.name,
                    pod_id=uid,
                    name=str(item.get("name") or ""),
                    ssh=ssh,
                    key_path=self._key_path,
                    hourly_price_cents=0,
                    created_utc=datetime.now(timezone.utc),
                    ttl_hours=0.0,
                    raw={"status": status, "volume_uid": ""},
                )
            )
        return out

    def list_volumes(self) -> list[dict[str, Any]]:
        # Host-local persistent mounts die with the pod; network volumes are
        # listed for reap visibility if an operator created any manually.
        data = self._get("/v2/network-volumes")
        items = _as_item_list(data, "networkVolumes", "volumes")
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "id": str(item.get("id", "")),
                    "name": str(item.get("name", "")),
                    "raw": item,
                }
            )
        return out


__all__ = [
    "API_BASE",
    "MIN_CONTAINER_DISK_GB",
    "RUNPOD_DASHBOARD",
    "RunpodProvider",
    "_as_item_list",
    "_container_disk_gb",
    "_gpu_type_matches",
    "_normalize_gpu_type",
    "_parse_direct_ssh",
]
