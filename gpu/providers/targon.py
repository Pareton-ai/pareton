"""Targon GPU cloud provider (REST tha/v2)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from gpu.errors import DestroyError, ProvisionError
from gpu.keys import ensure_durable_keypair
from gpu.registry import _state_dir
from gpu.types import Offer, Pod, PodSpec, SshTarget

logger = logging.getLogger(__name__)

API_BASE = "https://api.targon.com/tha/v2"
TARGON_SSH_HOST = "ssh.deployments.targon.com"
TARGON_DASHBOARD = "https://targon.com/rentals"
VOLUME_RESOURCE_NAME = "storage-rentals"
VOLUME_MOUNT_PATH = "/workspace"
VOLUME_READY_TIMEOUT_S = 300
READY_TIMEOUT_S = 600
HTTP_TIMEOUT_S = 30.0

Transport = Callable[..., Any]


def _normalize_gpu_type(raw: str) -> str:
    for prefix in ("NVIDIA-GeForce-", "NVIDIA-"):
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw


def _volume_gib() -> int:
    try:
        import config as _cfg

        return int(getattr(_cfg, "GPU_VOLUME_GIB", 250))
    except Exception:  # noqa: BLE001
        return 250


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


class TargonProvider:
    name = "targon"

    def __init__(
        self,
        api_key: str,
        *,
        state_dir: Path | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise ProvisionError("PARETON_TARGON_API_KEY is not set")
        self._api_key = api_key
        self._state_dir = _state_dir(state_dir)
        self._transport = transport or _default_transport
        self._sleep = sleep or time.sleep
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._ssh_key_uid: str | None = None
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
                f"Targon {method} {path} failed HTTP {status}: {text[:300]}"
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

    def _delete(self, path: str, **kwargs: Any) -> Any:
        return self._req("DELETE", path, **kwargs)

    @staticmethod
    def _key_fingerprint(pub_key_line: str) -> str:
        parts = pub_key_line.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else pub_key_line.strip()

    def _ensure_ssh_key(self, ssh_public_key: str) -> str:
        if self._ssh_key_uid:
            return self._ssh_key_uid
        local_fp = self._key_fingerprint(ssh_public_key)
        existing = self._get("/ssh-keys") or {}
        for key in existing.get("items", []) if isinstance(existing, dict) else []:
            remote_fp = self._key_fingerprint(key.get("public_key_raw", ""))
            if remote_fp == local_fp:
                self._ssh_key_uid = key["uid"]
                logger.info("SSH key already registered (uid=%s)", self._ssh_key_uid)
                return self._ssh_key_uid
        resp = self._post(
            "/ssh-keys",
            json={"name": "pareton-gpu", "ssh_key": ssh_public_key},
        )
        self._ssh_key_uid = resp["uid"]
        logger.info("Registered SSH key with Targon: %s", self._ssh_key_uid)
        return self._ssh_key_uid

    def search(self, spec: PodSpec) -> list[Offer]:
        resp = self._transport(
            "GET",
            f"{API_BASE}/inventory",
            headers={"Authorization": f"Bearer {self._api_key}"},
            params={"type": "rental", "gpu": "true"},
            timeout=HTTP_TIMEOUT_S,
        )
        if not getattr(resp, "ok", False):
            raise ProvisionError(
                f"Targon inventory failed HTTP {getattr(resp, 'status_code', '?')}"
            )
        items = resp.json()
        out: list[Offer] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if int(item.get("available", 0) or 0) <= 0:
                continue
            spec_d = item.get("spec") or {}
            gpu_type = _normalize_gpu_type(str(spec_d.get("gpu_type", "")))
            gpu_count = int(spec_d.get("gpu_count", 0) or 0)
            if gpu_count <= 0:
                continue
            if spec.gpu_type and gpu_type.upper() != spec.gpu_type.upper():
                continue
            if spec.gpu_count and gpu_count < spec.gpu_count:
                continue
            price_dollars = float(item.get("cost_per_hour", 0) or 0)
            price_cents = int(round(price_dollars * 100))
            if price_cents > spec.max_hourly_cents:
                continue
            out.append(
                Offer(
                    provider=self.name,
                    instance_id=str(item.get("name", "")),
                    description=str(item.get("display_name", "")),
                    hourly_price_cents=price_cents,
                    gpu_count=gpu_count,
                    gpu_type=gpu_type,
                    raw=item,
                )
            )
        out.sort(key=lambda o: o.hourly_price_cents)
        return out

    def _create_volume(self, name: str) -> str:
        size_gib = _volume_gib()
        size_mb = size_gib * 1024
        resp = self._post(
            "/volumes",
            json={
                "name": name,
                "size_in_mb": size_mb,
                "resource_name": VOLUME_RESOURCE_NAME,
            },
        )
        volume_uid = resp["uid"]
        logger.info(
            "Created Targon volume %s (%d GiB, name=%s)", volume_uid, size_gib, name
        )
        logger.warning(
            "Targon volume storage is billed separately from GPU hourly cost "
            "(size=%d GiB)",
            size_gib,
        )
        return volume_uid

    def _wait_volume_ready(
        self, volume_uid: str, timeout_s: int = VOLUME_READY_TIMEOUT_S
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self._get(f"/volumes/{volume_uid}/state")
            status = str(state.get("status", "")).upper()
            if status in ("READY", "REGISTERED"):
                return
            if status in ("FAILED", "ERROR", "DELETING"):
                raise ProvisionError(
                    f"Targon volume {volume_uid} entered {status}: "
                    f"{state.get('message', '')}"
                )
            self._sleep(5)
        raise ProvisionError(f"Targon volume {volume_uid} not ready after {timeout_s}s")

    def _deploy_workload(self, workload_uid: str) -> None:
        url = f"{API_BASE}/workloads/{workload_uid}/deploy"
        last_status = 0
        for attempt in range(1, 4):
            resp = self._transport("POST", url, headers=self._headers, timeout=60.0)
            last_status = int(getattr(resp, "status_code", 0) or 0)
            if getattr(resp, "ok", False):
                return
            if last_status in (502, 503, 504) and attempt < 3:
                logger.warning(
                    "Targon deploy %s HTTP %s (attempt %d/3); retrying",
                    workload_uid,
                    last_status,
                    attempt,
                )
                self._sleep(5)
                continue
            break
        state = self._get(f"/workloads/{workload_uid}/state")
        status = str(state.get("status", "")).upper()
        if status in ("DEPLOYING", "PROVISIONING", "RUNNING"):
            logger.warning(
                "Targon deploy HTTP failed for %s but state is %s; continuing",
                workload_uid,
                status,
            )
            return
        try:
            self._delete(f"/workloads/{workload_uid}")
        except Exception:  # noqa: BLE001
            logger.exception("cleanup after deploy failure for %s", workload_uid)
        raise ProvisionError(
            f"Targon deploy failed for {workload_uid} (HTTP {last_status})"
        )

    def _abort_rent(self, workload_uid: str | None, volume_uid: str) -> None:
        if workload_uid:
            try:
                self._teardown_workload(workload_uid)
            except Exception:  # noqa: BLE001
                logger.exception("abort workload cleanup failed for %s", workload_uid)
        try:
            self._teardown_volume(volume_uid, raise_on_fail=False)
        except Exception:  # noqa: BLE001
            logger.exception("abort volume cleanup failed for %s", volume_uid)

    def _teardown_workload(self, workload_uid: str) -> None:
        try:
            self._delete(f"/workloads/{workload_uid}")
            logger.info("Targon workload %s deleted", workload_uid)
        except ProvisionError as exc:
            if "404" in str(exc):
                logger.info("Targon workload %s already deleted", workload_uid)
                return
            raise DestroyError(str(exc)) from exc

    def _teardown_volume(self, volume_uid: str, *, raise_on_fail: bool = True) -> None:
        deadline = time.monotonic() + 300
        last_err = ""
        while time.monotonic() < deadline:
            try:
                self._post(f"/volumes/{volume_uid}/delete")
                logger.info("Targon volume %s deleted", volume_uid)
                return
            except ProvisionError as exc:
                msg = str(exc)
                last_err = msg
                if "404" in msg:
                    logger.info("Targon volume %s already deleted", volume_uid)
                    return
                if any(code in msg for code in ("409", "423", "503")):
                    self._sleep(5)
                    continue
                break
        if raise_on_fail:
            raise DestroyError(
                f"Targon volume {volume_uid} still not deleted: {last_err}"
            )
        logger.error("Timed out deleting Targon volume %s", volume_uid)

    def provision(self, offer: Offer, *, name: str, ssh_public_key: str) -> Pod:
        ssh_key_uid = self._ensure_ssh_key(ssh_public_key)
        volume_name = name  # same TTL pattern
        volume_uid = self._create_volume(volume_name)
        workload_uid: str | None = None
        try:
            self._wait_volume_ready(volume_uid)
            body: dict[str, Any] = {
                "name": name,
                "image": "ghcr.io/manifold-inc/ubuntu-systemd-docker:v3",
                "resource_name": offer.instance_id,
                "type": "RENTAL",
                "ports": [{"port": 22, "protocol": "TCP", "routing": "DIRECT"}],
                "ssh_keys": [ssh_key_uid],
                "volumes": [{"uid": volume_uid, "mount_path": VOLUME_MOUNT_PATH}],
            }
            workload = self._post("/workloads", json=body)
            workload_uid = workload["uid"]
            self._deploy_workload(workload_uid)
            pod = self._wait_ready(
                Pod(
                    provider=self.name,
                    pod_id=workload_uid,
                    name=name,
                    ssh=SshTarget(host=TARGON_SSH_HOST, port=22, user=workload_uid),
                    key_path=self._key_path,
                    hourly_price_cents=offer.hourly_price_cents,
                    created_utc=datetime.now(timezone.utc),
                    ttl_hours=0.0,  # filled by caller from name
                    raw={
                        "volume_uid": volume_uid,
                        "volume_name": volume_name,
                        "resource_name": offer.instance_id,
                        "dashboard": f"{TARGON_DASHBOARD}/{workload_uid}",
                    },
                )
            )
            return pod
        except Exception:
            self._abort_rent(workload_uid, volume_uid)
            raise

    def _wait_ready(self, pod: Pod, timeout_s: int = READY_TIMEOUT_S) -> Pod:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self._get(f"/workloads/{pod.pod_id}/state")
            status = str(state.get("status", "")).upper()
            if status == "RUNNING":
                pod.ssh = SshTarget(host=TARGON_SSH_HOST, port=22, user=pod.pod_id)
                return pod
            if status in ("FAILED", "ERROR", "TERMINATED"):
                raise ProvisionError(
                    f"Targon workload {pod.pod_id} entered {status}: "
                    f"{state.get('message', '')}"
                )
            self._sleep(5)
        raise ProvisionError(
            f"Targon workload {pod.pod_id} not ready after {timeout_s}s"
        )

    def destroy(self, pod: Pod) -> None:
        volume_uid = str((pod.raw or {}).get("volume_uid") or "")
        workload_err: Exception | None = None
        try:
            self._teardown_workload(pod.pod_id)
        except DestroyError as exc:
            workload_err = exc
        if volume_uid:
            try:
                self._teardown_volume(volume_uid, raise_on_fail=True)
            except DestroyError:
                raise
        if workload_err is not None:
            raise workload_err

    def list_pods(self) -> list[Pod]:
        data = self._get("/workloads") or {}
        items = data.get("items", data if isinstance(data, list) else [])
        out: list[Pod] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            uid = str(item.get("uid", ""))
            if not uid:
                continue
            out.append(
                Pod(
                    provider=self.name,
                    pod_id=uid,
                    name=name,
                    ssh=SshTarget(host=TARGON_SSH_HOST, port=22, user=uid),
                    key_path=self._key_path,
                    hourly_price_cents=0,
                    created_utc=datetime.now(timezone.utc),
                    ttl_hours=0.0,
                    raw={
                        "volume_uid": "",
                        "status": item.get("status"),
                    },
                )
            )
        return out

    def list_volumes(self) -> list[dict[str, Any]]:
        data = self._get("/volumes") or {}
        items = data.get("items", data if isinstance(data, list) else [])
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "id": str(item.get("uid", "")),
                    "name": str(item.get("name", "")),
                    "raw": item,
                }
            )
        return out


# Re-export helper for tests that need name encoding without circular imports.
__all__ = ["TargonProvider", "API_BASE", "TARGON_SSH_HOST", "_normalize_gpu_type"]
