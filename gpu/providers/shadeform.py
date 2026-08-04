"""Shadeform GPU cloud provider (REST v1).

Implements Pareton's Provider protocol against the Shadeform control
plane. Uses ``gpu.ssh`` for workspace mount (no paramiko). Non-root SSH
user is handled by bootstrap (sudo + docker.sock ACL), not here.
"""

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
from gpu.ssh import exec as ssh_exec
from gpu.types import Offer, Pod, PodSpec, SshTarget

logger = logging.getLogger(__name__)

API_BASE = "https://api.shadeform.ai/v1"
SHADEFORM_DASHBOARD = "https://platform.shadeform.ai/instances"
SSH_KEY_NAME = "pareton-gpu"
_INSTANCE_ID_SEP = ":"
READY_TIMEOUT_S = 720
HTTP_TIMEOUT_S = 60.0
MOUNT_POLL_S = 15

Transport = Callable[..., Any]


def _parse_instance_id(instance_id: str) -> tuple[str, str, str]:
    parts = instance_id.split(_INSTANCE_ID_SEP, 2)
    if len(parts) != 3:
        raise ProvisionError(f"invalid Shadeform offer id: {instance_id!r}")
    return parts[0], parts[1], parts[2]


def _format_instance_id(cloud: str, region: str, shade_instance_type: str) -> str:
    return f"{cloud}{_INSTANCE_ID_SEP}{region}{_INSTANCE_ID_SEP}{shade_instance_type}"


def _pick_os(os_options: list[str]) -> str | None:
    for pref in ("ubuntu22.04", "ubuntu24.04", "ubuntu20.04"):
        for opt in os_options:
            if pref in opt.lower():
                return opt
    return os_options[0] if os_options else None


def _normalize_gpu_type(raw: str) -> str:
    for prefix in ("NVIDIA-GeForce-", "NVIDIA-", "GeForce-"):
        if raw.upper().startswith(prefix.upper()):
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


def _mount_workspace_script(*, volume_gib: int) -> str:
    """Bind/mount block storage at /workspace."""
    min_kb = int(volume_gib * 1024 * 1024 * 0.9)
    return f"""set -euo pipefail
MIN_KB={min_kb}

workspace_kb() {{
  df -Pk /workspace 2>/dev/null | awk 'NR==2 {{print $2}}'
}}

workspace_ready() {{
  local kb
  kb="$(workspace_kb)"
  [ -n "$kb" ] && [ "$kb" -ge "$MIN_KB" ]
}}

sudo mkdir -p /workspace

if workspace_ready; then
  echo "OK: /workspace ready ($(workspace_kb)KB on backing fs)"
  df -h /workspace
  exit 0
fi

is_system_mount() {{
  case "$1" in
    /|/boot|/boot/efi|/var/lib/docker/*|/snap/*) return 0 ;;
  esac
  return 1
}}

bind_to_workspace() {{
  local src="$1"
  if ! mountpoint -q "$src" 2>/dev/null; then
    return 1
  fi
  sudo mount --bind "$src" /workspace
  sudo chmod 1777 /workspace
  echo "OK: bind-mounted $src to /workspace"
  df -h /workspace
}}

# Prefer Shadeform auto-mount (or any large non-system mount). Never mkfs —
# formatting an unlabeled blockdev can wipe the wrong disk.
best_mp=""
best_mp_kb=0
while read -r mp size_kb; do
  if is_system_mount "$mp"; then
    continue
  fi
  if [ "$size_kb" -lt "$MIN_KB" ]; then
    continue
  fi
  if [ "$size_kb" -gt "$best_mp_kb" ]; then
    best_mp_kb=$size_kb
    best_mp="$mp"
  fi
done < <(df -Pk | awk 'NR>1 {{print $6, $2}}')

if [ -n "$best_mp" ]; then
  bind_to_workspace "$best_mp"
  exit 0
fi

echo "ERROR: /workspace not ready and no mountable volume >= ${{MIN_KB}}KB"
echo "Shadeform volume_mount.auto should attach storage; refusing raw mkfs."
lsblk
df -h
exit 1
"""


class ShadeformProvider:
    name = "shadeform"

    def __init__(
        self,
        api_key: str,
        *,
        state_dir: Path | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise ProvisionError("PARETON_SHADEFORM_API_KEY is not set")
        self._api_key = api_key
        self._state_dir = _state_dir(state_dir)
        self._transport = transport or _default_transport
        self._sleep = sleep or time.sleep
        self._headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        self._ssh_key_id: str | None = None
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
                f"Shadeform {method} {path} failed HTTP {status}: {text[:300]}"
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

    @staticmethod
    def _key_body(pub_key: str) -> str:
        parts = pub_key.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else pub_key.strip()

    def _ensure_ssh_key(self, ssh_public_key: str) -> str:
        if self._ssh_key_id:
            return self._ssh_key_id
        local_body = self._key_body(ssh_public_key)
        resp = self._get("/sshkeys") or {}
        for key in resp.get("ssh_keys", []) if isinstance(resp, dict) else []:
            if self._key_body(str(key.get("public_key", ""))) == local_body:
                self._ssh_key_id = str(key["id"])
                logger.info(
                    "SSH key already registered with Shadeform (id=%s)",
                    self._ssh_key_id,
                )
                return self._ssh_key_id
        created = self._post(
            "/sshkeys/add",
            json={"name": SSH_KEY_NAME, "public_key": ssh_public_key},
        )
        self._ssh_key_id = str(created["id"])
        logger.info("Registered SSH key with Shadeform: %s", self._ssh_key_id)
        return self._ssh_key_id

    def search(self, spec: PodSpec) -> list[Offer]:
        want = max(1, int(spec.gpu_count or 1))
        resp = self._get(
            "/instances/types",
            params={
                "num_gpus": str(want),
                "available": "true",
                "sort": "price",
            },
        )
        items = (resp or {}).get("instance_types", []) if isinstance(resp, dict) else []
        out: list[Offer] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cfg = item.get("configuration") or {}
            gpu_type = _normalize_gpu_type(str(cfg.get("gpu_type", "")))
            gpu_count = int(cfg.get("num_gpus", 0) or 0)
            # Shadeform SKUs are discrete; require exact count (not >=).
            if gpu_count != want:
                continue
            if spec.gpu_type and gpu_type.upper() != spec.gpu_type.upper():
                continue
            deployment = (item.get("deployment_type") or "").lower()
            if deployment not in ("vm", "baremetal"):
                continue
            # Shadeform hourly_price is already cents.
            price_cents = int(item.get("hourly_price", 0) or 0)
            if price_cents > spec.max_hourly_cents:
                continue
            for avail in item.get("availability", []) or []:
                if not isinstance(avail, dict) or not avail.get("available"):
                    continue
                region = str(avail.get("region", "") or "")
                cloud = str(item.get("cloud", "") or "")
                shade_type = str(item.get("shade_instance_type", "") or "")
                if not cloud or not region or not shade_type:
                    continue
                display = str(avail.get("display_name") or shade_type)
                out.append(
                    Offer(
                        provider=self.name,
                        instance_id=_format_instance_id(cloud, region, shade_type),
                        description=f"{display} ({shade_type})",
                        hourly_price_cents=price_cents,
                        gpu_count=gpu_count,
                        gpu_type=gpu_type,
                        raw={
                            "cloud": cloud,
                            "region": region,
                            "shade_instance_type": shade_type,
                            "os_options": list(cfg.get("os_options") or []),
                        },
                    )
                )
        out.sort(key=lambda o: (o.hourly_price_cents, o.gpu_count))
        return out

    def _create_volume(self, *, cloud: str, region: str, name: str) -> str:
        size_gib = _volume_gib()
        resp = self._post(
            "/volumes/create",
            json={
                "cloud": cloud,
                "region": region,
                "size_in_gb": size_gib,
                "name": name,
            },
        )
        volume_id = str(resp["id"])
        logger.info(
            "Created Shadeform volume %s (%d GiB, name=%s cloud=%s region=%s)",
            volume_id,
            size_gib,
            name,
            cloud,
            region,
        )
        logger.warning(
            "Shadeform volume storage is billed separately from GPU hourly cost "
            "(size=%d GiB)",
            size_gib,
        )
        return volume_id

    def _abort_rent(self, instance_id: str | None, volume_id: str) -> None:
        if instance_id:
            try:
                self._teardown_instance(instance_id)
            except Exception:  # noqa: BLE001
                logger.exception("abort instance cleanup failed for %s", instance_id)
        try:
            self._teardown_volume(volume_id, raise_on_fail=False)
        except Exception:  # noqa: BLE001
            logger.exception("abort volume cleanup failed for %s", volume_id)

    def provision(self, offer: Offer, *, name: str, ssh_public_key: str) -> Pod:
        raw = offer.raw or {}
        cloud = str(raw.get("cloud") or "")
        region = str(raw.get("region") or "")
        shade_type = str(raw.get("shade_instance_type") or "")
        if not cloud or not region or not shade_type:
            cloud, region, shade_type = _parse_instance_id(offer.instance_id)

        ssh_key_id = self._ensure_ssh_key(ssh_public_key)
        volume_id = self._create_volume(cloud=cloud, region=region, name=name)
        instance_id: str | None = None
        try:
            os_name = _pick_os(list(raw.get("os_options") or []))
            body: dict[str, Any] = {
                "cloud": cloud,
                "region": region,
                "shade_instance_type": shade_type,
                "shade_cloud": True,
                "name": name,
                "volume_ids": [volume_id],
                "ssh_key_id": ssh_key_id,
                "volume_mount": {"auto": True},
            }
            if os_name:
                body["os"] = os_name
            created = self._post("/instances/create", json=body)
            instance_id = str(created["id"])
            logger.info(
                "Shadeform instance %s created (cloud=%s region=%s type=%s volume=%s)",
                instance_id,
                cloud,
                region,
                shade_type,
                volume_id,
            )
            logger.info("Dashboard: %s/%s", SHADEFORM_DASHBOARD, instance_id)
            pod = Pod(
                provider=self.name,
                pod_id=instance_id,
                name=name,
                ssh=SshTarget(host="", port=22, user="shadeform"),
                key_path=self._key_path,
                hourly_price_cents=offer.hourly_price_cents,
                created_utc=datetime.now(timezone.utc),
                ttl_hours=0.0,
                raw={
                    "volume_uid": volume_id,
                    "volume_name": name,
                    "cloud": cloud,
                    "region": region,
                    "shade_instance_type": shade_type,
                    "dashboard": f"{SHADEFORM_DASHBOARD}/{instance_id}",
                },
            )
            return self._wait_ready(pod)
        except Exception:
            self._abort_rent(instance_id, volume_id)
            raise

    def _wait_ready(self, pod: Pod, timeout_s: int = READY_TIMEOUT_S) -> Pod:
        started = time.monotonic()
        deadline = started + timeout_s
        last_log = started - 60.0
        last_status = ""
        mount_attempted = False
        while time.monotonic() < deadline:
            info = self._get(f"/instances/{pod.pod_id}/info") or {}
            if not isinstance(info, dict):
                raise ProvisionError(
                    f"Shadeform instance {pod.pod_id} info: expected object"
                )
            status = str(info.get("status") or "").lower()
            last_status = status
            now = time.monotonic()
            if now - last_log >= 60.0:
                logger.info(
                    "Shadeform instance %s status=%s elapsed=%ds remaining=%ds",
                    pod.pod_id,
                    status or "?",
                    int(now - started),
                    max(0, int(deadline - now)),
                )
                last_log = now
            if status == "active":
                ip = str(info.get("ip") or "")
                if not ip:
                    self._sleep(10)
                    continue
                pod.ssh = SshTarget(
                    host=ip,
                    port=int(info.get("ssh_port") or 22),
                    user=str(info.get("ssh_user") or "shadeform"),
                )
                pod.raw = {**(pod.raw or {}), "instance_info": info}
                try:
                    if self._mount_workspace(pod):
                        return pod
                    mount_attempted = True
                except Exception as exc:  # noqa: BLE001
                    mount_attempted = True
                    logger.warning(
                        "Shadeform mount attempt failed for %s: %s", pod.pod_id, exc
                    )
                self._sleep(MOUNT_POLL_S)
                continue
            if status in ("error", "deleted"):
                detail = info.get("status_details") or ""
                raise ProvisionError(
                    f"Shadeform instance {pod.pod_id} entered {status}: {detail}"
                )
            self._sleep(15)
        reason = (
            "mount never succeeded"
            if mount_attempted
            else "instance never became active"
        )
        raise ProvisionError(
            f"Shadeform instance {pod.pod_id} not ready after {timeout_s}s "
            f"(last status={last_status or '?'}; {reason})"
        )

    def _mount_workspace(self, pod: Pod) -> bool:
        logger.info("Mounting /workspace on Shadeform instance %s", pod.pod_id)
        script = _mount_workspace_script(volume_gib=_volume_gib())
        result = ssh_exec(
            pod,
            f"bash -s <<'PARETON_MOUNT'\n{script}\nPARETON_MOUNT",
            timeout_s=600.0,
            check=False,
            state_dir=self._state_dir,
        )
        if result.exit_code == 0:
            return True
        logger.warning(
            "Mount /workspace failed on %s (exit=%s): stdout=%s stderr=%s",
            pod.pod_id,
            result.exit_code,
            (result.stdout or "")[-500:],
            (result.stderr or "")[-500:],
        )
        return False

    def destroy(self, pod: Pod) -> None:
        volume_uid = str((pod.raw or {}).get("volume_uid") or "")
        instance_err: Exception | None = None
        volume_err: Exception | None = None
        try:
            self._teardown_instance(pod.pod_id)
        except DestroyError as exc:
            instance_err = exc
        except Exception as exc:  # noqa: BLE001
            instance_err = DestroyError(str(exc))
        if volume_uid:
            try:
                self._teardown_volume(volume_uid, raise_on_fail=True)
            except DestroyError as exc:
                volume_err = exc
            except Exception as exc:  # noqa: BLE001
                volume_err = DestroyError(str(exc))
        if instance_err is not None and volume_err is not None:
            raise DestroyError(
                f"instance teardown failed: {instance_err}; "
                f"volume teardown also failed: {volume_err}"
            ) from volume_err
        if volume_err is not None:
            raise volume_err
        if instance_err is not None:
            raise instance_err

    def _teardown_instance(self, instance_id: str) -> None:
        try:
            self._post(f"/instances/{instance_id}/delete")
            logger.info("Shadeform instance %s deleted", instance_id)
        except ProvisionError as exc:
            if "404" in str(exc):
                logger.info("Shadeform instance %s already deleted", instance_id)
                return
            raise DestroyError(str(exc)) from exc

    def _teardown_volume(self, volume_id: str, *, raise_on_fail: bool = True) -> None:
        deadline = time.monotonic() + 300
        last_err = ""
        while time.monotonic() < deadline:
            try:
                self._post(f"/volumes/{volume_id}/delete")
                logger.info("Shadeform volume %s deleted", volume_id)
                return
            except ProvisionError as exc:
                msg = str(exc)
                last_err = msg
                if "404" in msg:
                    logger.info("Shadeform volume %s already deleted", volume_id)
                    return
                if any(code in msg for code in ("409", "423", "503")):
                    self._sleep(15)
                    continue
                break
        if raise_on_fail:
            raise DestroyError(
                f"Shadeform volume {volume_id} still not deleted: {last_err}"
            )
        logger.error("Timed out deleting Shadeform volume %s", volume_id)

    def list_pods(self) -> list[Pod]:
        data = self._get("/instances") or {}
        items = data.get("instances", data if isinstance(data, list) else [])
        out: list[Pod] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("id") or "")
            if not uid:
                continue
            name = str(item.get("name") or "")
            ssh_info = item.get("ssh") if isinstance(item.get("ssh"), dict) else {}
            out.append(
                Pod(
                    provider=self.name,
                    pod_id=uid,
                    name=name,
                    ssh=SshTarget(
                        host=str(ssh_info.get("host") or item.get("ip") or ""),
                        port=int(ssh_info.get("port") or item.get("ssh_port") or 22),
                        user=str(
                            ssh_info.get("user") or item.get("ssh_user") or "shadeform"
                        ),
                    ),
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
        items = data.get("volumes", data if isinstance(data, list) else [])
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
    "ShadeformProvider",
    "API_BASE",
    "SHADEFORM_DASHBOARD",
    "_format_instance_id",
    "_parse_instance_id",
    "_normalize_gpu_type",
    "_pick_os",
]
