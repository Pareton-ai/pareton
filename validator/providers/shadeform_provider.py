"""Shadeform GPU cloud provider using the v1 REST API.

Registers the host's SSH key, creates a fresh block volume per rental,
attaches it to an 8-GPU VM, mounts it at ``/workspace``, and tears down
instance + volume when done. Requires ``SHADEFORM_API_KEY`` only.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Generator

import paramiko
import requests

from . import GpuInstance, GpuProvider, PodHandle, lookup_vram
from .ssh_keys import discover_ssh_private_key, discover_ssh_public_key

logger = logging.getLogger(__name__)

API_BASE = "https://api.shadeform.ai/v1"
SHADEFORM_DASHBOARD = "https://platform.shadeform.ai/instances"
SSH_KEY_NAME = "cacheon-cpu-validator"
VOLUME_SIZE_GB = 500
_INSTANCE_ID_SEP = ":"


def _parse_instance_id(instance_id: str) -> tuple[str, str, str]:
    parts = instance_id.split(_INSTANCE_ID_SEP, 2)
    if len(parts) != 3:
        raise ValueError(f"invalid Shadeform instance_id: {instance_id!r}")
    return parts[0], parts[1], parts[2]


def _format_instance_id(cloud: str, region: str, shade_instance_type: str) -> str:
    return f"{cloud}{_INSTANCE_ID_SEP}{region}{_INSTANCE_ID_SEP}{shade_instance_type}"


def _pick_os(os_options: list[str]) -> str | None:
    for pref in (
        "ubuntu22.04",
        "ubuntu24.04",
        "ubuntu20.04",
    ):
        for opt in os_options:
            if pref in opt.lower():
                return opt
    return os_options[0] if os_options else None


class ShadeformProvider:
    """GpuProvider backed by the Shadeform REST API."""

    name = "shadeform"
    READY_TIMEOUT_S = 720

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        self._ssh_key_id: str | None = None
        self._ssh_key_path = None
        self._ssh_pkey: paramiko.PKey | None = None
        try:
            self._ssh_key_path, self._ssh_pkey = discover_ssh_private_key()
        except RuntimeError:
            pass

    # -- HTTP helpers ------------------------------------------------------

    def _get(self, path: str, **kwargs: Any) -> Any:
        resp = requests.get(
            f"{API_BASE}{path}", headers=self._headers, timeout=60, **kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict | None = None) -> Any:
        resp = requests.post(
            f"{API_BASE}{path}", headers=self._headers, json=json, timeout=60
        )
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def _delete(self, path: str) -> None:
        resp = requests.post(f"{API_BASE}{path}", headers=self._headers, timeout=60)
        resp.raise_for_status()

    # -- SSH key management ------------------------------------------------

    @staticmethod
    def _key_body(pub_key: str) -> str:
        parts = pub_key.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else pub_key.strip()

    def _ensure_ssh_key(self) -> str:
        if self._ssh_key_id:
            return self._ssh_key_id

        _pub_path, pub_key = discover_ssh_public_key()
        local_body = self._key_body(pub_key)

        resp = self._get("/sshkeys")
        for key in resp.get("ssh_keys", []):
            if self._key_body(key.get("public_key", "")) == local_body:
                self._ssh_key_id = key["id"]
                logger.info(
                    "SSH key already registered with Shadeform (id=%s)",
                    self._ssh_key_id,
                )
                return self._ssh_key_id

        created = self._post(
            "/sshkeys/add",
            json={"name": SSH_KEY_NAME, "public_key": pub_key},
        )
        self._ssh_key_id = created["id"]
        logger.info("Registered SSH key with Shadeform: %s", self._ssh_key_id)
        return self._ssh_key_id

    def _create_volume(self, cloud: str, region: str) -> str:
        name = f"cacheon-eval-{uuid.uuid4().hex[:8]}"
        resp = self._post(
            "/volumes/create",
            json={
                "cloud": cloud,
                "region": region,
                "size_in_gb": VOLUME_SIZE_GB,
                "name": name,
            },
        )
        volume_id = resp["id"]
        logger.info(
            "Created Shadeform volume %s (%d GB, cloud=%s region=%s)",
            volume_id,
            VOLUME_SIZE_GB,
            cloud,
            region,
        )
        return volume_id

    # -- GpuProvider interface ---------------------------------------------

    def search(self) -> list[GpuInstance]:
        params = {
            "num_gpus": "8",
            "available": "true",
            "sort": "price",
        }
        resp = self._get("/instances/types", params=params)

        out: list[GpuInstance] = []
        for item in resp.get("instance_types", []):
            cfg = item.get("configuration", {})
            gpu_type_raw = cfg.get("gpu_type", "")
            gpu_count = cfg.get("num_gpus", 0)
            vram_per_gpu = cfg.get("vram_per_gpu_in_gb", 0)

            canon, table_vram = lookup_vram(gpu_type_raw)
            if not table_vram:
                continue
            if gpu_count != 8:
                continue

            deployment = (item.get("deployment_type") or "").lower()
            if deployment not in ("vm", "baremetal"):
                continue

            hourly_price = int(item.get("hourly_price", 0))
            storage_gb = int(cfg.get("storage_in_gb", 0) or 0)
            memory_gb = int(cfg.get("memory_in_gb", 0) or 0)
            vcpus = int(cfg.get("vcpus", 0) or 0)
            effective_vram = vram_per_gpu or table_vram

            for avail in item.get("availability", []):
                if not avail.get("available"):
                    continue
                region = avail.get("region", "")
                cloud = item.get("cloud", "")
                shade_type = item.get("shade_instance_type", "")
                if not cloud or not region or not shade_type:
                    continue

                instance_id = _format_instance_id(cloud, region, shade_type)
                display = avail.get("display_name") or shade_type
                out.append(
                    GpuInstance(
                        provider="shadeform",
                        instance_id=instance_id,
                        description=f"{display} ({shade_type})",
                        hourly_price_cents=hourly_price,
                        num_gpus=gpu_count,
                        gpu_type=canon,
                        vram_per_gpu_gb=effective_vram,
                        total_vram_gb=gpu_count * effective_vram,
                        storage_gb=storage_gb,
                        memory_gb=memory_gb,
                        vcpus=vcpus,
                        docker_in_docker=True,
                        raw={
                            "cloud": cloud,
                            "region": region,
                            "shade_instance_type": shade_type,
                            "os_options": cfg.get("os_options", []),
                            "instance_type": item,
                            "availability": avail,
                        },
                    )
                )
        return out

    def rent(self, instance: GpuInstance) -> PodHandle:
        raw = instance.raw or {}
        cloud = raw.get("cloud", "")
        region = raw.get("region", "")
        shade_type = raw.get("shade_instance_type", "")
        if not cloud or not region or not shade_type:
            cloud, region, shade_type = _parse_instance_id(instance.instance_id)

        ssh_key_id = self._ensure_ssh_key()
        volume_id = self._create_volume(cloud, region)

        os_options = raw.get("os_options") or []
        os_name = _pick_os(os_options)

        body: dict[str, Any] = {
            "cloud": cloud,
            "region": region,
            "shade_instance_type": shade_type,
            "shade_cloud": True,
            "name": "cacheon-eval",
            "volume_ids": [volume_id],
            "ssh_key_id": ssh_key_id,
            "volume_mount": {"auto": True},
        }
        if os_name:
            body["os"] = os_name

        try:
            created = self._post("/instances/create", json=body)
            instance_id = created["id"]
        except Exception:
            self._abort_rent(None, volume_id)
            raise

        try:
            logger.info(
                "Shadeform instance %s created (cloud=%s region=%s type=%s volume=%s)",
                instance_id,
                cloud,
                region,
                shade_type,
                volume_id,
            )
            logger.info("Dashboard: %s/%s", SHADEFORM_DASHBOARD, instance_id)

            return PodHandle(
                provider="shadeform",
                pod_id=instance_id,
                gpu_count=instance.num_gpus,
                hourly_price_cents=instance.hourly_price_cents,
                raw={
                    "cloud": cloud,
                    "region": region,
                    "shade_instance_type": shade_type,
                    "volume_id": volume_id,
                },
            )
        except Exception:
            self._abort_rent(instance_id, volume_id)
            raise

    def wait_ready(
        self, handle: PodHandle, timeout_s: int = READY_TIMEOUT_S
    ) -> PodHandle:
        deadline = time.monotonic() + timeout_s
        poll_count = 0
        info: dict[str, Any] = {}
        mount_attempted = False

        while time.monotonic() < deadline:
            info = self._get(f"/instances/{handle.pod_id}/info")
            status = (info.get("status") or "").lower()
            elapsed = int(time.monotonic() - (deadline - timeout_s))
            poll_count += 1
            if poll_count % 3 == 1:
                logger.info(
                    "Waiting for Shadeform instance %s: status=%s (%ds elapsed)",
                    handle.pod_id,
                    status,
                    elapsed,
                )

            if status == "active":
                ip = info.get("ip") or ""
                if not ip:
                    time.sleep(10)
                    continue
                handle.raw["ssh"] = {
                    "host": ip,
                    "port": info.get("ssh_port") or 22,
                    "user": info.get("ssh_user") or "shadeform",
                }
                handle.raw["instance_info"] = info
                try:
                    if self._mount_workspace(handle):
                        return handle
                    mount_attempted = True
                except Exception as exc:
                    mount_attempted = True
                    logger.warning(
                        "Shadeform mount attempt failed for %s: %s",
                        handle.pod_id,
                        exc,
                    )
                logger.info(
                    "Workspace not ready on %s yet, retrying (%ds elapsed)",
                    handle.pod_id,
                    elapsed,
                )
                time.sleep(15)
                continue

            if status in ("error", "deleted"):
                detail = info.get("status_details") or ""
                raise RuntimeError(
                    f"Shadeform instance {handle.pod_id} entered {status}: {detail}"
                )
            time.sleep(15)

        detail = info.get("status_details") or ""
        reason = (
            "mount never succeeded"
            if mount_attempted
            else "instance never became active"
        )
        raise TimeoutError(
            f"Shadeform instance {handle.pod_id} not ready after {timeout_s}s "
            f"(last status={info.get('status')!r}, details={detail!r}; {reason})"
        )

    def _mount_workspace_script(self) -> str:
        min_kb = int(
            VOLUME_SIZE_GB * 1024 * 1024 * 0.9
        )  # 90% of nominal (450 GiB floor for 500)
        return f"""set -euo pipefail
MIN_KB={min_kb}

if mountpoint -q /workspace 2>/dev/null; then
  ws_kb=$(df -Pk /workspace | awk 'NR==2 {{print $2}}')
  if [ "$ws_kb" -ge "$MIN_KB" ]; then
    echo "OK: /workspace already mounted (${{ws_kb}}KB)"
    df -h /workspace
    exit 0
  fi
fi
sudo mkdir -p /workspace

is_system_mount() {{
  case "$1" in
    /|/boot|/boot/efi|/var/lib/docker/*|/snap/*) return 0 ;;
  esac
  return 1
}}

uses_system_storage() {{
  lsblk -rpno NAME,MOUNTPOINT "$1" 2>/dev/null | awk '
    $2=="/" || $2=="/boot" || $2=="/boot/efi" {{ exit 0 }}
    END {{ exit 1 }}
  '
}}

mount_blockdev() {{
  local dev="$1"
  if ! sudo blkid "$dev" >/dev/null 2>&1; then
    sudo mkfs.ext4 -F "$dev"
  fi
  if ! sudo mount "$dev" /workspace; then
    mountpoint -q /workspace 2>/dev/null || return 1
  fi
  sudo chmod 1777 /workspace
  echo "OK: mounted $dev on /workspace"
  df -h /workspace
}}

# Prefer the largest unmounted disk/partition at or above MIN_KB (Shadeform volume attach).
best_dev=""
best_dev_kb=0
while read -r name type size mount; do
  if [ "$type" != "disk" ] && [ "$type" != "part" ]; then
    continue
  fi
  if [ -n "${{mount:-}}" ]; then
    continue
  fi
  if uses_system_storage "$name"; then
    continue
  fi
  size_kb=$((size / 1024))
  if [ "$size_kb" -lt "$MIN_KB" ]; then
    continue
  fi
  if [ "$size_kb" -gt "$best_dev_kb" ]; then
    best_dev_kb=$size_kb
    best_dev="$name"
  fi
done < <(lsblk -b -rpno NAME,TYPE,SIZE,MOUNTPOINT)

if [ -n "$best_dev" ]; then
  mount_blockdev "$best_dev"
  exit 0
fi

# Else bind-mount the largest eligible filesystem already mounted elsewhere.
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
  sudo mount --bind "$best_mp" /workspace
  sudo chmod 1777 /workspace
  echo "OK: bind-mounted $best_mp to /workspace"
  df -h /workspace
  exit 0
fi

echo "ERROR: no mountable storage >= ${{MIN_KB}}KB for /workspace"
lsblk
df -h
exit 1
"""

    def _mount_workspace(self, handle: PodHandle) -> bool:
        logger.info("Mounting workspace storage at /workspace on %s", handle.pod_id)
        result = self.exec(handle, self._mount_workspace_script())
        if result.get("success"):
            return True
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        logger.warning(
            "Mount /workspace failed on %s (exit=%s): stdout=%s stderr=%s",
            handle.pod_id,
            result.get("exit_code"),
            stdout[-500:],
            stderr[-500:],
        )
        return False

    def _abort_rent(self, instance_id: str | None, volume_id: str) -> None:
        """Best-effort cleanup when rent fails after volume and/or instance creation."""
        if instance_id:
            self._teardown_instance(instance_id)
        self._teardown_volume(volume_id)

    def _load_private_key(self) -> paramiko.PKey:
        if self._ssh_pkey is not None:
            return self._ssh_pkey
        self._ssh_key_path, self._ssh_pkey = discover_ssh_private_key()
        return self._ssh_pkey

    def _ssh_connect(self, handle: PodHandle) -> paramiko.SSHClient:
        ssh_info = handle.raw.get("ssh", {})
        host = ssh_info.get("host", "")
        port = int(ssh_info.get("port") or 22)
        user = ssh_info.get("user") or "shadeform"

        if not host:
            raise RuntimeError(f"No SSH host for Shadeform instance {handle.pod_id}")

        pkey = self._load_private_key()
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=host, port=port, username=user, pkey=pkey, timeout=30
                )
                if attempt > 1:
                    logger.info("SSH connected on attempt %d", attempt)
                return client
            except paramiko.AuthenticationException:
                client.close()
                if attempt == max_retries:
                    raise
                logger.info(
                    "SSH auth failed (attempt %d/%d), retrying in 15s...",
                    attempt,
                    max_retries,
                )
                time.sleep(15)
            except Exception:
                client.close()
                raise
        raise RuntimeError("unreachable")

    def exec(self, handle: PodHandle, command: str) -> dict[str, Any]:
        client = self._ssh_connect(handle)
        try:
            _stdin, stdout, stderr = client.exec_command(command, timeout=10800)
            exit_code = stdout.channel.recv_exit_status()
            return {
                "stdout": stdout.read().decode("utf-8", errors="replace"),
                "stderr": stderr.read().decode("utf-8", errors="replace"),
                "exit_code": exit_code,
                "success": exit_code == 0,
            }
        finally:
            client.close()

    def stream_exec(
        self, handle: PodHandle, command: str
    ) -> Generator[dict[str, str], None, None]:
        client = self._ssh_connect(handle)
        try:
            _stdin, stdout, _stderr = client.exec_command(command, timeout=10800)
            for line in iter(stdout.readline, ""):
                yield {"type": "stdout", "data": line}
        finally:
            client.close()

    def teardown(self, handle: PodHandle) -> None:
        raw = handle.raw or {}
        volume_id = raw.get("volume_id")

        self._teardown_instance(handle.pod_id)
        if volume_id:
            self._teardown_volume(volume_id)

    def _teardown_instance(self, instance_id: str) -> None:
        try:
            self._delete(f"/instances/{instance_id}/delete")
            logger.info("Shadeform instance %s deleted", instance_id)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 404:
                logger.info("Shadeform instance %s already deleted", instance_id)
                return
            logger.error(
                "Shadeform instance teardown failed for %s: %s", instance_id, exc
            )
        except Exception as exc:
            logger.error(
                "Shadeform instance teardown failed for %s: %s", instance_id, exc
            )

    def _teardown_volume(self, volume_id: str) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                self._delete(f"/volumes/{volume_id}/delete")
                logger.info("Shadeform volume %s deleted", volume_id)
                return
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 404:
                    logger.info("Shadeform volume %s already deleted", volume_id)
                    return
                if status in (409, 423, 503):
                    logger.info(
                        "Volume %s not ready to delete (HTTP %s), retrying...",
                        volume_id,
                        status,
                    )
                    time.sleep(15)
                    continue
                logger.error(
                    "Shadeform volume teardown failed for %s: %s", volume_id, exc
                )
                return
            except Exception as exc:
                logger.warning(
                    "Shadeform volume delete retry for %s: %s", volume_id, exc
                )
                time.sleep(15)

        logger.error(
            "Timed out deleting Shadeform volume %s after instance delete", volume_id
        )
