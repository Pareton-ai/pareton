"""Lium GPU cloud provider (lium.io SDK).

Ports Cacheon's Lium control plane into Pareton's Provider protocol.
Uses ``gpu.ssh`` for data plane (no SDK exec/rsync). Per-pod TTL-named
volumes (not Cacheon's long-lived shared volume). Injectable ``client``
for offline tests.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from gpu.errors import DestroyError, ProvisionError
from gpu.keys import ensure_durable_keypair
from gpu.registry import _state_dir
from gpu.types import Offer, Pod, PodSpec, SshTarget

logger = logging.getLogger(__name__)

LIUM_DASHBOARD = "https://lium.io/your-pods"
SSH_KEY_NAME = "pareton-gpu"
READY_TIMEOUT_S = 600
READY_POLL_S = 10


class LiumClient(Protocol):
    def ls(
        self, *, gpu_type: str | None = None, gpu_count: int | None = None
    ) -> list[Any]: ...

    def up(self, **kwargs: Any) -> dict[str, Any]: ...

    def down(self, pod: Any) -> Any: ...

    def wait_ready(
        self, pod: Any, *, timeout: int = 300, poll_interval: int = 10
    ) -> Any: ...

    def volumes(self) -> list[Any]: ...

    def volume_create(self, name: str, *, description: str = "") -> Any: ...

    def volume_delete(self, volume_id: str) -> Any: ...

    def list_ssh_keys(self) -> list[Any]: ...

    def register_ssh_key(self, *, name: str, public_key: str) -> Any: ...

    def ps(self) -> list[Any]: ...


def _normalize_gpu_type(raw: str) -> str:
    t = (raw or "").strip()
    for prefix in ("NVIDIA-GeForce-", "NVIDIA-", "GeForce-"):
        if t.upper().startswith(prefix.upper()):
            return t[len(prefix) :]
    return t


def _key_body(pub_key: str) -> str:
    parts = pub_key.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else pub_key.strip()


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_ssh_target(ssh_cmd: str | None) -> SshTarget:
    """Parse ``ssh user@host -p N`` (Lium PodInfo.ssh_cmd shape)."""
    cmd = (ssh_cmd or "").strip()
    if not cmd:
        raise ProvisionError("Lium pod ready but ssh_cmd is empty")
    # Reuse PodInfo property logic without requiring a live SDK object.
    import re

    host = (re.findall(r"@(\S+)", cmd) or [None])[0]
    user = (re.findall(r"ssh (\S+)@", cmd) or [None])[0]
    port = 22
    if "-p " in cmd:
        try:
            port = int(cmd.split("-p ", 1)[1].split()[0])
        except (IndexError, ValueError) as exc:
            raise ProvisionError(f"cannot parse SSH port from {cmd!r}") from exc
    if not host or not user:
        raise ProvisionError(f"cannot parse SSH target from {cmd!r}")
    return SshTarget(host=host, port=port, user=user)


def _default_client(api_key: str, *, ssh_key_path: Path) -> LiumClient:
    from lium.sdk import Config, Lium

    return Lium(config=Config(api_key=api_key, ssh_key_path=ssh_key_path))


class LiumProvider:
    name = "lium"

    def __init__(
        self,
        api_key: str,
        *,
        state_dir: Path | None = None,
        client: LiumClient | None = None,
        ready_timeout_s: int = READY_TIMEOUT_S,
    ) -> None:
        if not api_key and client is None:
            raise ProvisionError("PARETON_LIUM_API_KEY is not set")
        self._state_dir = _state_dir(state_dir)
        self._key_path = ensure_durable_keypair(self._state_dir)[0]
        self._ready_timeout_s = ready_timeout_s
        self._client: LiumClient = client or _default_client(
            api_key, ssh_key_path=self._key_path
        )

    def _ensure_ssh_key(self, ssh_public_key: str) -> str:
        local_body = _key_body(ssh_public_key)
        for key in self._client.list_ssh_keys():
            remote = _key_body(str(_attr(key, "public_key", "") or ""))
            if remote == local_body:
                logger.info(
                    "SSH key already registered with Lium (name=%s id=%s)",
                    _attr(key, "name", "?"),
                    _attr(key, "id", "?"),
                )
                return ssh_public_key
        self._client.register_ssh_key(name=SSH_KEY_NAME, public_key=ssh_public_key)
        logger.info("Registered SSH key %s with Lium", SSH_KEY_NAME)
        return ssh_public_key

    def search(self, spec: PodSpec) -> list[Offer]:
        want = max(1, int(spec.gpu_count or 1))
        executors = self._client.ls(gpu_count=want)
        out: list[Offer] = []
        for ex in executors:
            if not bool(_attr(ex, "docker_in_docker", False)):
                continue
            gpu_count = int(_attr(ex, "gpu_count", 0) or 0)
            # Discrete SKUs: require exact count (same as Shadeform).
            if gpu_count != want:
                continue
            gpu_type = _normalize_gpu_type(str(_attr(ex, "gpu_type", "") or ""))
            if spec.gpu_type and gpu_type.upper() != spec.gpu_type.upper():
                # Also allow substring match (e.g. "H200 NVL" vs "H200").
                if spec.gpu_type.upper() not in gpu_type.upper():
                    continue
            price_cents = int(round(float(_attr(ex, "price_per_hour", 0) or 0) * 100))
            if price_cents > spec.max_hourly_cents:
                continue
            executor_id = str(_attr(ex, "id", "") or "")
            if not executor_id:
                continue
            desc = str(_attr(ex, "machine_name", "") or executor_id)
            out.append(
                Offer(
                    provider=self.name,
                    instance_id=executor_id,
                    description=desc,
                    hourly_price_cents=price_cents,
                    gpu_count=gpu_count,
                    gpu_type=gpu_type,
                    raw={"executor_id": executor_id},
                )
            )
        out.sort(key=lambda o: (o.hourly_price_cents, o.gpu_count))
        return out

    def _create_volume(self, name: str) -> str:
        vol = self._client.volume_create(
            name,
            description="Pareton bench workspace (TTL name, delete with pod)",
        )
        volume_id = str(_attr(vol, "id", "") or "")
        if not volume_id:
            raise ProvisionError(f"Lium volume_create returned no id for name={name!r}")
        logger.info("Created Lium volume %s (name=%s)", volume_id, name)
        return volume_id

    def _abort(self, pod_id: str | None, volume_id: str | None) -> None:
        if pod_id:
            try:
                self._client.down(SimpleNamespace(id=pod_id))
            except Exception:  # noqa: BLE001
                logger.exception("abort pod cleanup failed for %s", pod_id)
        if volume_id:
            try:
                self._client.volume_delete(volume_id)
            except Exception:  # noqa: BLE001
                logger.exception("abort volume cleanup failed for %s", volume_id)

    def provision(self, offer: Offer, *, name: str, ssh_public_key: str) -> Pod:
        executor_id = str(
            (offer.raw or {}).get("executor_id") or offer.instance_id or ""
        )
        if not executor_id:
            raise ProvisionError("Lium offer missing executor_id")

        ssh_pub = self._ensure_ssh_key(ssh_public_key)
        volume_id = self._create_volume(name)
        pod_id: str | None = None
        try:
            created = self._client.up(
                executor_id=executor_id,
                name=name,
                volume_id=volume_id,
                ssh_keys=[ssh_pub],
                ssh_name=SSH_KEY_NAME,
            )
            if isinstance(created, dict):
                pod_id = str(created.get("id") or created.get("pod_id") or "")
            else:
                pod_id = str(_attr(created, "id", "") or "")
            if not pod_id:
                raise ProvisionError("Lium up() returned no pod id")
            logger.info(
                "Lium pod %s created (executor=%s volume=%s)",
                pod_id,
                executor_id,
                volume_id,
            )
            logger.info("Dashboard: %s/%s", LIUM_DASHBOARD, pod_id)

            ready = self._client.wait_ready(
                pod_id,
                timeout=self._ready_timeout_s,
                poll_interval=READY_POLL_S,
            )
            if ready is None:
                raise ProvisionError(
                    f"Lium pod {pod_id} not ready after {self._ready_timeout_s}s"
                )
            ssh_cmd = str(_attr(ready, "ssh_cmd", "") or "")
            ssh = _parse_ssh_target(ssh_cmd)
            return Pod(
                provider=self.name,
                pod_id=pod_id,
                name=name,
                ssh=ssh,
                key_path=self._key_path,
                hourly_price_cents=offer.hourly_price_cents,
                created_utc=datetime.now(timezone.utc),
                ttl_hours=0.0,
                raw={
                    "volume_uid": volume_id,
                    "volume_name": name,
                    "executor_id": executor_id,
                    "ssh_cmd": ssh_cmd,
                    "dashboard": f"{LIUM_DASHBOARD}/{pod_id}",
                    "status": _attr(ready, "status", ""),
                },
            )
        except Exception:
            self._abort(pod_id, volume_id)
            raise

    def destroy(self, pod: Pod) -> None:
        volume_uid = str((pod.raw or {}).get("volume_uid") or "")
        instance_err: Exception | None = None
        volume_err: Exception | None = None
        try:
            self._client.down(SimpleNamespace(id=pod.pod_id))
            logger.info("Lium pod %s terminated", pod.pod_id)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "404" in msg or "not found" in msg.lower():
                logger.info("Lium pod %s already gone", pod.pod_id)
            else:
                instance_err = DestroyError(msg)
        if volume_uid:
            try:
                self._client.volume_delete(volume_uid)
                logger.info("Lium volume %s deleted", volume_uid)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "404" in msg or "not found" in msg.lower():
                    logger.info("Lium volume %s already gone", volume_uid)
                else:
                    volume_err = DestroyError(msg)
        if instance_err is not None and volume_err is not None:
            raise DestroyError(
                f"instance teardown failed: {instance_err}; "
                f"volume teardown also failed: {volume_err}"
            ) from volume_err
        if volume_err is not None:
            raise volume_err
        if instance_err is not None:
            raise instance_err

    def list_pods(self) -> list[Pod]:
        out: list[Pod] = []
        for item in self._client.ps():
            uid = str(_attr(item, "id", "") or "")
            if not uid:
                continue
            name = str(_attr(item, "name", "") or "")
            ssh_cmd = str(_attr(item, "ssh_cmd", "") or "")
            try:
                ssh = _parse_ssh_target(ssh_cmd) if ssh_cmd else SshTarget("", 22, "")
            except ProvisionError:
                ssh = SshTarget("", 22, "")
            out.append(
                Pod(
                    provider=self.name,
                    pod_id=uid,
                    name=name,
                    ssh=ssh,
                    key_path=self._key_path,
                    hourly_price_cents=0,
                    created_utc=datetime.now(timezone.utc),
                    ttl_hours=0.0,
                    raw={
                        "volume_uid": "",
                        "status": _attr(item, "status", ""),
                        "ssh_cmd": ssh_cmd,
                    },
                )
            )
        return out

    def list_volumes(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._client.volumes():
            out.append(
                {
                    "id": str(_attr(item, "id", "") or ""),
                    "name": str(_attr(item, "name", "") or ""),
                    "raw": item
                    if isinstance(item, dict)
                    else {"id": _attr(item, "id")},
                }
            )
        return out


__all__ = [
    "LiumProvider",
    "LIUM_DASHBOARD",
    "SSH_KEY_NAME",
    "_normalize_gpu_type",
    "_parse_ssh_target",
]
