"""Unmanaged static SSH target for local / lab GPU boxes."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu.errors import ProvisionError
from gpu.providers.lium import _gpu_type_matches
from gpu.ssh import exec as ssh_exec
from gpu.types import Offer, Pod, PodSpec, SshTarget

logger = logging.getLogger(__name__)

_TARGET_RE = re.compile(r"^(?P<user>[^@]+)@(?P<host>[^:]+)(?::(?P<port>\d+))?$")


def parse_static_ssh(target: str) -> SshTarget:
    m = _TARGET_RE.match(target.strip())
    if not m:
        raise ProvisionError(
            f"PARETON_GPU_STATIC_SSH must be user@host[:port], got {target!r}"
        )
    return SshTarget(
        host=m.group("host"),
        port=int(m.group("port") or 22),
        user=m.group("user"),
    )


def _env_or_config(env_name: str, config_attr: str) -> str:
    """Prefer process env (tests/CLI) over import-time config defaults."""
    val = os.environ.get(env_name, "")
    if val:
        return val
    try:
        import config as _cfg

        return str(getattr(_cfg, config_attr, "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _ssh_key_path_from_config() -> Path:
    """Require PARETON_GPU_SSH_KEY_PATH; never discover ~/.ssh."""
    raw = _env_or_config("PARETON_GPU_SSH_KEY_PATH", "GPU_SSH_KEY_PATH")
    if not raw:
        raise ProvisionError(
            "PARETON_GPU_SSH_KEY_PATH is required for static_ssh (no ~/.ssh discovery)"
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ProvisionError(f"PARETON_GPU_SSH_KEY_PATH not found: {path}")
    return path.resolve()


class StaticSshProvider:
    name = "static_ssh"

    def __init__(self, target: str | None = None) -> None:
        if target is None:
            target = _env_or_config("PARETON_GPU_STATIC_SSH", "GPU_STATIC_SSH")
        if not target:
            raise ProvisionError("PARETON_GPU_STATIC_SSH is not set")
        self._target = parse_static_ssh(target)
        self._key_path = _ssh_key_path_from_config()

    def search(self, spec: PodSpec) -> list[Offer]:
        return [
            Offer(
                provider=self.name,
                instance_id="static",
                description=f"static {self._target.user}@{self._target.host}",
                hourly_price_cents=0,
                gpu_count=spec.gpu_count or 1,
                gpu_type=spec.gpu_type or "static",
                raw={"gpu_type": spec.gpu_type, "gpu_count": spec.gpu_count},
            )
        ]

    def maintenance_pod(self, *, name: str = "static-host") -> Pod:
        """Describe the unmanaged target without provisioning or needing healthy GPUs."""
        return Pod(
            provider=self.name,
            pod_id=f"{self._target.user}@{self._target.host}:{self._target.port}",
            name=name,
            ssh=self._target,
            key_path=self._key_path,
            hourly_price_cents=0,
            created_utc=datetime.now(timezone.utc),
            ttl_hours=0.0,
            raw={"unmanaged": True},
        )

    def provision(self, offer: Offer, *, name: str, ssh_public_key: str) -> Pod:
        del ssh_public_key  # unused; caller provides durable key for cloud only
        pod = self.maintenance_pod(name=name)
        # Validate the actual host; a static target is global across campaigns.
        result = ssh_exec(
            pod, "nvidia-smi --query-gpu=name --format=csv,noheader", timeout_s=60.0
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        count = int(offer.raw.get("gpu_count", offer.gpu_count))
        expected = str(offer.raw.get("gpu_type", offer.gpu_type) or "")
        if len(names) < count or (
            expected
            and any(
                not _gpu_type_matches(_gpu_name(name), _gpu_name(expected))
                for name in names
            )
        ):
            raise ProvisionError(
                f"static_ssh hardware mismatch: requested {count}x {expected or 'GPU'}, "
                f"found {names}; use a matching host or drain incompatible campaigns"
            )
        if len(names) > count:
            logger.warning(
                "static_ssh host has %d GPUs; this campaign uses %d (%d idle)",
                len(names),
                count,
                len(names) - count,
            )
        return pod

    def destroy(self, pod: Pod) -> None:
        logger.info("static_ssh destroy is a no-op for unmanaged pod %s", pod.name)

    def list_pods(self) -> list[Pod]:
        return []

    def list_volumes(self) -> list[dict[str, Any]]:
        return []


def _gpu_name(value: str) -> str:
    value = re.sub(r"\b(nvidia|geforce)\b", "", value.lower())
    value = re.sub(r"\brtx[\s_-]*(?=\d)", "rtx", value)
    return " ".join(re.findall(r"[a-z0-9]+", value))
