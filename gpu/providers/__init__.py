"""Provider factory."""

from __future__ import annotations

import os

from gpu.errors import ProvisionError
from gpu.providers.base import Provider
from gpu.providers.static_ssh import StaticSshProvider
from gpu.providers.targon import TargonProvider


def get_provider(name: str, **kwargs) -> Provider:
    n = (name or "auto").strip().lower()
    if n in ("auto", "targon"):
        key = os.environ.get("PARETON_TARGON_API_KEY", "")
        if not key:
            try:
                import config as _cfg

                key = getattr(_cfg, "TARGON_API_KEY", "") or ""
            except Exception:  # noqa: BLE001
                key = ""
        if not key:
            raise ProvisionError(
                "PARETON_TARGON_API_KEY is not set (required for provider targon)"
            )
        return TargonProvider(key, **kwargs)
    if n == "static_ssh":
        # static_ssh ignores state_dir / transport kwargs from cloud callers
        target = kwargs.get("target")
        return StaticSshProvider(target=target) if target else StaticSshProvider()
    raise ProvisionError(
        f"unknown GPU provider {name!r}; supported: targon, static_ssh"
    )


__all__ = ["Provider", "get_provider", "TargonProvider", "StaticSshProvider"]
