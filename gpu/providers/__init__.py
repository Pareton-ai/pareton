"""Provider factory."""

from __future__ import annotations

import os

from gpu.errors import ProvisionError
from gpu.providers.base import Provider
from gpu.providers.shadeform import ShadeformProvider
from gpu.providers.static_ssh import StaticSshProvider
from gpu.providers.targon import TargonProvider


def _env_or_config(env_name: str, config_attr: str) -> str:
    val = os.environ.get(env_name, "")
    if val:
        return val
    try:
        import config as _cfg

        return str(getattr(_cfg, config_attr, "") or "")
    except Exception:  # noqa: BLE001
        return ""


def get_provider(name: str, **kwargs) -> Provider:
    n = (name or "auto").strip().lower()
    if n in ("auto", "targon"):
        key = _env_or_config("PARETON_TARGON_API_KEY", "TARGON_API_KEY")
        if not key:
            raise ProvisionError(
                "PARETON_TARGON_API_KEY is not set (required for provider targon)"
            )
        return TargonProvider(key, **kwargs)
    if n == "shadeform":
        key = _env_or_config("PARETON_SHADEFORM_API_KEY", "SHADEFORM_API_KEY")
        if not key:
            raise ProvisionError(
                "PARETON_SHADEFORM_API_KEY is not set (required for provider shadeform)"
            )
        return ShadeformProvider(key, **kwargs)
    if n == "static_ssh":
        # static_ssh ignores state_dir / transport kwargs from cloud callers
        target = kwargs.get("target")
        return StaticSshProvider(target=target) if target else StaticSshProvider()
    raise ProvisionError(
        f"unknown GPU provider {name!r}; supported: targon, shadeform, static_ssh"
    )


__all__ = [
    "Provider",
    "get_provider",
    "TargonProvider",
    "ShadeformProvider",
    "StaticSshProvider",
]
