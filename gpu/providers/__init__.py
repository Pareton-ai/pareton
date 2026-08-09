"""Provider factory."""

from __future__ import annotations

import os

from gpu.errors import ProvisionError
from gpu.providers.base import Provider
from gpu.providers.lium import LiumProvider
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


def resolve_provider_name(name: str | None = None) -> str:
    """Map CLI/worker provider names; ``auto`` → ``PARETON_GPU_PROVIDER`` (default lium)."""
    n = (name or "auto").strip().lower() or "auto"
    if n != "auto":
        return n
    configured = (
        _env_or_config("PARETON_GPU_PROVIDER", "GPU_PROVIDER") or "lium"
    ).strip()
    configured = configured.lower() or "lium"
    if configured == "auto":
        # Avoid recursion / footgun if someone sets PARETON_GPU_PROVIDER=auto.
        return "lium"
    return configured


def _fallback_names() -> list[str]:
    raw = os.environ.get("PARETON_GPU_PROVIDER_FALLBACKS")
    if raw is not None:
        return [p.strip().lower() for p in raw.split(",") if p.strip()]
    try:
        import config as _cfg

        val = getattr(_cfg, "GPU_PROVIDER_FALLBACKS", None)
        if val is None:
            return ["shadeform"]
        return [str(p).strip().lower() for p in val if str(p).strip()]
    except Exception:  # noqa: BLE001
        return ["shadeform"]


def provider_order(name: str | None = None) -> list[str]:
    """Primary provider plus configured fallbacks, deduped.

    static_ssh never falls back: a local SSH target failure must not rent a
    cloud pod as a surprise side effect.
    """
    primary = resolve_provider_name(name)
    if primary == "static_ssh":
        return [primary]
    return [primary] + [f for f in _fallback_names() if f != primary]


def get_provider(name: str, **kwargs) -> Provider:
    n = resolve_provider_name(name)
    if n == "targon":
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
    if n == "lium":
        # Drop Shadeform/Targon-only kwargs (transport) if a caller spreads them.
        lium_kwargs = {k: v for k, v in kwargs.items() if k != "transport"}
        key = _env_or_config("PARETON_LIUM_API_KEY", "LIUM_API_KEY")
        if not key and lium_kwargs.get("client") is None:
            raise ProvisionError(
                "PARETON_LIUM_API_KEY is not set (required for provider lium)"
            )
        return LiumProvider(key, **lium_kwargs)
    if n == "static_ssh":
        # static_ssh ignores state_dir / transport kwargs from cloud callers
        target = kwargs.get("target")
        return StaticSshProvider(target=target) if target else StaticSshProvider()
    raise ProvisionError(
        f"unknown GPU provider {name!r} (resolved {n!r}); "
        "supported: auto, targon, shadeform, lium, static_ssh"
    )


__all__ = [
    "Provider",
    "get_provider",
    "provider_order",
    "resolve_provider_name",
    "TargonProvider",
    "ShadeformProvider",
    "LiumProvider",
    "StaticSshProvider",
]
