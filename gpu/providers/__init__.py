"""Provider factory."""

from __future__ import annotations

import os

from gpu.errors import ProvisionError
from gpu.providers.base import Provider
from gpu.providers.lium import LiumProvider
from gpu.providers.runpod import RunpodProvider
from gpu.providers.shadeform import ShadeformProvider
from gpu.providers.static_ssh import StaticSshProvider
from gpu.providers.targon import TargonProvider

_DEFAULT_GPU_PROVIDERS = ("lium", "shadeform", "runpod", "targon")


def _env_or_config(env_name: str, config_attr: str) -> str:
    val = os.environ.get(env_name, "")
    if val:
        return val
    try:
        import config as _cfg

        return str(getattr(_cfg, config_attr, "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _parse_provider_list(raw: str) -> list[str]:
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def configured_providers() -> list[str]:
    """Ordered provider try-list from env/config (first → last)."""
    raw = os.environ.get("PARETON_GPU_PROVIDERS")
    if raw is not None:
        parsed = _parse_provider_list(raw)
        return parsed or list(_DEFAULT_GPU_PROVIDERS)

    # Legacy primary + fallbacks (tests and older .env files).
    if (
        "PARETON_GPU_PROVIDER" in os.environ
        or "PARETON_GPU_PROVIDER_FALLBACKS" in os.environ
    ):
        primary = (os.environ.get("PARETON_GPU_PROVIDER") or "lium").strip().lower()
        if not primary or primary == "auto":
            primary = "lium"
        if "PARETON_GPU_PROVIDER_FALLBACKS" in os.environ:
            fallbacks = _parse_provider_list(
                os.environ.get("PARETON_GPU_PROVIDER_FALLBACKS", "")
            )
        else:
            try:
                import config as _cfg

                fallbacks = [
                    str(p).strip().lower()
                    for p in getattr(_cfg, "GPU_PROVIDER_FALLBACKS", ["shadeform"])
                    if str(p).strip()
                ]
            except Exception:  # noqa: BLE001
                fallbacks = ["shadeform"]
        out: list[str] = []
        for name in [primary, *fallbacks]:
            if name and name != "auto" and name not in out:
                out.append(name)
        return out or list(_DEFAULT_GPU_PROVIDERS)

    try:
        import config as _cfg

        vals = getattr(_cfg, "GPU_PROVIDERS", None)
        if vals:
            parsed = [str(p).strip().lower() for p in vals if str(p).strip()]
            if parsed:
                return parsed
    except Exception:  # noqa: BLE001, S110
        pass
    return list(_DEFAULT_GPU_PROVIDERS)


def resolve_provider_name(name: str | None = None) -> str:
    """Map CLI/worker provider names; ``auto`` → first of ``PARETON_GPU_PROVIDERS``."""
    n = (name or "auto").strip().lower() or "auto"
    if n != "auto":
        return n
    providers = configured_providers()
    return providers[0] if providers else "lium"


def provider_order(name: str | None = None) -> list[str]:
    """Ordered providers to try for this request.

    ``auto`` / unset walks ``PARETON_GPU_PROVIDERS`` as-is. An explicit name is
    tried first, then the rest of the configured list (deduped). ``static_ssh``
    never falls back to cloud.
    """
    n = (name or "auto").strip().lower() or "auto"
    if n == "static_ssh":
        return ["static_ssh"]
    configured = configured_providers()
    if n == "auto":
        return list(configured)
    return [n] + [p for p in configured if p != n]


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
    if n == "runpod":
        key = _env_or_config("PARETON_RUNPOD_API_KEY", "RUNPOD_API_KEY")
        if not key:
            raise ProvisionError(
                "PARETON_RUNPOD_API_KEY is not set (required for provider runpod)"
            )
        return RunpodProvider(key, **kwargs)
    if n == "static_ssh":
        # static_ssh ignores state_dir / transport kwargs from cloud callers
        target = kwargs.get("target")
        return StaticSshProvider(target=target) if target else StaticSshProvider()
    raise ProvisionError(
        f"unknown GPU provider {name!r} (resolved {n!r}); "
        "supported: auto, targon, shadeform, lium, runpod, static_ssh"
    )


__all__ = [
    "LiumProvider",
    "Provider",
    "RunpodProvider",
    "ShadeformProvider",
    "StaticSshProvider",
    "TargonProvider",
    "configured_providers",
    "get_provider",
    "provider_order",
    "resolve_provider_name",
]
