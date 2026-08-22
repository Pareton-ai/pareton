"""Campaign ``engine`` profile: how to install and launch the inference engine.

Pareton is engine-agnostic by design — ``bench/lifecycle.py`` already treats the
engine as a black box behind an OpenAI-compatible base URL — but the *build*
recipe was a module constant in ``builder/hermetic.py``. This module turns that
recipe into a campaign field so one manifest can pin vLLM and another SGLang.

Only three things actually differ between the two engines (PAR-53 / PAR-54,
measured on live engines):

* ``install_cmd`` — vLLM's package root is the repo root; SGLang's is ``python/``
* ``entrypoint``  — different server module
* ``cache_dir``   — where the engine writes its compile cache in the container

Deliberately **not** fields:

* ``health_path``  — both engines are ready on ``/v1/models``, which is already
  the default in ``bench.lifecycle.wait_until_healthy``. SGLang's ``/health``
  returns 503 on a fully ready server, so nothing should point at it.
* ``TORCH_CUDA_ARCH_LIST`` — SGLang does no CUDA compilation at all; its kernels
  ship as the prebuilt ``sglang-kernel`` wheel.

Consuming this in the builder is PAR-57; this module only defines and validates.
"""

from __future__ import annotations

from typing import Any

# Engines with a verified recipe. Kept strict on purpose: PAR-58 selects
# response-shape fixtures and logprob alignment behaviour by name, so a typo
# must fail loudly rather than silently fall back to vLLM handling.
KNOWN_ENGINE_NAMES = frozenset({"vllm", "sglang"})

# vLLM v0.24.0 pin — matches builder.hermetic today. This is the implied
# profile for every manifest that carries no ``engine`` block.
VLLM_ENGINE: dict[str, Any] = {
    "name": "vllm",
    "install_cmd": "pip install --no-deps --no-build-isolation -e .",
    "entrypoint": ["python", "-m", "vllm.entrypoints.openai.api_server"],
    "cache_dir": "/root/.cache/vllm",
}

# SGLang v0.5.17. Verified on a live B300 (sm_103) in PAR-54: patched build
# completes in ~5s under --network=none, and the editable install correctly
# shadows the base image's preinstalled sglang.
SGLANG_ENGINE: dict[str, Any] = {
    "name": "sglang",
    "install_cmd": "pip install --no-deps --no-build-isolation -e python/",
    "entrypoint": ["python3", "-m", "sglang.launch_server"],
    "cache_dir": "/root/.cache/sglang",
}

DEFAULT_ENGINE: dict[str, Any] = VLLM_ENGINE

ENGINE_PRESETS: dict[str, dict[str, Any]] = {
    "vllm": VLLM_ENGINE,
    "sglang": SGLANG_ENGINE,
}


def _copy_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Copy deeply enough that callers cannot mutate a module-level preset.

    ``dict(profile)`` is not enough: ``entrypoint`` is a list, so a shallow copy
    leaves the constant's list aliased and one ``.append`` would poison the
    default for the whole process.
    """
    return {**profile, "entrypoint": list(profile["entrypoint"])}


def _clean_shell_token(value: Any, *, field: str) -> str:
    """Non-empty single-line string. Rejects anything that could break out.

    These values are written into a generated Dockerfile (``RUN``/``ENTRYPOINT``)
    by PAR-57. They come from the campaign manifest — Pareton ops, never a miner
    — but a stray newline would silently inject extra Dockerfile directives, so
    reject control characters here rather than trusting the caller.
    """
    if not isinstance(value, str):
        raise ValueError(f"engine.{field} must be a string, got {type(value).__name__}")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"engine.{field} must be a non-empty string")
    bad = [
        c for c in cleaned if c == "\n" or c == "\r" or ord(c) < 0x20 or ord(c) == 0x7F
    ]
    if bad:
        raise ValueError(
            f"engine.{field} must not contain newlines or control characters"
        )
    return cleaned


def validate_engine(engine: dict[str, Any]) -> dict[str, Any]:
    """Validate a campaigns.engine block; return a normalized copy.

    Normalization is total: the returned dict has exactly the four known keys,
    so an equal profile always hashes identically regardless of key order or
    extra junk in the input.
    """
    if not isinstance(engine, dict):
        raise ValueError("engine must be an object")

    name = engine.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("engine.name must be a non-empty string")
    normalized_name = name.strip().lower()
    if normalized_name not in KNOWN_ENGINE_NAMES:
        raise ValueError(
            f"engine.name must be one of {sorted(KNOWN_ENGINE_NAMES)}, got {name!r}"
        )

    install_cmd = _clean_shell_token(engine.get("install_cmd"), field="install_cmd")

    entrypoint = engine.get("entrypoint")
    if not isinstance(entrypoint, (list, tuple)) or not entrypoint:
        raise ValueError("engine.entrypoint must be a non-empty list of strings")
    cleaned_entrypoint = [
        _clean_shell_token(part, field=f"entrypoint[{i}]")
        for i, part in enumerate(entrypoint)
    ]

    # cache_dir is the target half of ``docker -v <host>:<target>``, where ':'
    # is the delimiter. A value like "/root/.cache/vllm:ro" would parse as a
    # read-only mount and silently cost the baseline its warm compile cache,
    # with no error anywhere. The sibling fields never reach a mount spec, so
    # this check belongs here rather than in _clean_shell_token.
    cache_dir = _clean_shell_token(engine.get("cache_dir"), field="cache_dir")
    if ":" in cache_dir:
        raise ValueError(f"engine.cache_dir must not contain ':', got {cache_dir!r}")
    if not cache_dir.startswith("/"):
        raise ValueError(
            f"engine.cache_dir must be an absolute path, got {cache_dir!r}"
        )

    unknown = set(engine) - {"name", "install_cmd", "entrypoint", "cache_dir"}
    if unknown:
        raise ValueError(
            f"engine has unknown keys: {sorted(unknown)} "
            f"(allowed: ['cache_dir', 'entrypoint', 'install_cmd', 'name'])"
        )

    return {
        "name": normalized_name,
        "install_cmd": install_cmd,
        "entrypoint": cleaned_entrypoint,
        "cache_dir": cache_dir,
    }


def resolve_engine(engine: dict[str, Any] | None) -> dict[str, Any]:
    """Effective profile for a manifest: the pinned one, or the vLLM default.

    Use this at consumption sites (PAR-57's builder) so ``engine is None`` and an
    explicit vLLM block behave identically. Note they still hash *differently* —
    absent stays out of the manifest pin set to preserve pre-existing hashes.
    """
    if engine is None:
        return _copy_profile(DEFAULT_ENGINE)
    return validate_engine(engine)


def preset(name: str) -> dict[str, Any]:
    """Look up a known engine profile by name (for seeding and CLIs)."""
    key = str(name).strip().lower()
    if key not in ENGINE_PRESETS:
        raise ValueError(
            f"unknown engine preset {name!r}; known: {sorted(ENGINE_PRESETS)}"
        )
    return _copy_profile(ENGINE_PRESETS[key])
