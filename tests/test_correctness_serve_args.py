"""Unit tests for the scorer's serve-arg overrides (no Docker/GPU)."""

from __future__ import annotations

from bench.main import (
    CORRECTNESS_EXTRA_SERVE_ARGS,
    scorer_engine_spec,
    correctness_extra_serve_args,
)
from bench.schemas import EngineSpec


def test_scorer_engine_spec_appends_flags_without_mutating():
    original_args = ["--model", "/model", "--dtype", "bfloat16"]
    spec = EngineSpec(
        image="sha256:" + ("a" * 64),
        serve_args=list(original_args),
        env={"FOO": "1"},
    )
    out = scorer_engine_spec(spec)
    assert out.serve_args == original_args + list(CORRECTNESS_EXTRA_SERVE_ARGS)
    assert "--no-enable-prefix-caching" in out.serve_args
    assert "--no-enable-flashinfer-autotune" in out.serve_args
    assert spec.serve_args == original_args
    assert out.image == spec.image
    assert out.env == {"FOO": "1"}
    assert out.env is not spec.env


def test_sglang_serve_args_skip_vllm_correctness_extras():
    args = [
        "--model",
        "/model",
        "--dtype",
        "auto",
        "--tp-size",
        "8",
        "--context-length",
        "131072",
    ]
    assert correctness_extra_serve_args(args) == []
    spec = EngineSpec(image="sha256:" + ("a" * 64), serve_args=list(args))
    out = scorer_engine_spec(spec)
    assert out.serve_args == args
    assert "--no-enable-prefix-caching" not in out.serve_args
    assert "--no-enable-flashinfer-autotune" not in out.serve_args
