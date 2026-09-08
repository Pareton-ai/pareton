"""Unit tests for the scorer's serve-arg overrides (no Docker/GPU)."""

from __future__ import annotations

import pytest

from bench.main import (
    CORRECTNESS_EXTRA_SERVE_ARGS,
    correctness_extra_serve_args,
    plan_round_starts,
    scorer_engine_spec,
)
from bench.schemas import EngineSpec, EnginesSpec


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


@pytest.mark.parametrize(
    "tp_args",
    [[], ["--tp-size", "1"], ["--tp-size=1"], ["--tensor-parallel-size", "1"]],
)
def test_sglang_serve_args_skip_vllm_correctness_extras(tp_args):
    args = [
        "--model-path",
        "/model",
        "--dtype",
        "auto",
        *tp_args,
        "--context-length",
        "131072",
    ]
    assert correctness_extra_serve_args("sglang") == []
    spec = EngineSpec(
        image="sha256:" + ("a" * 64),
        serve_args=list(args),
        name="sglang",
        cache_dir="/root/.cache/sglang",
    )
    out = scorer_engine_spec(spec)
    assert out.serve_args == args[:-1] + ["131079"]
    assert "--no-enable-prefix-caching" not in out.serve_args
    assert "--no-enable-flashinfer-autotune" not in out.serve_args
    assert out.name == "sglang"
    assert out.cache_dir == spec.cache_dir


@pytest.mark.parametrize(
    "context_args,scorer_context_args",
    [
        (["--context-length", "8192"], ["--context-length", "8199"]),
        (["--context-length=8192"], ["--context-length=8199"]),
        (
            ["--context-length", "4096", "--context-length=8192"],
            ["--context-length", "4103", "--context-length=8199"],
        ),
    ],
)
def test_sglang_context_headroom_is_only_applied_to_scorer(
    context_args, scorer_context_args
):
    args = ["--model-path", "/model", *context_args, "--tp-size", "1"]
    spec = EngineSpec(
        image="sha256:" + ("a" * 64),
        serve_args=list(args),
        env={"FOO": "1", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "0"},
        name="sglang",
    )
    plan = plan_round_starts(EnginesSpec(baseline=spec, candidates=[spec]))
    for start in plan:
        if start.kind == "scorer":
            assert start.spec.serve_args == [
                "--model-path",
                "/model",
                *scorer_context_args,
                "--tp-size",
                "1",
            ]
            assert start.spec.env == {
                "FOO": "1",
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            }
        else:
            assert start.spec.serve_args == args
            assert start.spec.env == {
                "FOO": "1",
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "0",
            }
