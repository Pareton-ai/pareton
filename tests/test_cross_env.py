"""WS-E aggregation + cross_env validation (offline, no DB/GPU)."""

from __future__ import annotations

import math

import pytest

from campaign.cross_env import validate_cross_env
from campaign.seed import build_seed_bench_spec
from campaign.store import derive_bench_verdict_from_events
from worker.bench_job import (
    SkuRunResult,
    aggregate_sku_outcomes,
    sku_speedup_value,
    worst_verdict,
)


def _result(
    sku: str,
    *,
    verdict: str = "pass",
    speedup: float | None = 1.2,
    error_role: str | None = None,
    metric: str = "output_tokens_per_s_ratio",
) -> SkuRunResult:
    report: dict = {
        "verdict": verdict,
        "sla_bench": {"verdict": verdict, "speedup": {}},
    }
    if speedup is not None:
        report["sla_bench"]["speedup"][metric] = speedup
    return SkuRunResult(
        gpu_sku=sku,
        task_id=f"task-{sku}",
        exit_code=0,
        report=report,
        report_sha256="sha256:" + ("a" * 64),
        verdict=verdict,
        error_role=error_role,
    )


def test_validate_cross_env_defaults():
    out = validate_cross_env({})
    assert out["aggregate"] == "min"
    assert out["min_speedup_each"] == 1.0
    assert out["speedup_metric"] == "output_tokens_per_s_ratio"


def test_validate_cross_env_rejects_bad_aggregate():
    with pytest.raises(ValueError, match="aggregate"):
        validate_cross_env({"aggregate": "median"})


def test_validate_cross_env_rejects_bad_metric():
    with pytest.raises(ValueError, match="speedup_metric"):
        validate_cross_env({"speedup_metric": "not_a_key"})


def test_build_seed_bench_spec_includes_cross_env():
    spec = build_seed_bench_spec()
    assert spec["cross_env"]["aggregate"] == "min"


def test_worst_verdict_precedence():
    assert worst_verdict(["pass", "fail_sla"]) == "fail_sla"
    assert worst_verdict(["fail_sla", "fail_correctness"]) == "fail_correctness"
    assert worst_verdict(["fail_perf_screen", "fail_sla"]) == "fail_perf_screen"
    assert worst_verdict(["pass", "pass"]) == "pass"


def test_aggregate_both_pass():
    v, role = aggregate_sku_outcomes(
        [_result("a"), _result("b")],
        cross_env=None,
    )
    assert v == "pass"
    assert role is None


def test_aggregate_fail_correctness_wins():
    v, _ = aggregate_sku_outcomes(
        [_result("a", verdict="pass"), _result("b", verdict="fail_correctness")],
        cross_env=None,
    )
    assert v == "fail_correctness"


def test_aggregate_fail_sla():
    v, _ = aggregate_sku_outcomes(
        [_result("a", verdict="fail_sla"), _result("b", verdict="pass")],
        cross_env=None,
    )
    assert v == "fail_sla"


def test_aggregate_candidate_error():
    v, role = aggregate_sku_outcomes(
        [
            _result("a", verdict="pass"),
            _result("b", verdict="error", error_role="candidate", speedup=None),
        ],
        cross_env=None,
    )
    assert v == "error"
    assert role == "candidate"


def test_aggregate_speedup_floor_pass_at_exact():
    ce = {
        "aggregate": "min",
        "min_speedup_each": 1.0,
        "speedup_metric": "output_tokens_per_s_ratio",
    }
    v, _ = aggregate_sku_outcomes(
        [_result("a", speedup=1.0), _result("b", speedup=1.5)],
        cross_env=ce,
    )
    assert v == "pass"


def test_aggregate_speedup_floor_shortfall():
    ce = {
        "aggregate": "min",
        "min_speedup_each": 1.1,
        "speedup_metric": "output_tokens_per_s_ratio",
    }
    v, _ = aggregate_sku_outcomes(
        [_result("a", speedup=1.0), _result("b", speedup=2.0)],
        cross_env=ce,
    )
    assert v == "fail_cross_env_speedup"


def test_aggregate_speedup_missing_key():
    ce = {
        "aggregate": "min",
        "min_speedup_each": 1.0,
        "speedup_metric": "output_tokens_per_s_ratio",
    }
    v, _ = aggregate_sku_outcomes(
        [_result("a", speedup=None), _result("b", speedup=2.0)],
        cross_env=ce,
    )
    assert v == "fail_cross_env_speedup"


def test_aggregate_speedup_nan():
    ce = {
        "aggregate": "min",
        "min_speedup_each": 1.0,
        "speedup_metric": "output_tokens_per_s_ratio",
    }
    r = _result("a", speedup=1.0)
    r.report["sla_bench"]["speedup"]["output_tokens_per_s_ratio"] = float("nan")
    v, _ = aggregate_sku_outcomes([r, _result("b", speedup=2.0)], cross_env=ce)
    assert v == "fail_cross_env_speedup"
    assert sku_speedup_value(r.report, "output_tokens_per_s_ratio") is None
    assert math.isnan(float("nan"))


def test_aggregate_speedup_zero_baseline_ratio():
    ce = {
        "aggregate": "min",
        "min_speedup_each": 1.0,
        "speedup_metric": "output_tokens_per_s_ratio",
    }
    v, _ = aggregate_sku_outcomes(
        [_result("a", speedup=0.0), _result("b", speedup=2.0)],
        cross_env=ce,
    )
    assert v == "fail_cross_env_speedup"


def test_no_floor_without_cross_env_even_if_slow():
    v, _ = aggregate_sku_outcomes(
        [_result("a", speedup=0.5), _result("b", speedup=0.5)],
        cross_env=None,
    )
    assert v == "pass"


def test_derive_bench_verdict_from_events():
    assert derive_bench_verdict_from_events([]) is None
    assert (
        derive_bench_verdict_from_events(
            [
                {"state": "correct", "detail": {}},
                {"state": "screened", "detail": {}},
            ]
        )
        is None
    )
    assert (
        derive_bench_verdict_from_events(
            [
                {"state": "correct", "detail": {}},
                {"state": "screened", "detail": {}},
                {"state": "benched", "detail": {}},
            ]
        )
        == "pass"
    )
    assert (
        derive_bench_verdict_from_events(
            [
                {"state": "correct", "detail": {}},
                {"state": "screened", "detail": {}},
                {"state": "rejected", "detail": {"reason": "fail_cross_env_speedup"}},
            ]
        )
        == "fail_cross_env_speedup"
    )
    # Gate reject must not masquerade as bench verdict.
    assert (
        derive_bench_verdict_from_events(
            [{"state": "rejected", "detail": {"reason": "fail_integrity"}}]
        )
        is None
    )
