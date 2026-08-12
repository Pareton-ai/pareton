"""Unit tests for z-score promotion (positive = worse)."""

from __future__ import annotations

import pytest

from bench.promote import PromoteError, decide_promotion, extract_observed_metrics

pytestmark = pytest.mark.unit


def _report(
    *,
    mean: float = 0.01,
    max_d: float = 0.02,
    argmax: float = 0.0,
    throughput_ratio: float = 1.0,
    p99_ttft: float = 100.0,
    p99_itl: float = 20.0,
) -> dict:
    return {
        "correctness": {
            "mean_abs_logprob_diff": mean,
            "max_abs_logprob_diff": max_d,
            "argmax_mismatch_rate": argmax,
        },
        "perf_screen": {"throughput_ratio": throughput_ratio},
        "sla_bench": {
            "candidate": {
                "ttft_ms": {"p99": p99_ttft},
                "itl_ms": {"p99": p99_itl},
            }
        },
    }


def _calib(**means_stds: tuple[float, float]) -> dict:
    defaults = {
        "mean_abs_logprob_diff": (0.01, 0.01),
        "max_abs_logprob_diff": (0.02, 0.01),
        "argmax_mismatch_rate": (0.0, 0.01),
        "neg_throughput_ratio": (-1.0, 0.05),
        "p99_ttft_ms": (100.0, 10.0),
        "p99_itl_ms": (20.0, 5.0),
    }
    defaults.update(means_stds)
    return {
        "metrics": {
            k: {"mean": v[0], "std": v[1]} for k, v in defaults.items()
        }
    }


def test_extract_inverts_throughput():
    obs = extract_observed_metrics(_report(throughput_ratio=1.2))
    assert obs["neg_throughput_ratio"] == pytest.approx(-1.2)


def test_promote_when_below_threshold():
    # observed == mean ⇒ all z ≈ 0
    result = decide_promotion(
        report=_report(),
        calibration=_calib(),
        z_threshold=3.0,
    )
    assert result.aggregate_z == pytest.approx(0.0)
    assert result.promoted is True


def test_reject_when_aggregate_z_at_or_above_threshold():
    # mean_abs = 0.04, mean=0.01, std=0.01 ⇒ z=3.0; threshold 3.0 ⇒ not <
    result = decide_promotion(
        report=_report(mean=0.04),
        calibration=_calib(),
        z_threshold=3.0,
    )
    assert result.z_scores["mean_abs_logprob_diff"] == pytest.approx(3.0)
    assert result.aggregate_z == pytest.approx(3.0)
    assert result.promoted is False


def test_positive_z_means_worse_latency():
    result = decide_promotion(
        report=_report(p99_ttft=130.0),  # (130-100)/10 = 3.0
        calibration=_calib(),
        z_threshold=2.5,
    )
    assert result.z_scores["p99_ttft_ms"] == pytest.approx(3.0)
    assert result.promoted is False


def test_hard_error_never_promotes():
    result = decide_promotion(
        report=_report(),
        calibration=_calib(),
        z_threshold=3.0,
        hard_error=True,
    )
    assert result.promoted is False


def test_zero_std_rejected():
    with pytest.raises(PromoteError, match="std > 0"):
        decide_promotion(
            report=_report(),
            calibration=_calib(mean_abs_logprob_diff=(0.01, 0.0)),
            z_threshold=3.0,
        )
