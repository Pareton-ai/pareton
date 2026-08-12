"""Z-score promotion: not anomalously worse than calibrated baseline noise.

Per-metric z = (observed - mean) / std with positive = worse.
aggregate_z = max(z_scores); promote iff no hard error and aggregate_z < z_threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

# Observed values where larger means worse for the candidate.
# throughput_ratio is inverted (negated) so lower throughput → higher (worse).
PROMOTION_METRICS: tuple[str, ...] = (
    "mean_abs_logprob_diff",
    "max_abs_logprob_diff",
    "argmax_mismatch_rate",
    "neg_throughput_ratio",
    "p99_ttft_ms",
    "p99_itl_ms",
)


class PromoteError(ValueError):
    """Invalid calibration or incomplete metric vector."""


@dataclass(frozen=True)
class PromotionResult:
    z_scores: dict[str, float]
    aggregate_z: float
    promoted: bool
    observed: dict[str, float]
    report: dict[str, Any]


def extract_observed_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Pull the promotion metric vector from a full bench report."""
    corr = report.get("correctness") or {}
    perf = report.get("perf_screen") or {}
    sla = report.get("sla_bench") or {}
    cand = sla.get("candidate") or {}
    ttft = (cand.get("ttft_ms") or {}).get("p99")
    itl = (cand.get("itl_ms") or {}).get("p99")
    try:
        throughput_ratio = float(perf["throughput_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromoteError("missing perf_screen.throughput_ratio") from exc
    try:
        out = {
            "mean_abs_logprob_diff": float(corr["mean_abs_logprob_diff"]),
            "max_abs_logprob_diff": float(corr["max_abs_logprob_diff"]),
            "argmax_mismatch_rate": float(corr["argmax_mismatch_rate"]),
            "neg_throughput_ratio": -throughput_ratio,
            "p99_ttft_ms": float(ttft),
            "p99_itl_ms": float(itl),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PromoteError(f"incomplete metric vector: {exc}") from exc
    for name, val in out.items():
        if not math.isfinite(val):
            raise PromoteError(f"{name} is not finite: {val!r}")
    return out


def _metric_stats(calibration: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    metrics = calibration.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise PromoteError("calibration.metrics missing")
    out: dict[str, tuple[float, float]] = {}
    for name in PROMOTION_METRICS:
        block = metrics.get(name)
        if not isinstance(block, dict):
            raise PromoteError(f"calibration.metrics.{name} missing")
        try:
            mean = float(block["mean"])
            std = float(block["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PromoteError(f"calibration.metrics.{name} invalid: {exc}") from exc
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise PromoteError(
                f"calibration.metrics.{name} requires finite mean and std > 0"
            )
        out[name] = (mean, std)
    return out


def compute_z_scores(
    observed: Mapping[str, float],
    calibration: Mapping[str, Any],
) -> dict[str, float]:
    stats = _metric_stats(calibration)
    z: dict[str, float] = {}
    for name in PROMOTION_METRICS:
        if name not in observed:
            raise PromoteError(f"observed missing {name}")
        mean, std = stats[name]
        z[name] = (float(observed[name]) - mean) / std
    return z


def decide_promotion(
    *,
    report: Mapping[str, Any],
    calibration: Mapping[str, Any],
    z_threshold: float,
    hard_error: bool = False,
) -> PromotionResult:
    """Compute z-scores and promotion bit.

    ``hard_error`` True (engine crash / Module A fail) always yields promoted=False.
    """
    thr = float(z_threshold)
    if not math.isfinite(thr):
        raise PromoteError(f"z_threshold must be finite, got {z_threshold!r}")
    observed = extract_observed_metrics(report)
    z_scores = compute_z_scores(observed, calibration)
    aggregate_z = max(z_scores.values())
    promoted = (not hard_error) and aggregate_z < thr
    promo_report = {
        "verdict": "pass" if promoted else "fail_promotion",
        "z_scores": dict(z_scores),
        "aggregate_z": aggregate_z,
        "z_threshold": thr,
        "promoted": promoted,
        "observed": dict(observed),
        "hard_error": bool(hard_error),
    }
    return PromotionResult(
        z_scores=dict(z_scores),
        aggregate_z=aggregate_z,
        promoted=promoted,
        observed=dict(observed),
        report=promo_report,
    )
