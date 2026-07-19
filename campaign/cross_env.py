"""Campaign bench.cross_env shape (WS-E speedup floor)."""

from __future__ import annotations

SPEEDUP_METRIC_KEYS = frozenset(
    {
        "output_tokens_per_s_ratio",
        "requests_per_s_ratio",
        "p99_ttft_ratio",
        "p99_itl_ratio",
        "p99_e2e_ratio",
    }
)

DEFAULT_CROSS_ENV: dict = {
    "aggregate": "min",
    "min_speedup_each": 1.0,
    "speedup_metric": "output_tokens_per_s_ratio",
}


def validate_cross_env(cross_env: dict) -> dict:
    """Validate campaigns.bench.cross_env shape; return a normalized copy."""
    if not isinstance(cross_env, dict):
        raise ValueError("cross_env must be an object")
    aggregate = cross_env.get("aggregate", "min")
    if aggregate != "min":
        raise ValueError(f"cross_env.aggregate must be 'min', got {aggregate!r}")
    min_speedup = cross_env.get("min_speedup_each", 1.0)
    try:
        min_speedup_f = float(min_speedup)
    except (TypeError, ValueError) as exc:
        raise ValueError("cross_env.min_speedup_each must be a number") from exc
    metric = cross_env.get("speedup_metric", "output_tokens_per_s_ratio")
    if metric not in SPEEDUP_METRIC_KEYS:
        raise ValueError(
            f"cross_env.speedup_metric must be one of {sorted(SPEEDUP_METRIC_KEYS)}, "
            f"got {metric!r}"
        )
    return {
        "aggregate": "min",
        "min_speedup_each": min_speedup_f,
        "speedup_metric": str(metric),
    }
