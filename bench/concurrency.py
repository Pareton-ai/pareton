"""Versioned tier scheduling and complete-group timing contracts."""

from __future__ import annotations

import math
import statistics

TIERS = ("2k", "4k", "8k", "16k")
CONCURRENCIES = (1, 2, 4, 8, 16, 32)
WEIGHTED_RULE = "weighted_tier_completion_speedup"


def validate_concurrency(value):
    if type(value) is not int or value not in CONCURRENCIES:
        raise ValueError(f"request_concurrency must be one of {CONCURRENCIES}")
    return value


def validate_request_timeout(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("request_timeout_s must be a finite positive number")
    return float(value)


def input_tiers(value=None):
    """Validate an explicitly selected tier set; omission preserves old contracts."""
    if value is None:
        return TIERS
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(t, str) or t not in TIERS for t in value)
        or tuple(value) != tuple(t for t in TIERS if t in value)
    ):
        raise ValueError(
            "input_tiers must be a nonempty ordered subset of 2k, 4k, 8k, 16k"
        )
    return tuple(value)


def validate_tier_contract(sampling, scoring):
    if set(input_tiers(sampling.get("input_tiers"))) != set(tier_weights(scoring)):
        raise ValueError("tier_weights must match sampling input_tiers")


def tier_weights(rule):
    weights = rule.get("tier_weights", dict.fromkeys(TIERS, 0.25))
    if not isinstance(weights, dict) or not weights or not set(weights) <= set(TIERS):
        raise ValueError(
            "tier_weights must name a nonempty subset of 2k, 4k, 8k and 16k"
        )
    if any(
        isinstance(w, bool)
        or not isinstance(w, (int, float))
        or not math.isfinite(w)
        or w < 0
        for w in weights.values()
    ) or not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("tier_weights must be finite, nonnegative and sum to one")
    return {tier: float(weights[tier]) for tier in TIERS if tier in weights}


def request_groups(requests, concurrency, tiers=None):
    from bench.lifecycle import EngineError

    tiers = input_tiers(tiers)
    validate_concurrency(concurrency)
    tiers_per_group = max(1, concurrency // 8)
    if any(r.input_length_group not in tiers for r in requests):
        raise EngineError("concurrency replay requires LongWriter input tiers")
    if any(not any(r.input_length_group == t for r in requests) for t in tiers):
        raise EngineError(
            "concurrency replay requires a nonempty eligible set in every tier"
        )
    groups = []
    for offset in range(0, len(tiers), tiers_per_group):
        group_tiers = tiers[offset : offset + tiers_per_group]
        # Round-robin tier admission keeps the initial mix independent of trace
        # selection density. FIFO order within each tier stays frozen.
        queues = [
            [r for r in requests if r.input_length_group == t] for t in group_tiers
        ]
        groups.append(
            [
                queue[i]
                for i in range(max(map(len, queues)))
                for queue in queues
                if i < len(queue)
            ]
        )
    return groups


def tier_completion_metrics(rows, repetitions, tiers=None):
    """Reconstruct full-tier durations from complete, eligible replay rows."""
    from bench.lifecycle import EngineError

    tiers = input_tiers(tiers)
    if any(r["input_length_group"] not in tiers for r in rows):
        raise EngineError("tier evidence contains an unselected input tier")
    result = {}
    for tier in tiers:
        samples, expected_ids = [], None
        for rep in range(1, repetitions + 1):
            selected = [
                r for r in rows if r["rep"] == rep and r["input_length_group"] == tier
            ]
            ids = sorted(r["request_id"] for r in selected)
            if (
                not ids
                or len(set(ids)) != len(ids)
                or (expected_ids is not None and ids != expected_ids)
            ):
                raise EngineError(
                    "tier evidence has missing, duplicate or changed request IDs"
                )
            expected_ids = ids
            if any(r.get("error") for r in selected):
                raise EngineError(
                    "failed responses cannot form a tier completion measurement"
                )
            starts = {r["group_start_offset_ms"] for r in selected}
            if len(starts) != 1:
                raise EngineError("tier evidence must share a group start")
            elapsed = (
                max(r["protocol_completion_offset_ms"] for r in selected) - starts.pop()
            ) / 1000
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise EngineError("invalid tier completion duration")
            samples.append(elapsed)
        median = statistics.median(samples)
        result[tier] = {
            "request_ids": expected_ids,
            "repetitions_s": samples,
            "completion_s": median,
            "relative_range": (max(samples) - min(samples)) / median,
        }
    return result


def concurrency_observations(rows):
    """Observed occupied client slots, including refill gaps and final drain."""
    observations = []
    keys = sorted({(r["rep"], r["group_start_offset_ms"]) for r in rows})
    for rep, start in keys:
        group = [
            r for r in rows if (r["rep"], r["group_start_offset_ms"]) == (rep, start)
        ]
        events = sorted(
            [(r["admission_offset_ms"], 1) for r in group]
            + [(r["slot_release_offset_ms"], -1) for r in group]
        )
        previous, occupied, peak, area = start, 0, 0, 0.0
        for timestamp, delta in events:
            area += occupied * (timestamp - previous)
            occupied += delta
            peak = max(peak, occupied)
            previous = timestamp
        duration = previous - start
        observations.append(
            {
                "rep": rep,
                "group_start_offset_ms": start,
                "request_ids": sorted(r["request_id"] for r in group),
                "requested_concurrency": group[0]["requested_concurrency"],
                "effective_concurrency": group[0]["effective_concurrency"],
                "observed_peak": peak,
                "token_timing_unavailable_requests": sorted(
                    r["request_id"]
                    for r in group
                    if r.get("token_timing_available") is False
                ),
                "time_weighted_mean": area / duration if duration > 0 else 0,
            }
        )
    return observations
