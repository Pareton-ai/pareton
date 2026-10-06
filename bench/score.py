"""Scoring rule dispatch: per-prompt timings in, one round score out.

``campaigns.scoring_rule`` names the formula and is pinned in
``manifest_hash``; the resolved rule is copied onto ``rounds.scoring_rule`` so
every round records the formula that produced its numbers. Historical rounds
use ``median_e2e_speedup``; version 5 uses weighted full-tier completion speedup.

Pure math. No HTTP, no Docker, no database.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from bench.concurrency import WEIGHTED_RULE, tier_weights

# Minimum fraction of the baseline's output tokens a candidate must emit
# before it earns speed credit on that prompt. Overridable per campaign with
# a "tolerance" key on scoring_rule.
DEFAULT_SPEED_TOLERANCE: float = 0.9

# Why a prompt was forced to 0.0. Named because they are read back out of a
# stored report and counted: matching these strings at the call site would
# break silently the first time one is reworded.
REASON_NO_CANDIDATE_TIMING = "no candidate timing"
REASON_BASELINE_NO_TOKENS = "baseline emitted no tokens"
REASON_BELOW_TOLERANCE = "candidate output below tolerance"
REASON_INSUFFICIENT_TIMING = "insufficient timing"


@dataclass(frozen=True)
class PromptTiming:
    """One prompt's timings from one engine, as the SLA replay recorded them.

    With one token per chunk, ``itl_s`` holds each gap after the first token,
    so time to token k is ``ttft_s + sum(itl_s[: k - 1])``. Batched streams
    may lack those per-token timings; v5 validity uses completion evidence.
    """

    ttft_s: float
    itl_s: list[float] = field(default_factory=list)
    completion_tokens: int = 0
    finish_reason: str | None = None


@dataclass(frozen=True)
class PromptScore:
    """Per-prompt detail behind the round score.

    Absolute seconds are kept, not only the ratio, so cross-campaign
    questions stay answerable from ``round_entries.report``.
    """

    request_id: str
    speedup: float
    aligned_tokens: int
    baseline_e2e_s: float | None = None
    candidate_e2e_s: float | None = None
    reason: str | None = None
    candidate_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "speedup": self.speedup,
            "aligned_tokens": self.aligned_tokens,
            "baseline_e2e_s": self.baseline_e2e_s,
            "candidate_e2e_s": self.candidate_e2e_s,
            "reason": self.reason,
            "candidate_failed": self.candidate_failed,
        }


@dataclass(frozen=True)
class ScoreResult:
    score: float
    rule: str
    per_prompt: list[PromptScore]
    breakdown: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        """The ``round_entries.report`` payload for this entry."""
        return {
            "rule": self.rule,
            "score": self.score,
            "prompts": [p.to_dict() for p in self.per_prompt],
            "score_breakdown": self.breakdown,
        }


def aligned_e2e_s(timing: PromptTiming, aligned_k: int) -> float | None:
    """Wall time from request start to the aligned_k-th output token.

    None when the engine did not emit that many tokens, or when it emitted
    them without recording the gaps.
    """
    if aligned_k < 1 or timing.completion_tokens < aligned_k:
        return None
    if len(timing.itl_s) < aligned_k - 1:
        return None
    samples = [timing.ttft_s, *timing.itl_s[: aligned_k - 1]]
    if any(not math.isfinite(x) or x < 0 for x in samples):
        return None
    return timing.ttft_s + math.fsum(samples[1:])


def failure_penalty(rule: Mapping[str, Any]) -> float:
    value = rule.get("failure_penalty", 0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            "scoring_rule.failure_penalty must be a finite nonnegative number"
        )
    return float(value)


def _min_aligned_tokens(baseline_tokens: int, tolerance: float) -> int:
    if baseline_tokens < 2:
        return baseline_tokens
    return max(2, math.ceil(tolerance * baseline_tokens))


def prompt_speedup(
    request_id: str,
    baseline: PromptTiming,
    candidate: PromptTiming | None,
    *,
    tolerance: float = DEFAULT_SPEED_TOLERANCE,
) -> PromptScore:
    """End-to-end speedup on one prompt: (baseline - candidate) / baseline.

    Both engines are compared at the same output token count, so a candidate
    that stops early cannot buy speed by answering less. Stopping too early
    fails the tolerance gate outright and scores 0.0 for the prompt, which
    never clears the crown bar on its own.
    """
    # A failed candidate request only counts against a valid baseline reference.
    reference_e2e = aligned_e2e_s(baseline, baseline.completion_tokens)
    valid_reference = reference_e2e is not None and reference_e2e > 0
    if candidate is None:
        return PromptScore(
            request_id,
            0.0,
            0,
            reason=REASON_NO_CANDIDATE_TIMING,
            candidate_failed=valid_reference,
        )
    if baseline.completion_tokens < 1:
        return PromptScore(request_id, 0.0, 0, reason=REASON_BASELINE_NO_TOKENS)

    aligned_k = min(baseline.completion_tokens, candidate.completion_tokens)
    if aligned_k < _min_aligned_tokens(baseline.completion_tokens, tolerance):
        return PromptScore(
            request_id,
            0.0,
            aligned_k,
            reason=REASON_BELOW_TOLERANCE,
            candidate_failed=valid_reference,
        )

    base_e2e = aligned_e2e_s(baseline, aligned_k)
    cand_e2e = aligned_e2e_s(candidate, aligned_k)
    if base_e2e is None or cand_e2e is None or base_e2e <= 0:
        return PromptScore(
            request_id,
            0.0,
            aligned_k,
            baseline_e2e_s=base_e2e,
            candidate_e2e_s=cand_e2e,
            reason=REASON_INSUFFICIENT_TIMING,
            candidate_failed=valid_reference,
        )

    return PromptScore(
        request_id,
        (base_e2e - cand_e2e) / base_e2e,
        aligned_k,
        baseline_e2e_s=base_e2e,
        candidate_e2e_s=cand_e2e,
    )


def summarize_prompt_scores(
    prompts: Sequence[Mapping[str, Any]],
    *,
    rule: str | None = None,
) -> dict[str, Any]:
    """Counts behind one entry's score: how many prompts paid, and what did not.

    Reads the ``prompts`` array of a stored ``round_entries.report``, so it
    works on any past round without rerunning the bench.

    A prompt is "zeroed" when it carries a ``reason``, never when its speedup
    happens to be 0.0: a candidate exactly as fast as the baseline earns a
    real 0.0 and must not be counted as a failure. ``zeroed_by_reason`` keeps
    the reasons apart because they mean different things to a miner: the
    tolerance gate is the patch answering less, while a timing gap is the
    harness having nothing to compare.

    For weighted tier scoring, the existing scored/zeroed fields count valid
    completions/failures instead: per-token timings do not determine credit.
    Use the recorded candidate_failed flag, retaining the legacy fallback for
    older reports without it. The stored per-prompt diagnostics stay intact.
    """
    total = 0
    zeroed_by_reason: dict[str, int] = {}
    for p in prompts:
        if not isinstance(p, Mapping):
            continue
        total += 1
        reason = p.get("reason")
        if rule == WEIGHTED_RULE and isinstance(p.get("candidate_failed"), bool):
            reason = (
                reason or "candidate completion failed"
                if p["candidate_failed"]
                else None
            )
        if reason:
            key = str(reason)
            zeroed_by_reason[key] = zeroed_by_reason.get(key, 0) + 1
    zeroed = sum(zeroed_by_reason.values())
    return {
        "total": total,
        "scored": total - zeroed,
        "zeroed": zeroed,
        "below_tolerance": zeroed_by_reason.get(REASON_BELOW_TOLERANCE, 0),
        "zeroed_by_reason": zeroed_by_reason,
    }


def _median_e2e_speedup(
    rule: Mapping[str, Any],
    baseline: Mapping[str, PromptTiming],
    candidate: Mapping[str, PromptTiming],
) -> ScoreResult:
    """Median per-prompt e2e speedup. 0.35 means 35 percent faster."""
    tolerance = float(rule.get("tolerance", DEFAULT_SPEED_TOLERANCE))
    coefficient = failure_penalty(rule)
    per_prompt = [
        prompt_speedup(rid, baseline[rid], candidate.get(rid), tolerance=tolerance)
        for rid in baseline
    ]
    median = (
        float(statistics.median([p.speedup for p in per_prompt])) if per_prompt else 0.0
    )
    failed = sum(p.candidate_failed for p in per_prompt)
    rate = failed / len(per_prompt) if per_prompt else 0.0
    penalty = coefficient * rate
    return ScoreResult(
        score=median - penalty,
        rule="median_e2e_speedup",
        per_prompt=per_prompt,
        breakdown={
            "median_speedup": median,
            "scheduled_requests": len(per_prompt),
            "failed_requests": failed,
            "failure_rate": rate,
            "failure_penalty": coefficient,
            "penalty": penalty,
        },
    )


# Dispatch by name. campaign/models.py validates the name against
# SCORING_RULE_NAMES; the two sets are asserted equal in the tests.
def _weighted_tier_completion_speedup(
    rule, baseline, candidate, *, baseline_tiers=None, candidate_tiers=None
):
    """All eligible work contributes; incomplete work cannot buy speed credit."""
    weights = tier_weights(rule)
    coefficient = failure_penalty(rule)
    if not baseline or not baseline_tiers or not candidate_tiers:
        raise ValueError(
            "weighted score requires complete baseline and candidate tier evidence"
        )
    if set(baseline_tiers) != set(weights) or set(candidate_tiers) != set(weights):
        raise ValueError("weighted score requires every configured tier")
    if any(
        t.completion_tokens < 1 or t.finish_reason not in ("stop", "length")
        for t in baseline.values()
    ):
        raise ValueError("weighted score requires valid baseline completions")
    per_prompt = []
    for rid, timing in baseline.items():
        completion = candidate.get(rid)
        # Chunk timing is diagnostic only: speculative decoding can stream
        # several tokens per SSE chunk without per-token arrival timestamps.
        diagnostic = prompt_speedup(rid, timing, completion)
        failed = (
            completion is None
            or completion.completion_tokens
            < _min_aligned_tokens(timing.completion_tokens, DEFAULT_SPEED_TOLERANCE)
            or completion.finish_reason not in ("stop", "length")
        )
        per_prompt.append(
            replace(
                diagnostic,
                candidate_failed=failed,
                reason=(
                    "invalid completion finish reason"
                    if completion is not None
                    and completion.finish_reason not in ("stop", "length")
                    else diagnostic.reason
                ),
                speedup=0.0 if failed else diagnostic.speedup,
            )
        )
    details, seen = {}, set()
    for tier in weights:
        b, c = baseline_tiers[tier], candidate_tiers[tier]
        ids = b["request_ids"]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or seen.intersection(ids)
            or set(ids) != set(c["request_ids"])
        ):
            raise ValueError("tier membership differs between baseline and candidate")
        seen.update(ids)
        bt, ct = b["completion_s"], c["completion_s"]
        if any(
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not math.isfinite(t)
            or t <= 0
            for t in (bt, ct)
        ):
            raise ValueError("invalid tier completion time")
        details[tier] = {
            "weight": weights[tier],
            "baseline_completion_s": bt,
            "candidate_completion_s": ct,
            "speedup": 1 - ct / bt,
            "scheduled_requests": len(ids),
        }
    if seen != set(baseline) or set(candidate) - seen:
        raise ValueError("tier membership does not match eligible request timings")
    raw = sum(d["weight"] * d["speedup"] for d in details.values())
    failed = sum(p.candidate_failed for p in per_prompt)
    rate = failed / len(per_prompt)
    return ScoreResult(
        score=raw - coefficient * rate,
        rule=WEIGHTED_RULE,
        per_prompt=per_prompt,
        breakdown={
            "weighted_speedup": raw,
            "eligible_speedup": raw,
            "tiers": details,
            "scheduled_requests": len(per_prompt),
            "failed_requests": failed,
            "failure_rate": rate,
            "failure_penalty": coefficient,
            "penalty": coefficient * rate,
        },
    )


SCORING_RULES: dict[
    str,
    Callable[
        [Mapping[str, Any], Mapping[str, PromptTiming], Mapping[str, PromptTiming]],
        ScoreResult,
    ],
] = {
    "median_e2e_speedup": _median_e2e_speedup,
    WEIGHTED_RULE: _weighted_tier_completion_speedup,
}


def score_candidate(
    rule: Mapping[str, Any],
    *,
    baseline: Mapping[str, PromptTiming],
    candidate: Mapping[str, PromptTiming],
    baseline_tiers: Mapping[str, Any] | None = None,
    candidate_tiers: Mapping[str, Any] | None = None,
) -> ScoreResult:
    """Score one candidate against the round's baseline under a named rule.

    ``baseline`` and ``candidate`` map request id to timing. The baseline's
    prompts define the eligible set. Missing candidate timings count as failures;
    they never disappear from the denominator. Weighted scoring additionally
    requires complete-tier evidence with matching request membership.
    """
    name = str(rule.get("name") or "")
    impl = SCORING_RULES.get(name)
    if impl is None:
        raise ValueError(
            f"scoring_rule.name must be one of {sorted(SCORING_RULES)}, got {name!r}"
        )
    if name == WEIGHTED_RULE:
        return impl(
            rule,
            baseline,
            candidate,
            baseline_tiers=baseline_tiers,
            candidate_tiers=candidate_tiers,
        )
    return impl(rule, baseline, candidate)
