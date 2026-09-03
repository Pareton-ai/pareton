"""Scoring rule dispatch: per-prompt timings in, one round score out.

``campaigns.scoring_rule`` names the formula and is pinned in
``manifest_hash``; the resolved rule is copied onto ``rounds.scoring_rule`` so
every round records the formula that produced its numbers. One rule ships:
``median_e2e_speedup``.

Pure math. No HTTP, no Docker, no database.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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

    ``itl_s`` holds the gap before each output token after the first, so the
    wall time to token k is ``ttft_s + sum(itl_s[: k - 1])``.
    """

    ttft_s: float
    itl_s: list[float] = field(default_factory=list)
    completion_tokens: int = 0


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "speedup": self.speedup,
            "aligned_tokens": self.aligned_tokens,
            "baseline_e2e_s": self.baseline_e2e_s,
            "candidate_e2e_s": self.candidate_e2e_s,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScoreResult:
    score: float
    rule: str
    per_prompt: list[PromptScore]

    def to_report(self) -> dict[str, Any]:
        """The ``round_entries.report`` payload for this entry."""
        return {
            "rule": self.rule,
            "score": self.score,
            "prompts": [p.to_dict() for p in self.per_prompt],
        }


def aligned_e2e_s(timing: PromptTiming, aligned_k: int) -> float | None:
    """Wall time from request start to the aligned_k-th output token.

    None when the engine did not emit that many tokens, or when it emitted
    them without recording the gaps.
    """
    if aligned_k < 1 or timing.completion_tokens < aligned_k:
        return None
    if aligned_k == 1:
        return timing.ttft_s
    if len(timing.itl_s) < aligned_k - 1:
        return None
    return timing.ttft_s + math.fsum(timing.itl_s[: aligned_k - 1])


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
    if candidate is None:
        return PromptScore(request_id, 0.0, 0, reason=REASON_NO_CANDIDATE_TIMING)
    if baseline.completion_tokens < 1:
        return PromptScore(request_id, 0.0, 0, reason=REASON_BASELINE_NO_TOKENS)

    aligned_k = min(baseline.completion_tokens, candidate.completion_tokens)
    if aligned_k < _min_aligned_tokens(baseline.completion_tokens, tolerance):
        return PromptScore(request_id, 0.0, aligned_k, reason=REASON_BELOW_TOLERANCE)

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
    """
    total = 0
    zeroed_by_reason: dict[str, int] = {}
    for p in prompts:
        if not isinstance(p, Mapping):
            continue
        total += 1
        reason = p.get("reason")
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
    per_prompt = [
        prompt_speedup(rid, baseline[rid], candidate.get(rid), tolerance=tolerance)
        for rid in baseline
    ]
    score = statistics.median([p.speedup for p in per_prompt]) if per_prompt else 0.0
    return ScoreResult(
        score=float(score), rule="median_e2e_speedup", per_prompt=per_prompt
    )


# Dispatch by name. campaign/models.py validates the name against
# SCORING_RULE_NAMES; the two sets are asserted equal in the tests.
SCORING_RULES: dict[
    str,
    Callable[
        [Mapping[str, Any], Mapping[str, PromptTiming], Mapping[str, PromptTiming]],
        ScoreResult,
    ],
] = {"median_e2e_speedup": _median_e2e_speedup}


def score_candidate(
    rule: Mapping[str, Any],
    *,
    baseline: Mapping[str, PromptTiming],
    candidate: Mapping[str, PromptTiming],
) -> ScoreResult:
    """Score one candidate against the round's baseline under a named rule.

    ``baseline`` and ``candidate`` map request id to timing. The baseline's
    prompts define the set: a prompt the candidate never answered scores 0.0
    rather than dropping out of the median.
    """
    name = str(rule.get("name") or "")
    impl = SCORING_RULES.get(name)
    if impl is None:
        raise ValueError(
            f"scoring_rule.name must be one of {sorted(SCORING_RULES)}, got {name!r}"
        )
    return impl(rule, baseline, candidate)
