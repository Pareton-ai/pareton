"""Correctness checking and scoring for containerized evaluation.

Pure math -- no I/O, no Docker, no bittensor. All inputs are lists of
floats or strings produced by the HTTP client in ``docker_eval``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CorrectnessVerdict:
    """Result of teacher-forcing correctness gate.

    The baseline model independently scores the miner's output tokens.
    If the mean or min logprob falls below thresholds, the output is
    implausible under the real model and the miner is DQ'd.
    """

    passed: bool
    token_match_rate: float
    mean_logprob: float = 0.0
    min_logprob: float = 0.0
    reason: str | None = None


SPEED_TOLERANCE_RATIO: float = 0.9
"""Minimum fraction of baseline output tokens a miner must emit before
end-to-end speed credit applies (``ceil(ratio * baseline_N)``)."""


def _min_aligned_token_count(baseline_n: int, tolerance: float) -> int:
    if baseline_n < 2:
        return baseline_n
    return max(2, math.ceil(tolerance * baseline_n))


def aligned_e2e_seconds(
    ttft_s: float,
    decode_elapsed_secs: list[float],
    aligned_k: int,
) -> float | None:
    """Wall time from request start to the aligned_k-th token.

    Returns None when timing data is insufficient.
    """
    if aligned_k < 1:
        return None
    if len(decode_elapsed_secs) < aligned_k:
        return None
    return ttft_s + decode_elapsed_secs[aligned_k - 1]


def aligned_e2e_improvement(
    bl_ttft: float,
    bl_decode: list[float],
    bl_n: int,
    mn_ttft: float,
    mn_decode: list[float],
    mn_n: int,
    *,
    tolerance: float = SPEED_TOLERANCE_RATIO,
) -> tuple[float, float | None, float | None]:
    """Per-prompt end-to-end speed improvement vs baseline.

    Returns ``(improvement, baseline_e2e_s, miner_e2e_s)``. Improvement is
    0.0 when the tolerance gate fails or timing is insufficient.
    """
    if bl_n < 1:
        return (0.0, None, None)

    k = min(bl_n, mn_n)
    min_k = _min_aligned_token_count(bl_n, tolerance)
    if k < min_k:
        return (0.0, None, None)

    bl_e2e = aligned_e2e_seconds(bl_ttft, bl_decode, k)
    mn_e2e = aligned_e2e_seconds(mn_ttft, mn_decode, k)
    if bl_e2e is None or mn_e2e is None or bl_e2e <= 0:
        return (0.0, bl_e2e, mn_e2e)

    improvement = max(0.0, (bl_e2e - mn_e2e) / bl_e2e)
    return (improvement, bl_e2e, mn_e2e)


def compute_speed_improvement(improvements: list[float]) -> float:
    """Median per-prompt e2e improvement; empty list -> 0.0."""
    if not improvements:
        return 0.0
    return statistics.median(improvements)


def compute_token_match_rate(
    baseline_tokens: list[str],
    miner_tokens: list[str],
) -> float:
    """Fraction of positions where tokens agree (0.0 -- 1.0).

    Length mismatch: positions beyond the shorter list count as mismatches.
    """
    if not baseline_tokens and not miner_tokens:
        return 1.0
    total = max(len(baseline_tokens), len(miner_tokens))
    matches = sum(b == m for b, m in zip(baseline_tokens, miner_tokens))
    return matches / total


def compute_pass1_aggregate_match(
    baseline_tokens_list: list[list[str]],
    miner_tokens_list: list[list[str]],
) -> float:
    """Mean token match rate across Pass 1 speed prompt pairs."""
    if not baseline_tokens_list or not miner_tokens_list:
        return 0.0
    n = min(len(baseline_tokens_list), len(miner_tokens_list))
    rates = [
        compute_token_match_rate(baseline_tokens_list[i], miner_tokens_list[i])
        for i in range(n)
    ]
    return statistics.mean(rates)


def pass1_match_passes(aggregate_match: float, threshold: float) -> bool:
    """Return True when aggregate match meets or exceeds the Pass 1 gate."""
    return aggregate_match >= threshold


# Pass 2: minimum fraction of scoring_canonical_token_count that must be
# covered by extracted prompt_logprobs. Gaps from EOS/special-token suffixes
# (e.g. 256/261) are OK; near-empty extraction (e.g. 10/50) is infra fail.
SCORING_LOGPROB_MIN_COVERAGE_RATIO: float = 0.5


def compute_teacher_forcing_verdict(
    scoring_logprobs: list[float],
    *,
    scoring_canonical_token_count: int,
    mean_logprob_threshold: float = -4.0,
    min_logprob_threshold: float = -12.0,
    min_coverage_ratio: float = SCORING_LOGPROB_MIN_COVERAGE_RATIO,
) -> CorrectnessVerdict:
    """Pass 2 correctness gate: teacher-forcing logprobs on scoring canonical tokens.

    Nomenclature (see docs/evaluation/scoring):
      - challenger_output_text: miner SSE text (scored as-is)
      - challenger_stream_token_count: miner SSE logprob token count (telemetry)
      - scoring_canonical_token_count: scoring vLLM /tokenize assistant positions (N)
      - scoring_logprob_count: len(scoring_logprobs) from prompt_logprobs extraction (M)

    Correctness runs on min(N, M) extracted logprobs (both from scoring vLLM).
    Never compare M to challenger_stream_token_count for DQ.
    """
    scoring_logprob_count = len(scoring_logprobs)
    if scoring_logprob_count == 0:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=0.0,
            reason="scoring_infra_fail: no scoring logprobs available",
        )

    if scoring_canonical_token_count > 0:
        coverage = scoring_logprob_count / scoring_canonical_token_count
        if coverage < min_coverage_ratio:
            return CorrectnessVerdict(
                passed=False,
                token_match_rate=0.0,
                reason=(
                    "scoring_infra_fail: extracted "
                    f"{scoring_logprob_count}/{scoring_canonical_token_count} "
                    f"scoring canonical tokens (below {min_coverage_ratio:.0%} coverage floor)"
                ),
            )

    scored_logprobs = scoring_logprobs[
        : min(scoring_canonical_token_count, scoring_logprob_count)
        if scoring_canonical_token_count > 0
        else scoring_logprob_count
    ]

    mean_lp = statistics.mean(scored_logprobs)
    min_lp = min(scored_logprobs)

    if mean_lp < mean_logprob_threshold:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=0.0,
            mean_logprob=mean_lp,
            min_logprob=min_lp,
            reason=(
                f"mean_logprob {mean_lp:.3f} below threshold {mean_logprob_threshold}"
            ),
        )

    if min_lp < min_logprob_threshold:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=0.0,
            mean_logprob=mean_lp,
            min_logprob=min_lp,
            reason=(
                f"min_logprob {min_lp:.3f} below threshold {min_logprob_threshold}"
            ),
        )

    return CorrectnessVerdict(
        passed=True,
        token_match_rate=0.0,
        mean_logprob=mean_lp,
        min_logprob=min_lp,
    )


def compute_correctness(
    baseline_tokens: list[str],
    miner_tokens: list[str],
    baseline_scoring_logprobs: list[float],
    *,
    mean_logprob_threshold: float = -4.0,
    min_logprob_threshold: float = -12.0,
) -> CorrectnessVerdict:
    """Teacher-forcing correctness gate.

    After the miner streams its output, the baseline model scores
    each miner token in a single forward pass (teacher-forced). This
    returns the per-token logprob assigned by the baseline to each of
    the miner's tokens given the miner's own preceding context.

    A legitimate miner running the same model will produce tokens that
    the baseline assigns high probability to. A gaming miner dumping
    garbage tokens will produce tokens the baseline assigns near-zero
    probability.

    Thresholds:
      - mean_logprob_threshold: average logprob across all miner tokens
        must be above this. Real cross-model outputs typically score
        -0.5 to -2.0; garbage scores -15 to -30.
      - min_logprob_threshold: no single token may score below this.
        Catches isolated garbage tokens injected into otherwise-valid
        output.
    """
    rate = compute_token_match_rate(baseline_tokens, miner_tokens)

    if not baseline_scoring_logprobs:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=rate,
            reason="no baseline scoring logprobs available",
        )

    mean_lp = statistics.mean(baseline_scoring_logprobs)
    min_lp = min(baseline_scoring_logprobs)

    if mean_lp < mean_logprob_threshold:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=rate,
            mean_logprob=mean_lp,
            min_logprob=min_lp,
            reason=(
                f"mean_logprob {mean_lp:.3f} below threshold {mean_logprob_threshold}"
            ),
        )

    if min_lp < min_logprob_threshold:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=rate,
            mean_logprob=mean_lp,
            min_logprob=min_lp,
            reason=(
                f"min_logprob {min_lp:.3f} below threshold {min_logprob_threshold}"
            ),
        )

    return CorrectnessVerdict(
        passed=True,
        token_match_rate=rate,
        mean_logprob=mean_lp,
        min_logprob=min_lp,
    )
