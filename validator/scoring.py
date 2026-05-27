"""Correctness checking and scoring for containerized evaluation.

Pure math -- no I/O, no Docker, no bittensor. All inputs are lists of
floats or strings produced by the HTTP client in ``docker_eval``.
"""

from __future__ import annotations

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


def compute_aligned_throughput_tps(
    target_token_count: int,
    decode_elapsed_secs: list[float],
) -> float:
    """Throughput using a fixed token budget (baseline output length).

    TPS = target_token_count / time from first to target_token_count-th token.
    Returns 0.0 if the stream ended early, target < 2, or elapsed <= 0.
    """
    if target_token_count < 2:
        return 0.0
    if len(decode_elapsed_secs) < target_token_count:
        return 0.0
    elapsed = decode_elapsed_secs[target_token_count - 1] - decode_elapsed_secs[0]
    if elapsed <= 0:
        return 0.0
    return target_token_count / elapsed


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
    """Mean token match rate across Pass 1 stress prompt pairs."""
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


def compute_teacher_forcing_verdict(
    miner_tokens: list[str],
    scoring_logprobs: list[float],
    *,
    mean_logprob_threshold: float = -4.0,
    min_logprob_threshold: float = -12.0,
) -> CorrectnessVerdict:
    """Pass 2 audit gate: teacher-forcing logprobs only (no baseline token match).

    Empty logprobs indicate scoring infra failure (OOM, HTTP 500), not
    miner correctness failure.
    """
    if not scoring_logprobs:
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=0.0,
            reason="scoring_infra_fail: no baseline scoring logprobs available",
        )

    if len(scoring_logprobs) < len(miner_tokens):
        return CorrectnessVerdict(
            passed=False,
            token_match_rate=0.0,
            reason=(
                "correctness_fail: teacher-forcing covered "
                f"{len(scoring_logprobs)}/{len(miner_tokens)} streamed tokens"
            ),
        )

    mean_lp = statistics.mean(scoring_logprobs)
    min_lp = min(scoring_logprobs)

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


def compute_improvements(
    baseline_ttfts: list[float],
    miner_ttfts: list[float],
    baseline_tps_list: list[float],
    miner_tps_list: list[float],
) -> tuple[float, float, float]:
    """Compute the final score from per-prompt timing measurements.

    Returns ``(score, ttft_improvement, throughput_improvement)``.

    1. Take median TTFT and median throughput across prompts.
    2. Compute relative improvement vs. baseline, floored at 0.
    3. Score = 0.5 * ttft_improvement + 0.5 * throughput_improvement.
    """
    if not baseline_ttfts or not miner_ttfts:
        return (0.0, 0.0, 0.0)
    if not baseline_tps_list or not miner_tps_list:
        return (0.0, 0.0, 0.0)

    med_bl_ttft = statistics.median(baseline_ttfts)
    med_mn_ttft = statistics.median(miner_ttfts)
    med_bl_tps = statistics.median(baseline_tps_list)
    med_mn_tps = statistics.median(miner_tps_list)

    if med_bl_ttft > 0:
        ttft_imp = max(0.0, (med_bl_ttft - med_mn_ttft) / med_bl_ttft)
    else:
        ttft_imp = 0.0

    if med_bl_tps > 0:
        tps_imp = max(0.0, (med_mn_tps - med_bl_tps) / med_bl_tps)
    else:
        tps_imp = 0.0

    score = 0.5 * ttft_imp + 0.5 * tps_imp
    return (score, ttft_imp, tps_imp)
