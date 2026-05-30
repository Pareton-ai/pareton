"""Unit tests for validator.scoring -- pure math, no I/O."""

from __future__ import annotations

import pytest

from validator.scoring import (
    CorrectnessVerdict,
    SPEED_TOLERANCE_RATIO,
    aligned_e2e_improvement,
    aligned_e2e_seconds,
    compute_correctness,
    compute_pass1_aggregate_match,
    compute_speed_improvement,
    compute_teacher_forcing_verdict,
    compute_token_match_rate,
    pass1_match_passes,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# End-to-end speed (aligned_e2e_improvement)
# --------------------------------------------------------------------------- #


class TestAlignedE2eSeconds:
    def test_complete_timing(self):
        decode = [0.0, 0.05, 0.1, 0.2]
        assert aligned_e2e_seconds(0.5, decode, 4) == pytest.approx(0.7)

    def test_insufficient_decode_returns_none(self):
        assert aligned_e2e_seconds(0.5, [0.0], 4) is None


class TestAlignedE2eImprovement:
    def test_faster_miner(self):
        bl_decode = [0.0, 0.5, 1.0]
        mn_decode = [0.0, 0.4, 0.8]
        imp, bl_e2e, mn_e2e = aligned_e2e_improvement(
            1.0, bl_decode, 3, 0.8, mn_decode, 3
        )
        assert bl_e2e == pytest.approx(2.0)
        assert mn_e2e == pytest.approx(1.6)
        assert imp == pytest.approx(0.2)

    def test_slower_miner_floors_at_zero(self):
        imp, _, _ = aligned_e2e_improvement(1.0, [0.0, 1.0], 2, 2.0, [0.0, 2.0], 2)
        assert imp == 0.0

    def test_tolerance_band_rejects_short_output(self):
        bl_n = 512
        mn_n = 400
        bl_decode = [float(i) * 0.02 for i in range(bl_n)]
        mn_decode = [float(i) * 0.02 for i in range(mn_n)]
        imp, _, _ = aligned_e2e_improvement(
            0.6, bl_decode, bl_n, 0.5, mn_decode, mn_n, tolerance=SPEED_TOLERANCE_RATIO
        )
        assert imp == 0.0

    def test_tolerance_band_allows_ninety_percent(self):
        bl_n = 512
        mn_n = 461
        bl_decode = [float(i) * 0.02 for i in range(bl_n)]
        mn_decode = [float(i) * 0.01 for i in range(mn_n)]
        imp, _, _ = aligned_e2e_improvement(
            0.6, bl_decode, bl_n, 0.5, mn_decode, mn_n, tolerance=SPEED_TOLERANCE_RATIO
        )
        assert imp > 0.0

    def test_buffering_exploit_not_inflated(self):
        """Burst after long TTFT: modest e2e gain, not hundreds of x."""
        bl_n = 512
        bl_decode = [float(i) * (12.0 / (bl_n - 1)) for i in range(bl_n)]
        mn_decode = [0.0] * bl_n
        imp, bl_e2e, mn_e2e = aligned_e2e_improvement(
            0.6, bl_decode, bl_n, 12.0, mn_decode, bl_n
        )
        assert bl_e2e == pytest.approx(12.6)
        assert mn_e2e == pytest.approx(12.0)
        assert imp == pytest.approx((12.6 - 12.0) / 12.6, rel=1e-3)
        assert imp < 0.2


class TestComputeSpeedImprovement:
    def test_median_across_prompts(self):
        assert compute_speed_improvement([0.1, 0.2, 0.3]) == pytest.approx(0.2)

    def test_empty_returns_zero(self):
        assert compute_speed_improvement([]) == 0.0


# --------------------------------------------------------------------------- #
# compute_token_match_rate
# --------------------------------------------------------------------------- #


class TestTokenMatchRate:
    def test_exact_match(self):
        tokens = ["hello", "world", "foo"]
        assert compute_token_match_rate(tokens, tokens) == 1.0

    def test_one_mismatch_in_100(self):
        base = [f"t{i}" for i in range(100)]
        miner = list(base)
        miner[50] = "WRONG"
        assert compute_token_match_rate(base, miner) == pytest.approx(0.99)

    def test_all_different(self):
        base = ["a", "b", "c"]
        miner = ["x", "y", "z"]
        assert compute_token_match_rate(base, miner) == pytest.approx(0.0)

    def test_empty_both(self):
        assert compute_token_match_rate([], []) == 1.0

    def test_miner_shorter(self):
        base = ["a", "b", "c", "d"]
        miner = ["a", "b"]
        assert compute_token_match_rate(base, miner) == pytest.approx(0.5)

    def test_miner_longer(self):
        base = ["a"]
        miner = ["a", "b", "c"]
        assert compute_token_match_rate(base, miner) == pytest.approx(1 / 3)

    def test_single_token_match(self):
        assert compute_token_match_rate(["x"], ["x"]) == 1.0

    def test_single_token_mismatch(self):
        assert compute_token_match_rate(["x"], ["y"]) == 0.0


# --------------------------------------------------------------------------- #
# Pass 1 match gate
# --------------------------------------------------------------------------- #


class TestPass1MatchGate:
    def test_aggregate_match_mean(self):
        base = [["a", "b"], ["x", "y"]]
        miner = [["a", "b"], ["x", "z"]]
        assert compute_pass1_aggregate_match(base, miner) == pytest.approx(0.75)

    def test_pass_at_threshold(self):
        assert pass1_match_passes(0.25, 0.25) is True

    def test_fail_below_threshold(self):
        assert pass1_match_passes(0.24, 0.25) is False


# --------------------------------------------------------------------------- #
# compute_teacher_forcing_verdict (Pass 2 correctness)
# --------------------------------------------------------------------------- #


class TestComputeTeacherForcingVerdict:
    def test_high_logprobs_pass(self):
        v = compute_teacher_forcing_verdict(
            [-0.3, -0.5], scoring_canonical_token_count=2
        )
        assert v.passed is True
        assert v.token_match_rate == 0.0

    def test_empty_logprobs_infra_fail(self):
        v = compute_teacher_forcing_verdict([], scoring_canonical_token_count=2)
        assert v.passed is False
        assert (v.reason or "").startswith("scoring_infra_fail:")

    def test_garbage_fails_mean(self):
        v = compute_teacher_forcing_verdict(
            [-15.0, -20.0], scoring_canonical_token_count=2
        )
        assert v.passed is False
        assert "mean_logprob" in (v.reason or "")

    def test_catastrophic_scoring_coverage_infra_fail(self):
        """Near-empty extraction vs /tokenize count is infra fail."""
        logprobs = [-0.5] * 10
        v = compute_teacher_forcing_verdict(logprobs, scoring_canonical_token_count=50)
        assert v.passed is False
        assert (v.reason or "").startswith("scoring_infra_fail:")
        assert "10/50 scoring canonical tokens" in (v.reason or "")

    def test_partial_coverage_scores_extracted_logprobs(self):
        """EOS/suffix gap: score min(N, M) logprobs, do not infra-fail."""
        logprobs = [-0.3] * 256
        v = compute_teacher_forcing_verdict(logprobs, scoring_canonical_token_count=261)
        assert v.passed is True

    def test_cross_engine_stream_drift_not_compared(self):
        """Stream token count is never compared for DQ."""
        logprobs = [-0.3] * 161
        v = compute_teacher_forcing_verdict(logprobs, scoring_canonical_token_count=161)
        assert v.passed is True

    def test_equal_length_passes(self):
        logprobs = [-0.3, -0.5, -0.4]
        v = compute_teacher_forcing_verdict(logprobs, scoring_canonical_token_count=3)
        assert v.passed is True

    def test_extra_logprobs_still_passes(self):
        """More extracted logprobs than canonical count does not auto-fail."""
        v = compute_teacher_forcing_verdict(
            [-0.3, -0.5, -0.4], scoring_canonical_token_count=2
        )
        assert v.passed is True


# --------------------------------------------------------------------------- #
# compute_correctness (teacher-forcing gate)
# --------------------------------------------------------------------------- #


class TestComputeCorrectness:
    def test_high_logprobs_pass(self):
        """Legitimate miner: baseline assigns high logprob to each token."""
        base = ["a", "b", "c", "d", "e"]
        miner = ["a", "b", "c", "d", "e"]
        logprobs = [-0.3, -0.5, -0.2, -0.8, -0.4]
        v = compute_correctness(base, miner, logprobs)
        assert v.passed is True
        assert v.mean_logprob == pytest.approx(-0.44)
        assert v.min_logprob == pytest.approx(-0.8)

    def test_tp_cascade_still_passes(self):
        """TP cascade: tokens diverge but baseline still scores them well."""
        base = ["a", "b", "c", "d", "e"]
        miner = ["a", "b", "X", "Y", "Z"]
        logprobs = [-0.3, -0.5, -1.2, -1.5, -1.8]
        v = compute_correctness(base, miner, logprobs)
        assert v.passed is True
        assert v.token_match_rate == pytest.approx(2 / 5)

    def test_garbage_output_fails_mean_threshold(self):
        """Gaming miner: garbage tokens score extremely low."""
        base = ["a", "b", "c", "d", "e"]
        miner = ["a", "XX", "$$", "AB", "!!"]
        logprobs = [-0.3, -8.0, -15.0, -20.0, -18.0]
        v = compute_correctness(base, miner, logprobs)
        assert v.passed is False
        assert "mean_logprob" in (v.reason or "")

    def test_single_garbage_token_fails_min_threshold(self):
        """One injected garbage token triggers min_logprob gate."""
        base = ["a", "b", "c", "d", "e"]
        miner = ["a", "b", "c", "GARBAGE", "e"]
        logprobs = [-0.3, -0.5, -0.2, -15.0, -0.4]
        v = compute_correctness(base, miner, logprobs)
        assert v.passed is False
        assert "min_logprob" in (v.reason or "")

    def test_empty_logprobs_fails(self):
        """No scoring data available: deny by default."""
        base = ["a", "b"]
        miner = ["a", "b"]
        v = compute_correctness(base, miner, [])
        assert v.passed is False
        assert "no baseline scoring logprobs" in (v.reason or "")

    def test_custom_thresholds(self):
        """Thresholds are configurable."""
        base = ["a", "b", "c"]
        miner = ["a", "b", "c"]
        logprobs = [-3.0, -3.5, -3.2]
        # Default threshold -4.0 → passes
        v = compute_correctness(base, miner, logprobs)
        assert v.passed is True
        # Stricter threshold → fails
        v = compute_correctness(base, miner, logprobs, mean_logprob_threshold=-2.0)
        assert v.passed is False

    def test_borderline_mean_passes(self):
        """Exactly at threshold passes (threshold is exclusive lower bound)."""
        base = ["a", "b"]
        miner = ["a", "b"]
        logprobs = [-4.0, -4.0]
        v = compute_correctness(base, miner, logprobs)
        assert v.passed is True  # -4.0 is not < -4.0

    def test_borderline_mean_fails(self):
        """Just below threshold fails."""
        base = ["a", "b"]
        miner = ["a", "b"]
        logprobs = [-4.0, -4.01]
        v = compute_correctness(base, miner, logprobs)
        assert v.passed is False

    def test_real_exploit_scenario(self):
        """Reproduce the actual gaming attack: 2 correct tokens, 254 garbage."""
        base = ["The", "Roman"] + [f"t{i}" for i in range(254)]
        miner = ["The", "Roman"] + ["GARBAGE"] * 254
        # First 2 tokens score well, rest are garbage
        logprobs = [-0.1, -0.2] + [-25.0] * 254
        v = compute_correctness(base, miner, logprobs)
        assert v.passed is False
        assert v.mean_logprob < -20.0
