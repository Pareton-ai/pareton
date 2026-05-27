"""Unit tests for validator.scoring -- pure math, no I/O."""

from __future__ import annotations

import pytest

from validator.scoring import (
    CorrectnessVerdict,
    compute_aligned_throughput_tps,
    compute_correctness,
    compute_improvements,
    compute_pass1_aggregate_match,
    compute_teacher_forcing_verdict,
    compute_token_match_rate,
    pass1_match_passes,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# compute_aligned_throughput_tps
# --------------------------------------------------------------------------- #


class TestComputeAlignedThroughputTps:
    def test_complete_stream(self):
        elapsed = [0.0, 0.05, 0.1, 0.2]
        assert compute_aligned_throughput_tps(4, elapsed) == pytest.approx(4 / 0.2)

    def test_early_stop_returns_zero(self):
        elapsed = [0.0, 0.05]
        assert compute_aligned_throughput_tps(4, elapsed) == 0.0

    def test_target_less_than_two_returns_zero(self):
        assert compute_aligned_throughput_tps(1, [0.0, 0.1]) == 0.0

    def test_zero_elapsed_returns_zero(self):
        assert compute_aligned_throughput_tps(3, [0.0, 0.0, 0.0]) == 0.0

    def test_empty_elapsed_returns_zero(self):
        assert compute_aligned_throughput_tps(3, []) == 0.0


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
# compute_teacher_forcing_verdict (Pass 2 audit)
# --------------------------------------------------------------------------- #


class TestComputeTeacherForcingVerdict:
    def test_high_logprobs_pass(self):
        v = compute_teacher_forcing_verdict(["a", "b"], [-0.3, -0.5])
        assert v.passed is True
        assert v.token_match_rate == 0.0

    def test_empty_logprobs_infra_fail(self):
        v = compute_teacher_forcing_verdict(["a", "b"], [])
        assert v.passed is False
        assert (v.reason or "").startswith("scoring_infra_fail:")

    def test_garbage_fails_mean(self):
        v = compute_teacher_forcing_verdict(["a", "b"], [-15.0, -20.0])
        assert v.passed is False
        assert "mean_logprob" in (v.reason or "")

    def test_incomplete_coverage_fails_even_with_high_logprobs(self):
        """Early EOS / truncated scoring: short logprob list vs long stream."""
        miner = [f"t{i}" for i in range(50)]
        logprobs = [-0.5] * 10
        v = compute_teacher_forcing_verdict(miner, logprobs)
        assert v.passed is False
        assert (v.reason or "").startswith("correctness_fail:")
        assert "10/50" in (v.reason or "")

    def test_equal_length_passes(self):
        tokens = ["a", "b", "c"]
        logprobs = [-0.3, -0.5, -0.4]
        v = compute_teacher_forcing_verdict(tokens, logprobs)
        assert v.passed is True

    def test_scoring_longer_than_stream_passes(self):
        """Under-score only: extra scored positions do not auto-fail."""
        v = compute_teacher_forcing_verdict(["a", "b"], [-0.3, -0.5, -0.4])
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


# --------------------------------------------------------------------------- #
# compute_improvements
# --------------------------------------------------------------------------- #


class TestComputeImprovements:
    def test_miner_faster_on_both_axes(self):
        bl_ttft = [1.0, 1.0, 1.0]
        mn_ttft = [0.5, 0.5, 0.5]
        bl_tps = [100.0, 100.0, 100.0]
        mn_tps = [150.0, 150.0, 150.0]
        score, ttft_imp, tps_imp = compute_improvements(
            bl_ttft, mn_ttft, bl_tps, mn_tps
        )
        assert ttft_imp == pytest.approx(0.5)
        assert tps_imp == pytest.approx(0.5)
        assert score == pytest.approx(0.5)

    def test_miner_slower_floors_at_zero(self):
        bl_ttft = [0.5]
        mn_ttft = [1.0]
        bl_tps = [100.0]
        mn_tps = [50.0]
        score, ttft_imp, tps_imp = compute_improvements(
            bl_ttft, mn_ttft, bl_tps, mn_tps
        )
        assert ttft_imp == 0.0
        assert tps_imp == 0.0
        assert score == 0.0

    def test_mixed_axes(self):
        bl_ttft = [1.0]
        mn_ttft = [0.8]
        bl_tps = [100.0]
        mn_tps = [80.0]
        score, ttft_imp, tps_imp = compute_improvements(
            bl_ttft, mn_ttft, bl_tps, mn_tps
        )
        assert ttft_imp == pytest.approx(0.2)
        assert tps_imp == 0.0
        assert score == pytest.approx(0.1)

    def test_median_with_outlier(self):
        bl_ttft = [1.0, 1.0, 1.0, 1.0, 100.0]
        mn_ttft = [0.5, 0.5, 0.5, 0.5, 0.5]
        bl_tps = [100.0, 100.0, 100.0, 100.0, 100.0]
        mn_tps = [120.0, 120.0, 120.0, 120.0, 120.0]
        score, ttft_imp, tps_imp = compute_improvements(
            bl_ttft, mn_ttft, bl_tps, mn_tps
        )
        assert ttft_imp == pytest.approx(0.5)
        assert tps_imp == pytest.approx(0.2)

    def test_single_prompt(self):
        score, ttft_imp, tps_imp = compute_improvements([2.0], [1.0], [50.0], [75.0])
        assert ttft_imp == pytest.approx(0.5)
        assert tps_imp == pytest.approx(0.5)
        assert score == pytest.approx(0.5)

    def test_even_number_of_prompts(self):
        score, _, _ = compute_improvements(
            [1.0, 2.0], [0.5, 1.0], [100.0, 100.0], [150.0, 150.0]
        )
        assert score == pytest.approx(0.5)

    def test_empty_lists_return_zero(self):
        assert compute_improvements([], [], [], []) == (0.0, 0.0, 0.0)

    def test_empty_baseline_ttft(self):
        assert compute_improvements([], [1.0], [100.0], [150.0]) == (0.0, 0.0, 0.0)

    def test_zero_baseline_ttft(self):
        score, ttft_imp, tps_imp = compute_improvements([0.0], [0.5], [100.0], [150.0])
        assert ttft_imp == 0.0
        assert tps_imp == pytest.approx(0.5)

    def test_zero_baseline_tps(self):
        score, ttft_imp, tps_imp = compute_improvements([1.0], [0.5], [0.0], [150.0])
        assert ttft_imp == pytest.approx(0.5)
        assert tps_imp == 0.0

    def test_identical_performance(self):
        score, ttft_imp, tps_imp = compute_improvements([1.0], [1.0], [100.0], [100.0])
        assert ttft_imp == 0.0
        assert tps_imp == 0.0
        assert score == 0.0
