"""Unit tests for validator.state -- no chain, no GPU, no bittensor."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from validator.state import (
    DUPLICATE_SUBMISSION_REASON,
    OVERTAKE_EPSILON,
    EvaluationRecord,
    WinnerRecord,
    RecordResult,
    SCHEMA_VERSION,
    ValidatorState,
    append_leader_history,
    current_timestamp,
    unknown_commits,
)

pytestmark = pytest.mark.unit


def _make_eval(
    uid: int = 1,
    hotkey: str = "hk_alice",
    commit_block: int = 100,
    score: float = 0.25,
    speed_improvement: float = 0.25,
    token_match_rate: float = 0.995,
    disqualified: bool = False,
    reason: str | None = None,
    image: str = "user/server:latest",
    digest: str | None = None,
) -> EvaluationRecord:
    if digest is None:
        digest = "sha256:" + format(uid, "x").zfill(64)
    return EvaluationRecord(
        uid=uid,
        hotkey=hotkey,
        commit_block=commit_block,
        image=image,
        digest=digest,
        score=score,
        speed_improvement=speed_improvement,
        token_match_rate=token_match_rate,
        disqualified=disqualified,
        disqualify_reason=reason,
        evaluated_at=1700000000.0,
        evaluation_block=commit_block + 10,
    )


def _record(
    state: ValidatorState, ev: EvaluationRecord, *, current_block: int | None = None
) -> RecordResult:
    """Shorthand -- tests that don't care about the block pass
    ``commit_block + 10`` to match the existing `_make_eval` default."""
    if current_block is None:
        current_block = ev.commit_block + 10
    return state.record_evaluation(ev, current_block=current_block)


def _record_as_winner(
    state: ValidatorState, ev: EvaluationRecord, *, current_block: int | None = None
) -> None:
    """Record an eval AND crown it as winner (record_evaluation no longer
    does overtake, so tests that need a winner must set it explicitly)."""
    if current_block is None:
        current_block = ev.commit_block + 10
    state.record_evaluation(ev, current_block=current_block)
    state.winner = WinnerRecord.from_evaluation(ev, won_at_block=current_block)


class TestEvaluationRecord:
    def test_eval_key(self):
        ev = _make_eval(hotkey="hk_foo", commit_block=42)
        assert ev.eval_key == "hk_foo:42"

    def test_round_trip(self):
        ev = _make_eval()
        restored = EvaluationRecord.from_dict(ev.to_dict())
        assert restored == ev

    def test_from_dict_ignores_unknown_keys(self):
        ev = _make_eval()
        data = ev.to_dict()
        data["extra_future_field"] = "ignored"
        restored = EvaluationRecord.from_dict(data)
        assert restored == ev

    def test_new_metric_fields(self):
        ev = _make_eval(
            speed_improvement=0.44,
            token_match_rate=0.998,
        )
        assert ev.speed_improvement == 0.44
        assert ev.token_match_rate == 0.998

    def test_per_prompt_defaults_to_none(self):
        ev = _make_eval()
        assert ev.per_prompt is None

    def test_per_prompt_round_trip(self):
        pp = [
            {
                "ttft_s": 0.5,
                "e2e_s": 12.0,
                "output_tokens": 256,
                "token_match_rate": 1.0,
            },
            {
                "ttft_s": 0.6,
                "e2e_s": 11.0,
                "output_tokens": 200,
                "token_match_rate": 0.99,
            },
        ]
        ev = EvaluationRecord(
            uid=1,
            hotkey="hk",
            commit_block=100,
            image="i:v1",
            digest="sha256:" + "a" * 64,
            score=0.5,
            speed_improvement=0.2,
            token_match_rate=0.99,
            disqualified=False,
            disqualify_reason=None,
            evaluated_at=1.0,
            evaluation_block=110,
            per_prompt=pp,
        )
        d = ev.to_dict()
        assert d["per_prompt"] == pp
        restored = EvaluationRecord.from_dict(d)
        assert restored.per_prompt == pp

    def test_per_prompt_none_omitted_from_dict(self):
        ev = _make_eval()
        d = ev.to_dict()
        assert "per_prompt" not in d

    def test_from_dict_without_per_prompt_is_backward_compatible(self):
        ev = _make_eval()
        d = ev.to_dict()
        assert "per_prompt" not in d
        restored = EvaluationRecord.from_dict(d)
        assert restored.per_prompt is None


class TestWinnerRecord:
    def test_from_evaluation(self):
        ev = _make_eval(score=0.5)
        winner = WinnerRecord.from_evaluation(ev, won_at_block=500)
        assert winner.uid == ev.uid
        assert winner.score == ev.score
        assert winner.hotkey == ev.hotkey
        assert winner.won_at_block == 500
        assert winner.speed_improvement == ev.speed_improvement
        assert winner.token_match_rate == ev.token_match_rate

    def test_round_trip(self):
        ev = _make_eval()
        winner = WinnerRecord.from_evaluation(ev, won_at_block=123)
        restored = WinnerRecord.from_dict(winner.to_dict())
        assert restored == winner

    def test_image_digest_preserved(self):
        digest = "sha256:" + "b" * 64
        ev = _make_eval(digest=digest)
        winner = WinnerRecord.from_evaluation(ev, won_at_block=123)
        assert winner.digest == digest

    def test_from_dict_legacy_crowned_at_block(self):
        """Old state files use `crowned_at_block`; ensure migration works."""
        data = {
            "uid": 1,
            "hotkey": "hk1",
            "commit_block": 100,
            "image": "img:v1",
            "digest": "sha256:" + "a" * 64,
            "score": 0.5,
            "speed_improvement": 0.2,
            "token_match_rate": 0.99,
            "evaluated_at": 1.0,
            "evaluation_block": 110,
            "crowned_at_block": 500,
        }
        winner = WinnerRecord.from_dict(data)
        assert winner.won_at_block == 500


class TestValidatorStateRecording:
    """record_evaluation stores the record and applies duplicate-of-winner DQ
    but does NOT update state.winner (ranking is done by rerank_round)."""

    def test_empty_state(self):
        state = ValidatorState()
        assert state.winner is None
        assert state.evaluations == {}
        assert state.schema_version == SCHEMA_VERSION

    def test_record_stores_eval(self):
        state = ValidatorState()
        ev = _make_eval(score=0.3)
        out = _record(state, ev)
        assert out.overtook is False
        assert state.winner is None
        assert state.has_evaluation(ev.hotkey, ev.commit_block)

    def test_record_does_not_set_winner(self):
        state = ValidatorState()
        _record(state, _make_eval(uid=1, hotkey="hk1", score=0.5))
        assert state.winner is None
        assert len(state.evaluations) == 1

    def test_threshold_reflects_current_winner(self):
        state = ValidatorState()
        _record_as_winner(state, _make_eval(uid=1, hotkey="hk1", score=0.5))
        out = _record(
            state,
            _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.4),
        )
        assert out.overtake_threshold == pytest.approx(0.5 * (1 + OVERTAKE_EPSILON))

    def test_record_precheck_failure(self):
        state = ValidatorState()
        state.record_precheck_failure("hk1", 100, "container startup timeout")
        assert state.has_precheck_failure("hk1", 100)
        assert state.is_known("hk1", 100)
        assert not state.has_evaluation("hk1", 100)

    def test_is_known_flags_both_paths(self):
        state = ValidatorState()
        state.record_precheck_failure("hk_pre", 1, "bad")
        _record(state, _make_eval(hotkey="hk_eval", commit_block=2))
        assert state.is_known("hk_pre", 1)
        assert state.is_known("hk_eval", 2)
        assert not state.is_known("hk_other", 3)

    def test_is_known_other_block_still_unknown_after_eval(self):
        state = ValidatorState()
        _record(state, _make_eval(hotkey="hk1", commit_block=20))
        assert state.is_known("hk1", 20)
        assert not state.is_known("hk1", 999)

    def test_is_known_other_block_still_unknown_after_precheck_failure(self):
        state = ValidatorState()
        state.record_precheck_failure("hk1", 20, "bad")
        assert state.is_known("hk1", 20)
        assert not state.is_known("hk1", 999)

    def test_recording_eval_clears_stale_precheck_entry(self):
        state = ValidatorState()
        state.record_precheck_failure("hk1", 100, "stale")
        _record(state, _make_eval(hotkey="hk1", commit_block=100))
        assert not state.has_precheck_failure("hk1", 100)
        assert state.has_evaluation("hk1", 100)


class TestDuplicateOfLeaderDQ:
    """Byte-identical Docker image (same digest) cannot tie or overtake the
    earlier-committed winner -- whoever committed first holds position."""

    _DIGEST_A = "sha256:" + "a" * 64
    _DIGEST_B = "sha256:" + "b" * 64

    def test_later_duplicate_is_dqd_and_doesnt_tie(self):
        state = ValidatorState()
        _record_as_winner(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        ev_copy = _make_eval(
            uid=2,
            hotkey="hk2",
            commit_block=200,
            score=0.5,
            digest=self._DIGEST_A,
        )
        out = _record(state, ev_copy)
        assert out.overtook is False
        assert out.stored.disqualified is True
        assert out.stored.disqualify_reason == DUPLICATE_SUBMISSION_REASON
        assert out.stored.score == 0.0
        persisted = state.evaluations[ev_copy.eval_key]
        assert persisted.disqualified is True
        assert state.winner.uid == 1

    def test_same_digest_same_hotkey_not_dqd(self):
        """Re-committing your own winning image at a later block is fine --
        the DQ rule targets cross-hotkey copies only."""
        state = ValidatorState()
        _record_as_winner(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=200,
                score=0.4,
                digest=self._DIGEST_A,
            ),
        )
        assert out.stored.disqualified is False

    def test_earlier_commit_not_dqd(self):
        """A submission at a commit_block before the winner's own commit
        never gets duplicate-of-winner DQ."""
        state = ValidatorState()
        _record_as_winner(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=200,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=100,
                score=0.49,
                digest=self._DIGEST_A,
            ),
        )
        assert out.stored.disqualified is False

    def test_different_digest_not_dqd(self):
        state = ValidatorState()
        _record_as_winner(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest=self._DIGEST_A,
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=200,
                score=0.4,
                digest=self._DIGEST_B,
            ),
        )
        assert out.stored.disqualified is False

    def test_empty_digest_does_not_trigger_dq(self):
        """Empty digest = unknown; never trips the DQ rule."""
        state = ValidatorState()
        _record_as_winner(
            state,
            _make_eval(
                uid=1,
                hotkey="hk1",
                commit_block=100,
                score=0.5,
                digest="",
            ),
        )
        out = _record(
            state,
            _make_eval(
                uid=2,
                hotkey="hk2",
                commit_block=200,
                score=0.4,
                digest="",
            ),
        )
        assert out.stored.disqualified is False


class TestRerankRound:
    """Tests for ValidatorState.rerank_round -- batch ranking after all evals."""

    def test_no_candidates_returns_none(self):
        winner, ru = ValidatorState.rerank_round(
            leader_record=None,
            ru_record=None,
            challenger_records=[],
            current_block=1000,
        )
        assert winner is None
        assert ru is None

    def test_single_challenger_wins_open_seat(self):
        ev = _make_eval(uid=1, hotkey="hk1", score=0.3)
        winner, ru = ValidatorState.rerank_round(
            leader_record=None,
            ru_record=None,
            challenger_records=[ev],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 1
        assert winner.score == 0.3
        assert ru is None

    def test_leader_defends_with_epsilon(self):
        leader = _make_eval(uid=1, hotkey="hk1", score=0.50)
        challenger = _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.5025)
        winner, ru = ValidatorState.rerank_round(
            leader_record=leader,
            ru_record=None,
            challenger_records=[challenger],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 1
        assert ru is not None
        assert ru.uid == 2

    def test_challenger_dethrones_leader(self):
        leader = _make_eval(uid=1, hotkey="hk1", score=0.50)
        challenger = _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.60)
        winner, ru = ValidatorState.rerank_round(
            leader_record=leader,
            ru_record=None,
            challenger_records=[challenger],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 2
        assert winner.score == 0.60
        assert ru is not None
        assert ru.uid == 1

    def test_leader_dqd_opens_seat(self):
        leader = _make_eval(
            uid=1, hotkey="hk1", score=0.0, disqualified=True, reason="crash"
        )
        challenger = _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.10)
        winner, ru = ValidatorState.rerank_round(
            leader_record=leader,
            ru_record=None,
            challenger_records=[challenger],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 2
        assert ru is None

    def test_all_dqd_returns_none(self):
        leader = _make_eval(
            uid=1, hotkey="hk1", score=0.0, disqualified=True, reason="crash"
        )
        challenger = _make_eval(
            uid=2,
            hotkey="hk2",
            commit_block=200,
            score=0.0,
            disqualified=True,
            reason="crash",
        )
        winner, ru = ValidatorState.rerank_round(
            leader_record=leader,
            ru_record=None,
            challenger_records=[challenger],
            current_block=1000,
        )
        assert winner is None
        assert ru is None

    def test_ru_can_win_when_leader_dqd(self):
        leader = _make_eval(
            uid=1, hotkey="hk1", score=0.0, disqualified=True, reason="crash"
        )
        ru = _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.40)
        challenger = _make_eval(uid=3, hotkey="hk3", commit_block=300, score=0.30)
        winner, new_ru = ValidatorState.rerank_round(
            leader_record=leader,
            ru_record=ru,
            challenger_records=[challenger],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 2
        assert new_ru is not None
        assert new_ru.uid == 3

    def test_best_challenger_wins_among_many(self):
        leader = _make_eval(uid=1, hotkey="hk1", score=0.20)
        c1 = _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.50)
        c2 = _make_eval(uid=3, hotkey="hk3", commit_block=300, score=0.40)
        winner, ru = ValidatorState.rerank_round(
            leader_record=leader,
            ru_record=None,
            challenger_records=[c1, c2],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 2
        assert ru is not None
        assert ru.uid == 3

    def test_leader_keeps_throne_when_only_slightly_behind_best(self):
        leader = _make_eval(uid=1, hotkey="hk1", score=0.50)
        challenger = _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.505)
        winner, _ = ValidatorState.rerank_round(
            leader_record=leader,
            ru_record=None,
            challenger_records=[challenger],
            current_block=1000,
        )
        assert winner.uid == 1

    def test_nan_score_excluded(self):
        leader = _make_eval(uid=1, hotkey="hk1", score=float("nan"))
        challenger = _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.10)
        winner, ru = ValidatorState.rerank_round(
            leader_record=leader,
            ru_record=None,
            challenger_records=[challenger],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 2
        assert ru is None

    def test_zero_score_excluded(self):
        winner, ru = ValidatorState.rerank_round(
            leader_record=None,
            ru_record=None,
            challenger_records=[_make_eval(uid=1, hotkey="hk1", score=0.0)],
            current_block=1000,
        )
        assert winner is None
        assert ru is None

    def test_leader_none_ru_present(self):
        ru_in = _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.40)
        challenger = _make_eval(uid=3, hotkey="hk3", commit_block=300, score=0.30)
        winner, ru = ValidatorState.rerank_round(
            leader_record=None,
            ru_record=ru_in,
            challenger_records=[challenger],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 2
        assert ru is not None
        assert ru.uid == 3

    def test_runner_up_none_when_single_hotkey(self):
        ev = _make_eval(uid=1, hotkey="hk1", score=0.5)
        winner, ru = ValidatorState.rerank_round(
            leader_record=ev,
            ru_record=None,
            challenger_records=[],
            current_block=1000,
        )
        assert winner is not None
        assert winner.uid == 1
        assert ru is None


class TestRunnerUp:
    def test_no_record_returns_none(self):
        state = ValidatorState()
        assert state.runner_up is None

    def test_persisted_runner_up_returned(self):
        state = ValidatorState()
        ev = _make_eval(uid=5, hotkey="hk5", score=0.4, commit_block=500)
        state.runner_up_record = WinnerRecord.from_evaluation(ev, won_at_block=510)
        assert state.runner_up is not None
        assert state.runner_up.uid == 5

    def test_evaluations_without_record_returns_none(self):
        """Past eval scores do not infer a runner-up; only rerank_round crowns one."""
        state = ValidatorState()
        _record_as_winner(state, _make_eval(uid=1, hotkey="hk1", score=0.5))
        _record(state, _make_eval(uid=2, hotkey="hk2", commit_block=200, score=0.3))
        assert state.runner_up is None


class TestScoreHistory:
    def test_history_ordered_by_commit_block(self):
        state = ValidatorState()
        _record(state, _make_eval(hotkey="hk_a", commit_block=30))
        _record(state, _make_eval(hotkey="hk_a", commit_block=10))
        _record(state, _make_eval(hotkey="hk_b", commit_block=20))
        history = state.score_history_for_hotkey("hk_a")
        assert [e.commit_block for e in history] == [10, 30]

    def test_history_empty_for_unknown_hotkey(self):
        state = ValidatorState()
        assert state.score_history_for_hotkey("hk_nobody") == []


class TestFromDictLegacy:
    def test_from_dict_legacy_king_key_fallback(self):
        old_payload = {
            "schema_version": 1,
            "king": {
                "uid": 7,
                "hotkey": "hk_legacy",
                "commit_block": 100,
                "image": "img:v1",
                "digest": "sha256:" + "c" * 64,
                "score": 0.42,
                "ttft_improvement": 0.1,
                "throughput_improvement": 0.2,
                "token_match_rate": 0.99,
                "evaluated_at": 123.0,
                "evaluation_block": 110,
                "crowned_at_block": 105,
            },
            "evaluations": {},
            "precheck_failures": {},
            "last_scan_block": 0,
            "last_weights_set_block": 0,
        }
        loaded = ValidatorState.from_dict(old_payload)
        assert loaded.winner is not None
        assert loaded.winner.uid == 7
        assert loaded.winner.won_at_block == 105
        assert loaded.winner.speed_improvement == pytest.approx(0.2)

    def test_from_dict_newer_schema_rejected(self):
        payload = {
            "schema_version": SCHEMA_VERSION + 99,
            "king": None,
            "evaluations": {},
            "precheck_failures": {},
            "last_scan_block": 0,
            "last_weights_set_block": 0,
        }
        with pytest.raises(ValueError, match="schema_version"):
            ValidatorState.from_dict(payload)


@patch("cacheon_db.sync_validator_state")
class TestPostgresSave:
    def test_save_calls_mirror(self, mock_sync):
        state = ValidatorState()
        state.save()
        mock_sync.assert_called_once_with(state)


class TestUnknownCommits:
    def test_filters_known(self):
        state = ValidatorState()
        _record(state, _make_eval(hotkey="hk1", commit_block=100))
        state.record_precheck_failure("hk2", 200, "bad")
        incoming = [
            ("hk1", 100),
            ("hk2", 200),
            ("hk3", 300),
            ("hk1", 150),
        ]
        result = unknown_commits(state, incoming)
        assert result == [("hk3", 300), ("hk1", 150)]

    def test_empty_input(self):
        state = ValidatorState()
        assert unknown_commits(state, []) == []


class TestCloneAndTimestamp:
    def test_clone_is_deep(self):
        state = ValidatorState()
        _record_as_winner(state, _make_eval())
        clone = state.clone()
        ev2 = _make_eval(uid=99, hotkey="hk_new", commit_block=999, score=0.9)
        _record_as_winner(clone, ev2)
        assert state.winner.uid == 1
        assert clone.winner.uid == 99

    def test_current_timestamp_is_monotonic_enough(self):
        assert current_timestamp() > 0


class TestAppendWinnerHistory:
    def _winner(self, uid=1, score=0.5) -> WinnerRecord:
        return WinnerRecord(
            uid=uid,
            hotkey=f"hk{uid}",
            commit_block=100,
            image="user/server:latest",
            digest="sha256:" + "a" * 64,
            score=score,
            speed_improvement=0.35,
            token_match_rate=0.995,
            evaluated_at=1700000000.0,
            evaluation_block=1000,
            won_at_block=1000,
        )

    @patch("cacheon_db.append_leader_history")
    def test_first_leader_no_prev(self, mock_append):
        ev = _make_eval(uid=1, score=0.5)
        append_leader_history(ev, None, current_block=1000, overtake_threshold=0.0)
        mock_append.assert_called_once()
        assert mock_append.call_args.kwargs["new_leader_uid"] == 1
        assert mock_append.call_args.kwargs["prev_leader_uid"] is None

    @patch("cacheon_db.append_leader_history")
    def test_overtake_includes_prev_leader(self, mock_append):
        ev = _make_eval(uid=2, hotkey="hk2", score=0.6)
        prev = self._winner(uid=1, score=0.4)
        append_leader_history(ev, prev, current_block=2000, overtake_threshold=0.404)
        assert mock_append.call_args.kwargs["prev_leader_uid"] == 1
        assert mock_append.call_args.kwargs["new_leader_score"] == 0.6

    @patch("cacheon_db.append_leader_history")
    def test_multiple_appends(self, mock_append):
        for i in range(3):
            ev = _make_eval(uid=i, hotkey=f"hk{i}", score=0.1 * (i + 1))
            append_leader_history(
                ev, None, current_block=1000 + i, overtake_threshold=0.0
            )
        assert mock_append.call_count == 3
