"""Unit tests for the pure round ranking function (no engines, pod, or DB)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from round.rank import (
    EVENT_OVERTAKEN,
    EVENT_SEATED,
    EVENT_VACATED,
    VOID_BASELINE_DRIFT,
    VOID_BASELINE_FAILED,
    VOID_LEADER_INFRA_FAILED,
    VOID_NO_SURVIVING_CHALLENGER,
    Entry,
    rank_round,
)

EPSILON = 0.01
CEILING = 0.05


def baseline(status: str = "scored") -> Entry:
    return Entry(
        role="baseline",
        submission_id=None,
        score=0.0 if status == "scored" else None,
        status=status,
    )


def leader(sid: str, score: float | None, status: str = "scored") -> Entry:
    return Entry(role="leader", submission_id=sid, score=score, status=status)


def challenger(sid: str, score: float | None, status: str = "scored") -> Entry:
    return Entry(role="challenger", submission_id=sid, score=score, status=status)


def rank(entries, *, drift: float | None = 0.0, leader_score: float | None = None):
    return rank_round(
        entries,
        epsilon=EPSILON,
        drift=drift,
        drift_ceiling=CEILING,
        leader_score=leader_score,
    )


# -- Crown rules -------------------------------------------------------------


def test_leader_holds_on_a_near_tie_inside_epsilon():
    # 0.2000 -> 0.2020 is the bar; 0.2015 is inside the moat.
    decision = rank([baseline(), leader("L", 0.20), challenger("C", 0.2015)])
    assert decision.event is None
    assert decision.leader_changed is False
    assert decision.leader_submission_id == "L"
    assert decision.overtake_threshold == pytest.approx(0.202)


def test_a_challenger_exactly_on_the_threshold_does_not_take_the_crown():
    on_the_bar = 0.20 * (1.0 + EPSILON)
    decision = rank([baseline(), leader("L", 0.20), challenger("C", on_the_bar)])
    assert decision.event is None
    assert decision.leader_submission_id == "L"


def test_leader_overtaken_outside_epsilon():
    decision = rank(
        [
            baseline(),
            leader("L", 0.20),
            challenger("C1", 0.2015),
            challenger("C2", 0.30),
        ]
    )
    assert decision.event == EVENT_OVERTAKEN
    assert decision.leader_changed is True
    assert decision.leader_submission_id == "C2"
    assert decision.leader_score == pytest.approx(0.30)
    assert decision.prev_submission_id == "L"
    assert decision.prev_score == pytest.approx(0.20)


def test_leader_disqualified_hands_the_crown_to_the_best_passing_challenger():
    decision = rank(
        [
            baseline(),
            leader("L", None, status="disqualified"),
            challenger("C1", 0.01),
            challenger("C2", 0.05),
        ],
        leader_score=0.20,
    )
    assert decision.event == EVENT_OVERTAKEN
    assert decision.leader_submission_id == "C2"
    # The epsilon moat dies with the leader: the bar is score > 0.
    assert decision.overtake_threshold == 0.0
    # No in-round score for the leader, so the stored one is reported.
    assert decision.prev_score == pytest.approx(0.20)


def test_leader_disqualified_with_no_passing_challenger_vacates():
    decision = rank(
        [
            baseline(),
            leader("L", None, status="disqualified"),
            challenger("C1", 0.0),
            challenger("C2", -0.10),
        ]
    )
    assert decision.event == EVENT_VACATED
    assert decision.leader_submission_id is None
    assert decision.prev_submission_id == "L"


def test_a_leader_that_falls_back_to_baseline_speed_loses_the_crown():
    # Decision 20: holding the crown at all needs a score strictly above 0.
    decision = rank([baseline(), leader("L", 0.0), challenger("C", -0.02)])
    assert decision.event == EVENT_VACATED
    assert decision.leader_submission_id is None
    assert decision.prev_score == pytest.approx(0.0)


def test_first_round_seats_the_best_challenger():
    decision = rank(
        [
            baseline(),
            challenger("C1", 0.10),
            challenger("C2", 0.42),
            challenger("C3", 0.20),
        ]
    )
    assert decision.event == EVENT_SEATED
    assert decision.leader_submission_id == "C2"
    assert decision.leader_score == pytest.approx(0.42)
    assert decision.prev_submission_id is None
    assert decision.overtake_threshold == 0.0


def test_first_round_with_nobody_above_zero_stays_vacant():
    decision = rank([baseline(), challenger("C1", 0.0), challenger("C2", -0.30)])
    assert decision.event is None
    assert decision.leader_changed is False
    assert decision.leader_submission_id is None


def test_a_tie_goes_to_the_earlier_entry():
    decision = rank([baseline(), challenger("C1", 0.25), challenger("C2", 0.25)])
    assert decision.leader_submission_id == "C1"


def test_a_disqualified_challenger_cannot_win():
    decision = rank(
        [
            baseline(),
            challenger("C1", None, status="disqualified"),
            challenger("C2", 0.05),
        ]
    )
    assert decision.leader_submission_id == "C2"


def test_all_challengers_disqualified_leaves_the_leader_seated():
    decision = rank(
        [
            baseline(),
            leader("L", 0.20),
            challenger("C1", None, status="disqualified"),
            challenger("C2", None, status="disqualified"),
        ]
    )
    assert decision.void is False
    assert decision.event is None
    assert decision.leader_submission_id == "L"


# -- Void conditions ---------------------------------------------------------


def test_void_when_the_baseline_failed():
    decision = rank([baseline("infra_failed"), challenger("C", 0.42)])
    assert decision.void_reason == VOID_BASELINE_FAILED
    assert decision.event is None
    assert decision.leader_changed is False


def test_void_when_the_round_carries_no_baseline_entry():
    decision = rank([challenger("C", 0.42)])
    assert decision.void_reason == VOID_BASELINE_FAILED


def test_void_when_the_leader_fails_on_infrastructure():
    decision = rank(
        [baseline(), leader("L", None, status="infra_failed"), challenger("C", 0.42)]
    )
    assert decision.void_reason == VOID_LEADER_INFRA_FAILED


def test_void_when_no_challenger_survives():
    decision = rank(
        [
            baseline(),
            leader("L", 0.20),
            challenger("C1", None, status="infra_failed"),
            challenger("C2", None, status="pending"),
        ]
    )
    assert decision.void_reason == VOID_NO_SURVIVING_CHALLENGER


def test_void_when_baseline_drift_is_over_the_ceiling():
    entries = [baseline(), leader("L", 0.20), challenger("C", 0.42)]
    assert rank(entries, drift=-0.06).void_reason == VOID_BASELINE_DRIFT
    assert rank(entries, drift=0.06).void_reason == VOID_BASELINE_DRIFT
    # On the ceiling is still inside it.
    assert rank(entries, drift=0.05).void is False


def test_void_when_the_closing_baseline_produced_no_drift_number():
    decision = rank([baseline(), challenger("C", 0.42)], drift=None)
    assert decision.void_reason == VOID_BASELINE_FAILED


def test_the_baseline_check_runs_before_the_drift_check():
    decision = rank([baseline("infra_failed"), challenger("C", 0.42)], drift=9.0)
    assert decision.void_reason == VOID_BASELINE_FAILED


# -- Input validation --------------------------------------------------------


def test_an_unknown_role_or_status_is_refused():
    with pytest.raises(ValueError, match="role must be one of"):
        Entry(role="winner", submission_id="A", score=0.1, status="scored")
    with pytest.raises(ValueError, match="status must be one of"):
        Entry(role="challenger", submission_id="A", score=0.1, status="benched")


def test_a_scored_entry_without_a_score_is_refused():
    with pytest.raises(ValueError, match="carries no score"):
        Entry(role="challenger", submission_id="A", score=None, status="scored")


def test_two_leaders_in_one_round_are_refused():
    with pytest.raises(ValueError, match="at most one leader"):
        rank([baseline(), leader("L1", 0.2), leader("L2", 0.3), challenger("C", 0.1)])
