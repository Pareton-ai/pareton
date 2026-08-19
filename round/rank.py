"""Pure ranking: one round's entries in, one leader decision out.

No database, no Docker, no pod, no filesystem. Everything the crown rules
need arrives as an argument, so the rules can be read and tested on their
own. The caller (TODO(PAR-83): the round runner) persists the result.

Import this as ``from round.rank import rank_round``. Binding the package
itself (``import round``) would shadow the builtin in that module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# round_entries.role and round_entries.status, mirrored from db/schema.sql.
ENTRY_ROLES: tuple[str, ...] = ("baseline", "leader", "challenger")
ENTRY_STATUSES: tuple[str, ...] = (
    "pending",
    "running",
    "scored",
    "disqualified",
    "infra_failed",
)

# An entry that reached one of these ran to a verdict. Anything else means the
# measurement never happened.
SETTLED_STATUSES: frozenset[str] = frozenset({"scored", "disqualified"})

# leader_history.event
EVENT_SEATED = "seated"
EVENT_OVERTAKEN = "overtaken"
EVENT_VACATED = "vacated"

# rounds.void_reason values this module can decide. The runner owns the rest
# (pod provision failure, pod death, round over its duration cap).
VOID_BASELINE_FAILED = "baseline_failed"
VOID_LEADER_INFRA_FAILED = "leader_infra_failed"
VOID_NO_SURVIVING_CHALLENGER = "no_surviving_challenger"
VOID_BASELINE_DRIFT = "baseline_drift"


@dataclass(frozen=True)
class Entry:
    """One row of ``round_entries``, reduced to what ranking reads."""

    role: str
    submission_id: str | None
    score: float | None
    status: str

    def __post_init__(self) -> None:
        if self.role not in ENTRY_ROLES:
            raise ValueError(f"role must be one of {ENTRY_ROLES}, got {self.role!r}")
        if self.status not in ENTRY_STATUSES:
            raise ValueError(
                f"status must be one of {ENTRY_STATUSES}, got {self.status!r}"
            )
        if self.status == "scored" and self.score is None:
            raise ValueError(f"{self.role} entry is scored but carries no score")


@dataclass(frozen=True)
class RankDecision:
    """What the round did to the crown.

    ``event`` is None when nothing changed: the leader held, or the crown was
    already vacant and stayed vacant. Those rounds write no `leader_history`
    row. A void round writes no leader state at all.
    """

    void_reason: str | None = None
    event: str | None = None
    leader_submission_id: str | None = None
    leader_score: float | None = None
    prev_submission_id: str | None = None
    prev_score: float | None = None
    overtake_threshold: float | None = None

    @property
    def void(self) -> bool:
        return self.void_reason is not None

    @property
    def leader_changed(self) -> bool:
        return self.event is not None


def _sole(entries: Sequence[Entry], role: str) -> Entry | None:
    found = [e for e in entries if e.role == role]
    if len(found) > 1:
        raise ValueError(f"a round holds at most one {role} entry, got {len(found)}")
    return found[0] if found else None


def rank_round(
    entries: Sequence[Entry],
    *,
    epsilon: float,
    drift: float | None,
    drift_ceiling: float,
    leader_score: float | None = None,
) -> RankDecision:
    """Decide the leader for one finished round.

    ``epsilon`` is the leader's moat and ``drift_ceiling`` bounds the baseline
    re-run. Both are operator knobs the caller reads from config; this module
    holds no defaults for them.

    ``drift`` is ``last_baseline_score - first_baseline_score``. None means the
    closing baseline run produced no number, which is a baseline failure.

    ``leader_score`` is the stored ``leaders.last_score``. Overtaking always
    compares in-round scores, because scores are comparable inside one round
    only; this value is reported as ``prev_score`` when the leader carries no
    score for this round.
    """
    baseline = _sole(entries, "baseline")
    if baseline is None or baseline.status != "scored":
        return RankDecision(void_reason=VOID_BASELINE_FAILED)

    leader = _sole(entries, "leader")
    if leader is not None and leader.status == "infra_failed":
        return RankDecision(void_reason=VOID_LEADER_INFRA_FAILED)

    # A disqualified challenger survived: it ran and earned its verdict, which
    # is terminal. Only entries that never produced one leave nothing to rank.
    challengers = [e for e in entries if e.role == "challenger"]
    if not any(c.status in SETTLED_STATUSES for c in challengers):
        return RankDecision(void_reason=VOID_NO_SURVIVING_CHALLENGER)

    if drift is None:
        return RankDecision(void_reason=VOID_BASELINE_FAILED)
    if abs(drift) > drift_ceiling:
        return RankDecision(void_reason=VOID_BASELINE_DRIFT)

    scored = [e for e in entries if e.role != "baseline" and e.status == "scored"]

    # Holding the crown at all needs a score strictly above baseline speed, so
    # a leader that fell back to the baseline is no longer an incumbent.
    holds = leader is not None and leader.status == "scored" and leader.score > 0
    threshold = leader.score * (1.0 + epsilon) if holds else 0.0

    contenders = [e for e in scored if e is not leader and e.score > threshold]
    prev_id = leader.submission_id if leader is not None else None
    prev_score = (
        leader.score
        if leader is not None and leader.status == "scored"
        else leader_score
    )

    if not contenders:
        if leader is not None and not holds:
            return RankDecision(
                event=EVENT_VACATED,
                prev_submission_id=prev_id,
                prev_score=prev_score,
                overtake_threshold=threshold,
            )
        # The leader held, or the crown was vacant and nobody cleared the bar.
        return RankDecision(
            leader_submission_id=prev_id if holds else None,
            leader_score=leader.score if holds else None,
            overtake_threshold=threshold,
        )

    # max() keeps the first maximal entry, so a tie goes to the earlier
    # commit_block: the cohort is built in that order.
    winner = max(contenders, key=lambda e: e.score)
    return RankDecision(
        event=EVENT_OVERTAKEN if leader is not None else EVENT_SEATED,
        leader_submission_id=winner.submission_id,
        leader_score=winner.score,
        prev_submission_id=prev_id,
        prev_score=prev_score,
        overtake_threshold=threshold,
    )
