"""Round, entry, and leader persistence. Every round SQL statement lives here.

Round creation is one transaction on one cursor: the cohort lock, the round
row, its entries, and the ``round_assigned`` events all commit or all roll
back. ``campaign.store.append_event`` opens its own connection, so using it
here would leave a committed ``round_assigned`` behind a round that never
existed.
"""

from __future__ import annotations

from typing import Any, Callable
from uuid import UUID

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from db.connection import db_connection
from gate.types import SubmissionState
from round.rank import (
    EVENT_OVERTAKEN,
    EVENT_SEATED,
    EVENT_VACATED,
    SETTLED_STATUSES,
    RankDecision,
)

# rounds.void_reason written by the watcher. The runner owns the rest.
VOID_HEARTBEAT_STALE = "heartbeat_stale"
VOID_POD_PROVISION_FAILED = "pod_provision_failed"
VOID_POD_FAILED = "pod_failed"
VOID_ROUND_TIMEOUT = "round_timeout"
VOID_TRACE_UNAVAILABLE = "trace_unavailable"
VOID_LEADER_IMAGE_MISSING = "leader_image_missing"


def campaigns_with_queue() -> list[dict[str, Any]]:
    """Campaigns holding queued work and no live round.

    A submission is queued when its newest event is ``bench_queued``; that
    event's timestamp is the wait clock the starvation trigger reads. A
    campaign with a live round is filtered out here so a running round does
    not cost a doomed create transaction on every cycle.
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT s.campaign_id,
                       COUNT(*) AS queued,
                       MIN(latest.created_at) AS oldest_queued_at
                FROM submissions s
                JOIN LATERAL (
                    SELECT e.state, e.created_at
                    FROM submission_events e
                    WHERE e.submission_id = s.id
                    ORDER BY e.created_at DESC, e.id DESC
                    LIMIT 1
                ) latest ON true
                WHERE latest.state = 'bench_queued'
                  AND NOT EXISTS (
                    SELECT 1 FROM rounds r
                    WHERE r.campaign_id = s.campaign_id
                      AND r.status IN ('pending', 'running')
                  )
                GROUP BY s.campaign_id
                """
            )
            return [dict(r) for r in cur.fetchall()]


def create_round(
    *,
    campaign_id: UUID | str,
    cohort_limit: int,
    dedupe: Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], list[dict]]],
    gpu_sku: str,
    baseline_image_ref: str,
    seed_block: int,
    seed_block_hash: str,
    seed_hex: str,
    sampled_trace_sha256: str,
    sampling_receipt: dict[str, Any],
    scoring_rule: dict[str, Any],
) -> dict[str, Any] | None:
    """Insert one pending round with its entries. Returns None when it loses.

    ``dedupe`` collapses cohort rows that share an image digest; it is passed
    in because the cohort is only known inside this transaction.

    Two watchers can reach this together. The lock on the campaign row
    serializes them: the cohort must be the oldest queued submissions by
    ``commit_block``, and creators splitting one queue between them would seat
    newer rows while older ones wait. The second creator blocks, then sees the
    live round in the re-check and returns None, leaving its cohort rows their
    ``bench_queued`` event for the next cycle. The ``UniqueViolation`` catch
    stays as the backstop.
    """
    try:
        with db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT 1 FROM campaigns WHERE id = %s FOR UPDATE",
                    (str(campaign_id),),
                )
                cur.execute(
                    """
                    SELECT 1 FROM rounds
                    WHERE campaign_id = %s AND status IN ('pending', 'running')
                    LIMIT 1
                    """,
                    (str(campaign_id),),
                )
                if cur.fetchone() is not None:
                    return None
                cur.execute(
                    """
                    SELECT s.id, s.engine_image_ref, s.commit_block
                    FROM submissions s
                    JOIN LATERAL (
                        SELECT e.state
                        FROM submission_events e
                        WHERE e.submission_id = s.id
                        ORDER BY e.created_at DESC, e.id DESC
                        LIMIT 1
                    ) latest ON true
                    WHERE s.campaign_id = %s
                      AND latest.state = 'bench_queued'
                    ORDER BY s.commit_block ASC, s.id ASC
                    LIMIT %s
                    FOR UPDATE OF s
                    """,
                    (str(campaign_id), int(cohort_limit)),
                )
                cohort = [dict(r) for r in cur.fetchall()]
                keepers, duplicates = dedupe(cohort)
                if not keepers:
                    return None

                cur.execute(
                    """
                    SELECT submission_id, engine_image_ref FROM leaders
                    WHERE campaign_id = %s
                    """,
                    (str(campaign_id),),
                )
                leader = cur.fetchone()
                cur.execute(
                    "SELECT COALESCE(MAX(ordinal), 0) + 1 AS ordinal FROM rounds "
                    "WHERE campaign_id = %s",
                    (str(campaign_id),),
                )
                ordinal = int(cur.fetchone()["ordinal"])

                cur.execute(
                    """
                    INSERT INTO rounds (
                      campaign_id, ordinal, gpu_sku, seed_block, seed_block_hash,
                      seed_hex, sampled_trace_sha256, sampling_receipt,
                      scoring_rule, status, incumbent_submission_id
                    ) VALUES (
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s
                    )
                    RETURNING id
                    """,
                    (
                        str(campaign_id),
                        ordinal,
                        gpu_sku,
                        int(seed_block),
                        seed_block_hash,
                        seed_hex,
                        sampled_trace_sha256,
                        Json(sampling_receipt),
                        Json(scoring_rule),
                        str(leader["submission_id"]) if leader else None,
                    ),
                )
                round_id = cur.fetchone()["id"]

                cur.execute(
                    """
                    INSERT INTO round_entries (round_id, role, engine_image_ref)
                    VALUES (%s, 'baseline', %s)
                    """,
                    (str(round_id), baseline_image_ref),
                )
                if leader is not None:
                    cur.execute(
                        """
                        INSERT INTO round_entries (
                          round_id, submission_id, role, engine_image_ref
                        ) VALUES (%s, %s, 'leader', %s)
                        """,
                        (
                            str(round_id),
                            str(leader["submission_id"]),
                            leader["engine_image_ref"],
                        ),
                    )
                for row in keepers:
                    cur.execute(
                        """
                        INSERT INTO round_entries (
                          round_id, submission_id, role, engine_image_ref
                        ) VALUES (%s, %s, 'challenger', %s)
                        """,
                        (str(round_id), str(row["id"]), row["engine_image_ref"]),
                    )
                    cur.execute(
                        """
                        INSERT INTO submission_events (submission_id, state, detail)
                        VALUES (%s, 'round_assigned', %s)
                        """,
                        (
                            str(row["id"]),
                            Json({"round_id": str(round_id), "ordinal": ordinal}),
                        ),
                    )
                for row in duplicates:
                    cur.execute(
                        """
                        INSERT INTO submission_events (submission_id, state, detail)
                        VALUES (%s, 'rejected', %s)
                        """,
                        (
                            str(row["id"]),
                            Json(
                                {
                                    "reason": "duplicate_image",
                                    "kept_submission_id": str(row["kept_id"]),
                                }
                            ),
                        ),
                    )
    except psycopg2.errors.UniqueViolation:
        return None
    return {
        "round_id": str(round_id),
        "ordinal": ordinal,
        "challenger_ids": [str(r["id"]) for r in keepers],
        "duplicate_ids": [str(r["id"]) for r in duplicates],
    }


def reap_stale_rounds(stale_s: int) -> list[dict[str, Any]]:
    """Void running rounds whose heartbeat went stale; requeue their challengers.

    Only challenger entries go back to ``bench_queued``. The leader entry
    belongs to a submission that already holds a terminal score, and the
    ``leaders`` row is the worker's to write, so neither is touched. The round
    keeps its ordinal: the chart shows honest gaps.
    """
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE rounds
                SET status = 'void',
                    void_reason = %s,
                    completed_at = now()
                WHERE status = 'running'
                  AND COALESCE(heartbeat_at, started_at)
                      < now() - make_interval(secs => %s)
                RETURNING id, campaign_id, ordinal
                """,
                (VOID_HEARTBEAT_STALE, int(stale_s)),
            )
            voided = [dict(r) for r in cur.fetchall()]
            for row in voided:
                _requeue_challengers(cur, row["id"], VOID_HEARTBEAT_STALE)
    return voided


def _requeue_challengers(cur: Any, round_id: Any, void_reason: str) -> None:
    """Put unsettled challenger entries back on the round queue.

    Terminal entries are already judged; requeueing them would resurrect a
    disqualified or scored submission. infra_failed is not terminal: it gets
    its one requeue. SETTLED_STATUSES is the do-not-requeue test.
    """
    cur.execute(
        """
        INSERT INTO submission_events (submission_id, state, detail)
        SELECT submission_id, 'bench_queued', %s
        FROM round_entries
        WHERE round_id = %s AND role = 'challenger'
          AND NOT (status = ANY(%s))
        """,
        (
            Json({"round_id": str(round_id), "void_reason": void_reason}),
            str(round_id),
            list(SETTLED_STATUSES),
        ),
    )


def infra_failed_follow_up_states(had_prior: bool) -> tuple[str, ...]:
    """Events to write for an infra_failed entry. The requeue is bench_queued.

    No prior infra_failed event means this is the one retry. A prior row means
    write infra_failed and stop; the cohort query will not pick it up again.
    """
    if had_prior:
        return (SubmissionState.INFRA_FAILED,)
    return (SubmissionState.INFRA_FAILED, SubmissionState.BENCH_QUEUED)


def claim_pending_round() -> dict[str, Any] | None:
    """Claim the oldest pending round. started_at and heartbeat_at are set here.

    The PAR-79 reaper keys on COALESCE(heartbeat_at, started_at). A running
    round with both NULL is unreapable, so they land in the same statement as
    status='running'.
    """
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE rounds
                SET status = 'running',
                    started_at = now(),
                    heartbeat_at = now(),
                    phase = NULL,
                    progress = NULL
                WHERE id = (
                    SELECT id FROM rounds
                    WHERE status = 'pending'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING id, campaign_id, ordinal, gpu_sku,
                          seed_block, seed_block_hash, seed_hex,
                          sampled_trace_sha256, sampling_receipt, scoring_rule,
                          status, incumbent_submission_id, winner_submission_id,
                          leader_changed, baseline_drift, phase, progress,
                          created_at, started_at, heartbeat_at, completed_at
                """
            )
            row = cur.fetchone()
    return dict(row) if row is not None else None


def set_round_phase(
    *,
    round_id: UUID | str,
    phase: str,
    progress: dict[str, Any] | None = None,
) -> bool:
    """Record what a running round is doing now. Returns whether it landed."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rounds
                SET phase = %s,
                    phase_started_at = CASE
                      WHEN phase IS DISTINCT FROM %s THEN now()
                      ELSE phase_started_at
                    END,
                    heartbeat_at = now(),
                    progress = %s
                WHERE id = %s AND status = 'running'
                """,
                (
                    phase,
                    phase,
                    Json(progress) if progress else None,
                    str(round_id),
                ),
            )
            return cur.rowcount > 0


def touch_round_heartbeat(*, round_id: UUID | str) -> bool:
    """Confirm the round is still alive. Returns whether it landed."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rounds
                SET heartbeat_at = now()
                WHERE id = %s AND status = 'running'
                """,
                (str(round_id),),
            )
            return cur.rowcount > 0


def complete_round(
    *,
    round_id: UUID | str,
    campaign_id: UUID | str,
    ordinal: int,
    decision: RankDecision,
    entries: list[dict[str, Any]],
    baseline_drift: float | None,
    epsilon: float,
    evidence_s3_url: str | None = None,
) -> bool:
    """Settle a running round. Returns False when a concurrent void won the race.

    One transaction: entry results, submission events, leaders upsert or
    delete, leader_history, and running -> complete. winner_submission_id is
    written even when the incumbent holds, because it is the only per-round
    record of who wore the crown.
    """
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for entry in entries:
                report = entry.get("report") or {}
                cur.execute(
                    """
                    UPDATE round_entries
                    SET status = %s,
                        score = %s,
                        disqualify_reason = %s,
                        report = %s,
                        evidence_s3_url = COALESCE(%s, evidence_s3_url),
                        completed_at = now()
                    WHERE id = %s AND round_id = %s
                    """,
                    (
                        entry["status"],
                        entry.get("score"),
                        entry.get("disqualify_reason"),
                        Json(report),
                        evidence_s3_url,
                        entry["id"],
                        str(round_id),
                    ),
                )
                sid = entry.get("submission_id")
                if sid is None:
                    continue
                status = entry["status"]
                if status == SubmissionState.INFRA_FAILED:
                    cur.execute(
                        """
                        SELECT 1 FROM submission_events
                        WHERE submission_id = %s AND state = %s
                        LIMIT 1
                        """,
                        (str(sid), SubmissionState.INFRA_FAILED),
                    )
                    had_prior = cur.fetchone() is not None
                    states = infra_failed_follow_up_states(had_prior)
                    if entry.get("role") != "challenger":
                        states = (SubmissionState.INFRA_FAILED,)
                    for state in states:
                        cur.execute(
                            """
                            INSERT INTO submission_events
                                (submission_id, state, detail)
                            VALUES (%s, %s, %s)
                            """,
                            (
                                str(sid),
                                state,
                                Json(
                                    {
                                        "round_id": str(round_id),
                                        "ordinal": int(ordinal),
                                        "reason": entry.get("disqualify_reason"),
                                    }
                                ),
                            ),
                        )
                elif status in (SubmissionState.SCORED, SubmissionState.DISQUALIFIED):
                    cur.execute(
                        """
                        INSERT INTO submission_events (submission_id, state, detail)
                        VALUES (%s, %s, %s)
                        """,
                        (
                            str(sid),
                            status,
                            Json(
                                {
                                    "round_id": str(round_id),
                                    "ordinal": int(ordinal),
                                    "score": entry.get("score"),
                                    "reason": entry.get("disqualify_reason"),
                                }
                            ),
                        ),
                    )

            event = decision.event
            if event == EVENT_VACATED:
                cur.execute(
                    "DELETE FROM leaders WHERE campaign_id = %s",
                    (str(campaign_id),),
                )
            elif decision.leader_submission_id is not None:
                winner = next(
                    (
                        e
                        for e in entries
                        if e.get("submission_id") is not None
                        and str(e["submission_id"])
                        == str(decision.leader_submission_id)
                    ),
                    None,
                )
                hotkey = (winner or {}).get("hotkey")
                image_ref = (winner or {}).get("engine_image_ref")
                if hotkey is None or image_ref is None:
                    cur.execute(
                        """
                        SELECT hotkey, engine_image_ref FROM submissions
                        WHERE id = %s
                        """,
                        (str(decision.leader_submission_id),),
                    )
                    sub = cur.fetchone()
                    if sub is not None:
                        hotkey = hotkey or sub["hotkey"]
                        image_ref = image_ref or sub["engine_image_ref"]
                if hotkey is None or image_ref is None:
                    raise RuntimeError(
                        "complete_round: winner is missing hotkey or image ref"
                    )
                cur.execute(
                    """
                    INSERT INTO leaders (
                      campaign_id, submission_id, engine_image_ref, hotkey,
                      won_at_round_id, won_at_ordinal, last_score,
                      last_scored_round_id, updated_at
                    ) VALUES (
                      %s, %s, %s, %s, %s, %s, %s, %s, now()
                    )
                    ON CONFLICT (campaign_id) DO UPDATE SET
                      submission_id = EXCLUDED.submission_id,
                      engine_image_ref = EXCLUDED.engine_image_ref,
                      hotkey = EXCLUDED.hotkey,
                      won_at_round_id = CASE
                        WHEN leaders.submission_id IS DISTINCT FROM
                             EXCLUDED.submission_id
                        THEN EXCLUDED.won_at_round_id
                        ELSE leaders.won_at_round_id
                      END,
                      won_at_ordinal = CASE
                        WHEN leaders.submission_id IS DISTINCT FROM
                             EXCLUDED.submission_id
                        THEN EXCLUDED.won_at_ordinal
                        ELSE leaders.won_at_ordinal
                      END,
                      last_score = EXCLUDED.last_score,
                      last_scored_round_id = EXCLUDED.last_scored_round_id,
                      updated_at = now()
                    """,
                    (
                        str(campaign_id),
                        str(decision.leader_submission_id),
                        image_ref,
                        hotkey,
                        str(round_id),
                        int(ordinal),
                        decision.leader_score,
                        str(round_id),
                    ),
                )

            if event in (EVENT_SEATED, EVENT_OVERTAKEN, EVENT_VACATED):
                new_sid = decision.leader_submission_id
                prev_sid = decision.prev_submission_id
                new_hotkey = None
                prev_hotkey = None
                if new_sid is not None:
                    cur.execute(
                        "SELECT hotkey FROM submissions WHERE id = %s",
                        (str(new_sid),),
                    )
                    row = cur.fetchone()
                    new_hotkey = row["hotkey"] if row is not None else None
                if prev_sid is not None:
                    cur.execute(
                        "SELECT hotkey FROM submissions WHERE id = %s",
                        (str(prev_sid),),
                    )
                    row = cur.fetchone()
                    prev_hotkey = row["hotkey"] if row is not None else None
                cur.execute(
                    """
                    INSERT INTO leader_history (
                      campaign_id, round_id, ordinal, event,
                      new_submission_id, new_hotkey, new_score,
                      prev_submission_id, prev_hotkey, prev_score,
                      overtake_threshold, epsilon
                    ) VALUES (
                      %s, %s, %s, %s,
                      %s, %s, %s,
                      %s, %s, %s,
                      %s, %s
                    )
                    """,
                    (
                        str(campaign_id),
                        str(round_id),
                        int(ordinal),
                        event,
                        str(new_sid) if new_sid is not None else None,
                        new_hotkey,
                        decision.leader_score,
                        str(prev_sid) if prev_sid is not None else None,
                        prev_hotkey,
                        decision.prev_score,
                        decision.overtake_threshold,
                        float(epsilon),
                    ),
                )

            cur.execute(
                """
                UPDATE rounds
                SET status = 'complete',
                    winner_submission_id = %s,
                    leader_changed = %s,
                    baseline_drift = %s,
                    completed_at = now()
                WHERE id = %s AND status = 'running'
                RETURNING id
                """,
                (
                    (
                        str(decision.leader_submission_id)
                        if decision.leader_submission_id is not None
                        else None
                    ),
                    bool(decision.leader_changed),
                    baseline_drift,
                    str(round_id),
                ),
            )
            landed = cur.fetchone() is not None
            if not landed:
                conn.rollback()
                return False
    return True


def void_round(round_id: UUID | str, reason: str) -> bool:
    """Abandon a running round. Returns False when it was already settled.

    Never touches leaders or leader_history. Requeues challengers that have
    not reached a settled status.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rounds
                SET status = 'void',
                    void_reason = %s,
                    completed_at = now()
                WHERE id = %s AND status = 'running'
                RETURNING id
                """,
                (reason, str(round_id)),
            )
            landed = cur.fetchone() is not None
            if not landed:
                return False
            _requeue_challengers(cur, round_id, reason)
    return True


def get_leader(campaign_id: UUID | str) -> dict[str, Any] | None:
    """The campaign's crown holder, or None when the crown is vacant.

    ``patch_hash`` comes along because it is how the public API addresses a
    submission; ``leaders`` stores the internal id.
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT l.submission_id, l.hotkey, l.engine_image_ref,
                       l.won_at_round_id, l.won_at_ordinal, l.last_score,
                       l.last_scored_round_id, l.updated_at, s.patch_hash
                FROM leaders l
                JOIN submissions s ON s.id = l.submission_id
                WHERE l.campaign_id = %s
                """,
                (str(campaign_id),),
            )
            row = cur.fetchone()
    return dict(row) if row is not None else None


def list_rounds(
    campaign_id: UUID | str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated rounds for a campaign, newest ordinal first.

    Returns ``{"total": int, "items": [row, ...]}``. A void round keeps its
    ordinal, so the list shows honest gaps rather than renumbering.
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM rounds WHERE campaign_id = %s",
                (str(campaign_id),),
            )
            total = int(cur.fetchone()["n"])
            cur.execute(
                """
                SELECT r.id, r.ordinal, r.status, r.void_reason, r.gpu_sku,
                       r.seed_block, r.seed_block_hash, r.leader_changed,
                       r.created_at, r.completed_at,
                       (SELECT COUNT(*) FROM round_entries e
                        WHERE e.round_id = r.id) AS entry_count
                FROM rounds r
                WHERE r.campaign_id = %s
                ORDER BY r.ordinal DESC
                LIMIT %s OFFSET %s
                """,
                (str(campaign_id), int(limit), int(offset)),
            )
            rows = cur.fetchall()
    return {"total": total, "items": [dict(r) for r in rows]}


def get_round(round_id: UUID | str) -> dict[str, Any] | None:
    """One round row. ``sampling_receipt`` and ``report`` stay unexposed."""
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, campaign_id, ordinal, status, void_reason, gpu_sku,
                       seed_block, seed_block_hash, seed_hex,
                       sampled_trace_sha256, scoring_rule,
                       incumbent_submission_id, winner_submission_id,
                       leader_changed, baseline_drift, phase, phase_started_at,
                       heartbeat_at, progress, created_at, started_at,
                       completed_at
                FROM rounds
                WHERE id = %s
                """,
                (str(round_id),),
            )
            row = cur.fetchone()
    return dict(row) if row is not None else None


def get_latest_weight_set() -> dict[str, Any] | None:
    """The newest stored weight vector, or None before any cycle has run.

    ``weight_sets`` is append-only, so the newest row is the one in force.
    ``weights`` is the dense vector (`weights[i]` is UID `i`). The API serves
    it as-is. None means no cycle has run yet and must never be read as an
    empty vector: empty is a valid on-chain instruction to pay nobody.

    ``set_ok`` and ``set_error`` stay unselected. They record how the chain
    call went, not what the vector is.
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT computed_at_block, version_key, burn_uid,
                       weights, breakdown
                FROM weight_sets
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    return dict(row) if row is not None else None


def list_idle_seated_leaders() -> list[dict[str, Any]]:
    """Seated leaders on campaigns with no pending or running round.

    The weights process uses this to decide who may be vacated for
    deregistration. A live round keeps the round-settle path as the sole
    writer of that campaign's crown.
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT l.campaign_id, l.hotkey, l.submission_id,
                       l.last_score, l.won_at_round_id, l.won_at_ordinal
                FROM leaders l
                WHERE NOT EXISTS (
                  SELECT 1 FROM rounds r
                  WHERE r.campaign_id = l.campaign_id
                    AND r.status IN ('pending', 'running')
                )
                """
            )
            return [dict(row) for row in cur.fetchall()]


def vacate_leader_if_idle(campaign_id: UUID | str, *, epsilon: float) -> bool:
    """Drop the crown when the campaign has no live round.

    Returns False if a pending or running round appeared, or the crown was
    already vacant. The caller must log that: a silent no-op would hide a
    skipped vacate. ``epsilon`` is required by ``leader_history``; this event
    is not a ranking, so the live overtake epsilon records the rule in force.

    Takes the same ``campaigns`` row lock as ``create_round`` first, so a
    vacate and a round create serialize. The round re-check is the correctness
    test; the campaigns lock is what makes it hold.
    """
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT 1 FROM campaigns WHERE id = %s FOR UPDATE",
                (str(campaign_id),),
            )
            if cur.fetchone() is None:
                return False
            cur.execute(
                """
                SELECT submission_id, hotkey, last_score,
                       won_at_round_id, won_at_ordinal
                FROM leaders
                WHERE campaign_id = %s
                FOR UPDATE
                """,
                (str(campaign_id),),
            )
            leader = cur.fetchone()
            if leader is None:
                return False
            cur.execute(
                """
                SELECT 1 FROM rounds
                WHERE campaign_id = %s AND status IN ('pending', 'running')
                LIMIT 1
                """,
                (str(campaign_id),),
            )
            if cur.fetchone() is not None:
                return False
            cur.execute(
                "DELETE FROM leaders WHERE campaign_id = %s",
                (str(campaign_id),),
            )
            cur.execute(
                """
                INSERT INTO leader_history (
                  campaign_id, round_id, ordinal, event,
                  new_submission_id, new_hotkey, new_score,
                  prev_submission_id, prev_hotkey, prev_score,
                  overtake_threshold, epsilon
                ) VALUES (
                  %s, %s, %s, %s,
                  NULL, NULL, NULL,
                  %s, %s, %s,
                  NULL, %s
                )
                """,
                (
                    str(campaign_id),
                    str(leader["won_at_round_id"]),
                    int(leader["won_at_ordinal"]),
                    EVENT_VACATED,
                    str(leader["submission_id"]),
                    leader["hotkey"],
                    leader["last_score"],
                    float(epsilon),
                ),
            )
    return True


def list_open_campaign_emissions() -> list[dict[str, Any]]:
    """Open campaigns plus seated leader and the decay-clock seed block."""
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.id AS campaign_id, c.status, c.emission_rule,
                       l.hotkey AS leader_hotkey, r.seed_block
                FROM campaigns c
                LEFT JOIN leaders l ON l.campaign_id = c.id
                LEFT JOIN rounds r ON r.id = l.won_at_round_id
                WHERE c.status = 'open'
                ORDER BY c.created_at
                """
            )
            return [dict(row) for row in cur.fetchall()]


def insert_weight_set(
    *,
    computed_at_block: int,
    version_key: int,
    burn_uid: int,
    weights: list[float],
    breakdown: list[dict[str, Any]],
) -> int:
    """Append one compute cycle. ``set_ok`` stays null until the chain returns."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO weight_sets (
                  computed_at_block, version_key, burn_uid, weights, breakdown
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    int(computed_at_block),
                    int(version_key),
                    int(burn_uid),
                    Json(weights),
                    Json(breakdown),
                ),
            )
            return int(cur.fetchone()[0])


def mark_weight_set_result(row_id: int, *, ok: bool, error: str | None) -> None:
    """First and only write of ``set_ok``. A later call is a no-op."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE weight_sets
                SET set_ok = %s, set_error = %s
                WHERE id = %s AND set_ok IS NULL
                """,
                (ok, error, int(row_id)),
            )


def list_round_entries(round_id: UUID | str) -> list[dict[str, Any]]:
    """Every entry of one round, in run order.

    ``evidence_s3_url`` and ``report`` are deliberately not selected: evidence
    stays behind its current gate.
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT e.id, e.submission_id, e.role, e.engine_image_ref,
                       e.status, e.score, e.disqualify_reason,
                       e.started_at, e.completed_at,
                       s.patch_hash, s.hotkey
                FROM round_entries e
                LEFT JOIN submissions s ON s.id = e.submission_id
                WHERE e.round_id = %s
                ORDER BY e.id ASC
                """,
                (str(round_id),),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def list_score_progress(campaign_id: UUID | str) -> list[dict[str, Any]]:
    """One point per round ordinal, oldest first, already shaped for a chart.

    ``leader_score`` is the score of the round's winner, so a void round or a
    round that seated nobody leaves a gap in the line rather than renumbering
    the axis. ``entries`` is the scatter: every non-baseline entry other than
    the winner. The baseline is a fixed 0.0 zero line, not a competitor.
    A NULL score stays null; 0.0 is a real score meaning baseline speed.
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT r.id AS round_id, r.ordinal, r.status,
                       r.winner_submission_id,
                       e.submission_id, e.role, e.status AS entry_status,
                       e.score, s.hotkey
                FROM rounds r
                LEFT JOIN round_entries e
                       ON e.round_id = r.id AND e.role <> 'baseline'
                LEFT JOIN submissions s ON s.id = e.submission_id
                WHERE r.campaign_id = %s
                ORDER BY r.ordinal ASC, e.id ASC
                """,
                (str(campaign_id),),
            )
            rows = cur.fetchall()
    points: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = str(r["round_id"])
        point = points.setdefault(
            key,
            {
                "round_id": key,
                "ordinal": int(r["ordinal"]),
                "status": r["status"],
                "leader_score": None,
                "entries": [],
            },
        )
        if r["submission_id"] is None:
            continue
        winner = r["winner_submission_id"]
        if winner is not None and str(winner) == str(r["submission_id"]):
            point["leader_score"] = r["score"]
            continue
        point["entries"].append(
            {
                "submission_id": str(r["submission_id"]),
                "hotkey": r["hotkey"],
                "role": r["role"],
                "status": r["entry_status"],
                "score": r["score"],
            }
        )
    return list(points.values())


def list_submission_round_entries(
    submission_ids: list[UUID | str],
) -> dict[str, dict[str, Any]]:
    """Map submission_id -> the round entry the API reports. Missing ids absent.

    This is the outcome the submission API reports: the round a submission was
    assigned to, its score there, and its verdict.

    A submission can hold entries in several rounds, so newest is not the
    answer. A void round changed no submission state (decision 27), so it never
    surfaces. Among what is left, an entry that reached a verdict beats a live
    one: a leader re-seated into a running round must keep reporting the score
    it won with, not a fresh ``pending``. A submission whose only entry is live
    still reports that entry, so the live assignment has one source of truth
    rather than being reconstructed from the ``round_assigned`` event.
    """
    if not submission_ids:
        return {}
    ids = [str(s) for s in submission_ids]
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Not round.rank.SETTLED_STATUSES: that set is what ranking counts
            # as a measurement and leaves out infra_failed. Here the question
            # is only whether the entry can still change.
            cur.execute(
                """
                SELECT DISTINCT ON (e.submission_id)
                       e.submission_id, e.round_id, r.ordinal, e.status,
                       e.score, e.disqualify_reason
                FROM round_entries e
                JOIN rounds r ON r.id = e.round_id
                WHERE e.submission_id = ANY(%s::uuid[]) AND r.status <> 'void'
                ORDER BY e.submission_id,
                         (e.status IN ('scored','disqualified','infra_failed'))
                             DESC,
                         r.ordinal DESC
                """,
                (ids,),
            )
            rows = cur.fetchall()
    return {
        str(r["submission_id"]): {
            "round_id": str(r["round_id"]),
            "ordinal": int(r["ordinal"]),
            "status": r["status"],
            "score": r["score"],
            "disqualify_reason": r["disqualify_reason"],
        }
        for r in rows
    }
