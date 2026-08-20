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

# rounds.void_reason written by the watcher. The runner owns the rest.
VOID_HEARTBEAT_STALE = "heartbeat_stale"


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
                cur.execute(
                    """
                    INSERT INTO submission_events (submission_id, state, detail)
                    SELECT submission_id, 'bench_queued', %s
                    FROM round_entries
                    WHERE round_id = %s AND role = 'challenger'
                      -- Terminal entries are already judged; requeueing them
                      -- would resurrect a disqualified or scored submission.
                      -- infra_failed is not terminal: it gets its one requeue.
                      AND status IN ('pending', 'running', 'infra_failed')
                    """,
                    (
                        Json(
                            {
                                "round_id": str(row["id"]),
                                "void_reason": VOID_HEARTBEAT_STALE,
                            }
                        ),
                        str(row["id"]),
                    ),
                )
    return voided


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
    """Map submission_id -> its newest round entry. Missing ids are absent.

    This is the outcome the submission API reports: the round a submission was
    assigned to, its score there, and its verdict.
    """
    if not submission_ids:
        return {}
    ids = [str(s) for s in submission_ids]
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (e.submission_id)
                       e.submission_id, e.round_id, r.ordinal, e.status,
                       e.score, e.disqualify_reason
                FROM round_entries e
                JOIN rounds r ON r.id = e.round_id
                WHERE e.submission_id = ANY(%s::uuid[])
                ORDER BY e.submission_id, r.ordinal DESC
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
