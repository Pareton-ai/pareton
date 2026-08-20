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
                      -- A terminal entry is already judged; requeueing it
                      -- would resurrect a disqualified submission.
                      AND status IN ('pending', 'running')
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
