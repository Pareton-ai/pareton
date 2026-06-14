"""Postgres read layer for the monitoring API."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from psycopg2.extras import RealDictCursor

from .connection import read_db_connection

COMPLETE_LINGER_S = 900  # keep in sync with validator.eval_progress


def _progress_expired(payload: dict[str, Any]) -> bool:
    if payload.get("status") != "complete":
        return False
    completed_at = payload.get("completed_at") or payload.get("updated_at") or 0
    return time.time() - completed_at > COMPLETE_LINGER_S


def _leader_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": row["uid"],
        "hotkey": row["hotkey"],
        "commit_block": row["commit_block"],
        "image": row["image"],
        "digest": row["digest"],
        "score": row["score"],
        "speed_improvement": row["speed_improvement"],
        "token_match_rate": row["token_match_rate"],
        "evaluated_at": row["evaluated_at"],
        "evaluation_block": row["evaluation_block"],
        "won_at_block": row["won_at_block"],
    }


def _evaluation_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "uid": row["uid"],
        "hotkey": row["hotkey"],
        "commit_block": row["commit_block"],
        "image": row["image"],
        "digest": row["digest"],
        "score": row["score"],
        "speed_improvement": row["speed_improvement"],
        "token_match_rate": row["token_match_rate"],
        "disqualified": row["disqualified"],
        "disqualify_reason": row["disqualify_reason"],
        "evaluated_at": row["evaluated_at"],
        "evaluation_block": row["evaluation_block"],
    }
    per_prompt = row.get("per_prompt")
    if per_prompt is not None:
        out["per_prompt"] = per_prompt
    return out


def get_validator_meta() -> dict[str, Any]:
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT last_scan_block, last_weights_set_block, schema_version
                FROM validator_meta
                WHERE id = 1
                """
            )
            row = cur.fetchone()
    if row is None:
        return {
            "last_scan_block": 0,
            "last_weights_set_block": 0,
            "schema_version": 1,
        }
    return dict(row)


def get_leader_state() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, uid, hotkey, commit_block, image, digest, score,
                       speed_improvement, token_match_rate, evaluated_at,
                       evaluation_block, won_at_block
                FROM leader_state
                """
            )
            rows = cur.fetchall()
    leader = None
    runner_up = None
    for row in rows:
        rec = _leader_row_to_dict(row)
        if row["role"] == "leader":
            leader = rec
        elif row["role"] == "runner_up":
            runner_up = rec
    return leader, runner_up


def get_leader_history() -> list[dict[str, Any]]:
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT ts, block, new_leader_uid, new_leader_hotkey,
                       new_leader_score, new_leader_image, new_leader_digest,
                       overtake_threshold, prev_leader_uid, prev_leader_hotkey,
                       prev_leader_score
                FROM leader_history
                ORDER BY ts ASC
                """
            )
            rows = cur.fetchall()
    history: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {
            "ts": row["ts"],
            "block": row["block"],
            "new_leader_uid": row["new_leader_uid"],
            "new_leader_hotkey": row["new_leader_hotkey"],
            "new_leader_score": row["new_leader_score"],
            "new_leader_image": row["new_leader_image"],
            "new_leader_digest": row["new_leader_digest"],
            "overtake_threshold": row["overtake_threshold"],
        }
        if row["prev_leader_uid"] is not None:
            entry["prev_leader_uid"] = row["prev_leader_uid"]
            entry["prev_leader_hotkey"] = row["prev_leader_hotkey"]
            entry["prev_leader_score"] = row["prev_leader_score"]
        history.append(entry)
    return history


def list_evaluations(*, status: str | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT uid, hotkey, commit_block, image, digest, score,
               speed_improvement, token_match_rate, disqualified,
               disqualify_reason, evaluated_at, evaluation_block, per_prompt
        FROM evaluations
    """
    params: tuple[Any, ...] = ()
    if status == "dq":
        sql += " WHERE disqualified = TRUE"
    elif status == "active":
        sql += " WHERE disqualified = FALSE"
    sql += " ORDER BY evaluated_at DESC"
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [_evaluation_row_to_dict(row) for row in rows]


def get_evaluations_by_hotkey(hotkey: str) -> list[dict[str, Any]]:
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT uid, hotkey, commit_block, image, digest, score,
                       speed_improvement, token_match_rate, disqualified,
                       disqualify_reason, evaluated_at, evaluation_block, per_prompt
                FROM evaluations
                WHERE hotkey = %s
                ORDER BY evaluated_at DESC
                """,
                (hotkey,),
            )
            rows = cur.fetchall()
    return [_evaluation_row_to_dict(row) for row in rows]


def get_evaluations_by_uid(uid: int) -> list[dict[str, Any]]:
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT uid, hotkey, commit_block, image, digest, score,
                       speed_improvement, token_match_rate, disqualified,
                       disqualify_reason, evaluated_at, evaluation_block, per_prompt
                FROM evaluations
                WHERE uid = %s
                ORDER BY evaluated_at DESC
                """,
                (uid,),
            )
            rows = cur.fetchall()
    return [_evaluation_row_to_dict(row) for row in rows]


def count_evaluations() -> tuple[int, int, int, float | None]:
    """Return (total, active, dq, last_eval_ts)."""
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE NOT disqualified) AS active,
                    COUNT(*) FILTER (WHERE disqualified) AS dq,
                    MAX(evaluated_at) AS last_eval_ts
                FROM evaluations
                """
            )
            row = cur.fetchone()
    if row is None:
        return 0, 0, 0, None
    last_eval_ts = row["last_eval_ts"]
    return (
        int(row["total"] or 0),
        int(row["active"] or 0),
        int(row["dq"] or 0),
        float(last_eval_ts) if last_eval_ts else None,
    )


def list_rounds() -> list[dict[str, Any]]:
    evals = list_evaluations()
    by_block: dict[int, list[dict]] = defaultdict(list)
    ts_by_block: dict[int, float] = {}
    for e in evals:
        block = e.get("evaluation_block", 0)
        by_block[block].append(
            {
                "uid": e.get("uid"),
                "hotkey": e.get("hotkey"),
                "image": e.get("image"),
                "commit_block": e.get("commit_block"),
                "score": e.get("score"),
                "speed_improvement": e.get("speed_improvement"),
                "token_match_rate": e.get("token_match_rate"),
                "disqualified": e.get("disqualified"),
                "disqualify_reason": e.get("disqualify_reason"),
            }
        )
        ts = e.get("evaluated_at") or 0
        if ts > ts_by_block.get(block, 0):
            ts_by_block[block] = ts

    rounds = []
    for block in sorted(by_block, reverse=True):
        challengers = by_block[block]
        latest_ts = ts_by_block.get(block)
        rounds.append(
            {
                "evaluation_block": block,
                "evaluated_at": latest_ts or None,
                "n_challengers": len(challengers),
                "challengers": challengers,
            }
        )
    return rounds


def get_pending_eval_job() -> dict[str, Any] | None:
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT block, block_hash, challengers, leader, runner_up, created_at
                FROM eval_jobs
                WHERE status = 'pending'
                ORDER BY block DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    if row is None:
        return None
    return {
        "block": row["block"],
        "block_hash": row["block_hash"],
        "challengers": row["challengers"] or [],
        "leader": row["leader"],
        "runner_up": row["runner_up"],
        "created_at": row["created_at"],
    }


def get_eval_progress() -> dict[str, Any] | None:
    with read_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT status, phase, detail, round_block, current_idx,
                       challengers, leader, runner_up, gpu, steps,
                       started_at, updated_at, completed_at
                FROM eval_progress
                WHERE id = 1
                """
            )
            row = cur.fetchone()
    if row is None:
        return None
    data = {k: row[k] for k in row.keys() if row[k] is not None}
    if data.get("status") == "idle" or _progress_expired(data):
        return None
    return data
