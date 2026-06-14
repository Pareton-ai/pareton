"""Postgres loaders for validator processes (no API idle/expired filtering)."""

from __future__ import annotations

import logging
from typing import Any

from psycopg2.extras import RealDictCursor

from .config import enabled
from .connection import db_connection

logger = logging.getLogger(__name__)


def load_eval_progress_payload() -> dict[str, Any] | None:
    """Return the raw eval_progress row, or None when DB is disabled or idle."""
    if not enabled():
        return None
    try:
        with db_connection() as conn:
            if conn is None:
                return None
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
    except Exception:
        logger.debug("Postgres eval progress load failed", exc_info=True)
        return None
    if row is None:
        return None
    data = {k: row[k] for k in row.keys() if row[k] is not None}
    if data.get("status") == "idle":
        return None
    return data


def load_pending_eval_job_dict() -> dict[str, Any] | None:
    if not enabled():
        return None
    try:
        with db_connection() as conn:
            if conn is None:
                return None
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
    except Exception:
        logger.debug("Postgres eval job load failed", exc_info=True)
        return None
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


def _leader_role_to_winner_dict(row: dict[str, Any]) -> dict[str, Any]:
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


def load_validator_state_dict() -> dict[str, Any] | None:
    """Build a ``state.json``-shaped dict from Postgres tables."""
    if not enabled():
        return None
    try:
        with db_connection() as conn:
            if conn is None:
                return None
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT last_scan_block, last_weights_set_block, schema_version
                    FROM validator_meta
                    WHERE id = 1
                    """
                )
                meta = cur.fetchone()
                cur.execute(
                    """
                    SELECT role, uid, hotkey, commit_block, image, digest, score,
                           speed_improvement, token_match_rate, evaluated_at,
                           evaluation_block, won_at_block
                    FROM leader_state
                    """
                )
                leader_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT hotkey, commit_block, uid, image, digest, score,
                           speed_improvement, token_match_rate, disqualified,
                           disqualify_reason, evaluated_at, evaluation_block, per_prompt
                    FROM evaluations
                    """
                )
                eval_rows = cur.fetchall()
                cur.execute(
                    "SELECT hotkey, commit_block, reason FROM precheck_failures"
                )
                precheck_rows = cur.fetchall()
    except Exception:
        logger.debug("Postgres validator state load failed", exc_info=True)
        return None

    if meta is None:
        return None

    winner = None
    runner_up = None
    for row in leader_rows:
        rec = _leader_role_to_winner_dict(row)
        if row["role"] == "leader":
            winner = rec
        elif row["role"] == "runner_up":
            runner_up = rec

    evaluations: dict[str, Any] = {}
    for row in eval_rows:
        key = f"{row['hotkey']}:{row['commit_block']}"
        entry: dict[str, Any] = {
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
        if row["per_prompt"] is not None:
            entry["per_prompt"] = row["per_prompt"]
        evaluations[key] = entry

    precheck_failures = {
        f"{row['hotkey']}:{row['commit_block']}": row["reason"] for row in precheck_rows
    }

    return {
        "schema_version": int(meta["schema_version"]),
        "last_scan_block": int(meta["last_scan_block"]),
        "last_weights_set_block": int(meta["last_weights_set_block"]),
        "winner": winner,
        "runner_up": runner_up,
        "evaluations": evaluations,
        "precheck_failures": precheck_failures,
    }


def load_fingerprint_registry_dict() -> dict[str, Any] | None:
    if not enabled():
        return None
    try:
        with db_connection() as conn:
            if conn is None:
                return None
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT fingerprint, uid, hotkey, commit_block, image, digest,
                           registered_at
                    FROM content_fingerprints
                    """
                )
                rows = cur.fetchall()
    except Exception:
        logger.debug("Postgres fingerprint registry load failed", exc_info=True)
        return None

    entries: dict[str, Any] = {}
    for row in rows:
        entries[row["fingerprint"]] = {
            "uid": row["uid"],
            "hotkey": row["hotkey"],
            "commit_block": row["commit_block"],
            "image": row["image"],
            "digest": row["digest"],
            "registered_at": row["registered_at"],
        }
    return {"entries": entries}
