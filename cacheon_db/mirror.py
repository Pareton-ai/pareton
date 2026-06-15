"""High-level mirror hooks called from validator write paths."""

from __future__ import annotations

import logging
from typing import Any

from .connection import run_best_effort

logger = logging.getLogger(__name__)


def _json(value: Any) -> Any:
    from psycopg2.extras import Json

    return Json(value)


def _winner_record_to_leader_row(role: str, rec: Any) -> tuple:
    won_at = getattr(rec, "won_at_block", None)
    if won_at is None and hasattr(rec, "crowned_at_block"):
        won_at = rec.crowned_at_block
    return (
        role,
        rec.uid,
        rec.hotkey,
        rec.commit_block,
        rec.image,
        rec.digest,
        rec.score,
        rec.speed_improvement,
        rec.token_match_rate,
        rec.evaluated_at,
        rec.evaluation_block,
        won_at or 0,
    )


def _evaluation_row(ev: Any) -> tuple:
    per_prompt = getattr(ev, "per_prompt", None)
    return (
        ev.hotkey,
        ev.commit_block,
        ev.uid,
        ev.image,
        ev.digest,
        ev.score,
        ev.speed_improvement,
        ev.token_match_rate,
        ev.disqualified,
        ev.disqualify_reason,
        ev.evaluated_at,
        ev.evaluation_block,
        _json(per_prompt) if per_prompt is not None else None,
    )


def _validator_meta_params(state: Any) -> tuple:
    return (
        state.last_scan_block,
        state.last_weights_set_block,
        state.schema_version,
    )


_VALIDATOR_META_UPSERT = """
    INSERT INTO validator_meta (
        id, last_scan_block, last_weights_set_block, schema_version
    ) VALUES (1, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        last_scan_block = EXCLUDED.last_scan_block,
        last_weights_set_block = EXCLUDED.last_weights_set_block,
        schema_version = EXCLUDED.schema_version,
        updated_at = now()
"""


def sync_validator_state(state: Any) -> None:
    """Mirror validator meta, leader roles, and precheck failures."""

    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(_VALIDATOR_META_UPSERT, _validator_meta_params(state))
        for role, rec in (
            ("leader", state.winner),
            ("runner_up", state.runner_up_record),
        ):
            if rec is None:
                cur.execute("DELETE FROM leader_state WHERE role = %s", (role,))
                continue
            cur.execute(
                """
                INSERT INTO leader_state (
                    role, uid, hotkey, commit_block, image, digest, score,
                    speed_improvement, token_match_rate, evaluated_at,
                    evaluation_block, won_at_block
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (role) DO UPDATE SET
                    uid = EXCLUDED.uid,
                    hotkey = EXCLUDED.hotkey,
                    commit_block = EXCLUDED.commit_block,
                    image = EXCLUDED.image,
                    digest = EXCLUDED.digest,
                    score = EXCLUDED.score,
                    speed_improvement = EXCLUDED.speed_improvement,
                    token_match_rate = EXCLUDED.token_match_rate,
                    evaluated_at = EXCLUDED.evaluated_at,
                    evaluation_block = EXCLUDED.evaluation_block,
                    won_at_block = EXCLUDED.won_at_block,
                    updated_at = now()
                """,
                _winner_record_to_leader_row(role, rec),
            )
        cur.execute("DELETE FROM precheck_failures")
        for key, reason in (state.precheck_failures or {}).items():
            hotkey, commit_block = key.rsplit(":", 1)
            cur.execute(
                """
                INSERT INTO precheck_failures (hotkey, commit_block, reason)
                VALUES (%s, %s, %s)
                ON CONFLICT (hotkey, commit_block) DO UPDATE SET
                    reason = EXCLUDED.reason
                """,
                (hotkey, int(commit_block), reason),
            )

    run_best_effort("sync_validator_state", _write)


def sync_evaluation(ev: Any) -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO evaluations (
                hotkey, commit_block, uid, image, digest, score,
                speed_improvement, token_match_rate, disqualified,
                disqualify_reason, evaluated_at, evaluation_block, per_prompt
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (hotkey, commit_block, evaluation_block) DO UPDATE SET
                uid = EXCLUDED.uid,
                image = EXCLUDED.image,
                digest = EXCLUDED.digest,
                score = EXCLUDED.score,
                speed_improvement = EXCLUDED.speed_improvement,
                token_match_rate = EXCLUDED.token_match_rate,
                disqualified = EXCLUDED.disqualified,
                disqualify_reason = EXCLUDED.disqualify_reason,
                evaluated_at = EXCLUDED.evaluated_at,
                per_prompt = EXCLUDED.per_prompt
            """,
            _evaluation_row(ev),
        )
        cur.execute(
            "DELETE FROM precheck_failures WHERE hotkey = %s AND commit_block = %s",
            (ev.hotkey, ev.commit_block),
        )

    run_best_effort("sync_evaluation", _write)


def sync_precheck_failure(hotkey: str, commit_block: int, reason: str) -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO precheck_failures (hotkey, commit_block, reason)
            VALUES (%s, %s, %s)
            ON CONFLICT (hotkey, commit_block) DO UPDATE SET
                reason = EXCLUDED.reason
            """,
            (hotkey, commit_block, reason),
        )

    run_best_effort("sync_precheck_failure", _write)


def sync_eval_job(job: Any) -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute("DELETE FROM eval_jobs WHERE status = 'pending'")
        cur.execute(
            """
            INSERT INTO eval_jobs (
                block, block_hash, challengers, leader, runner_up, created_at, status
            ) VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            """,
            (
                job.block,
                job.block_hash,
                _json([c.to_dict() for c in job.challengers]),
                _json(job.leader.to_dict()) if job.leader else None,
                _json(job.runner_up.to_dict()) if job.runner_up else None,
                job.created_at,
            ),
        )

    run_best_effort("sync_eval_job", _write)


def complete_eval_job(block: int) -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE eval_jobs
            SET status = 'complete'
            WHERE block = %s AND status = 'pending'
            """,
            (block,),
        )

    run_best_effort("complete_eval_job", _write)


def delete_pending_eval_job(block: int) -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM eval_jobs WHERE block = %s AND status = 'pending'",
            (block,),
        )

    run_best_effort("delete_pending_eval_job", _write)


def _eval_progress_params(payload: dict[str, Any]) -> tuple:
    return (
        payload.get("status", "idle"),
        payload.get("phase"),
        payload.get("detail"),
        payload.get("round_block"),
        payload.get("current_idx"),
        _json(payload.get("challengers"))
        if payload.get("challengers") is not None
        else None,
        _json(payload.get("leader")) if payload.get("leader") is not None else None,
        _json(payload.get("runner_up"))
        if payload.get("runner_up") is not None
        else None,
        _json(payload.get("gpu")) if payload.get("gpu") is not None else None,
        _json(payload.get("steps")) if payload.get("steps") is not None else None,
        payload.get("started_at"),
        payload.get("updated_at"),
        payload.get("completed_at"),
    )


_EVAL_PROGRESS_UPSERT = """
    INSERT INTO eval_progress (
        id, status, phase, detail, round_block, current_idx,
        challengers, leader, runner_up, gpu, steps,
        started_at, updated_at, completed_at
    ) VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        status = EXCLUDED.status,
        phase = EXCLUDED.phase,
        detail = EXCLUDED.detail,
        round_block = EXCLUDED.round_block,
        current_idx = EXCLUDED.current_idx,
        challengers = EXCLUDED.challengers,
        leader = EXCLUDED.leader,
        runner_up = EXCLUDED.runner_up,
        gpu = EXCLUDED.gpu,
        steps = EXCLUDED.steps,
        started_at = EXCLUDED.started_at,
        updated_at = EXCLUDED.updated_at,
        completed_at = EXCLUDED.completed_at
"""


def sync_eval_progress(payload: dict[str, Any]) -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(_EVAL_PROGRESS_UPSERT, _eval_progress_params(payload))

    run_best_effort("sync_eval_progress", _write)


def clear_eval_progress() -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO eval_progress (id, status) VALUES (1, 'idle')
            ON CONFLICT (id) DO UPDATE SET
                status = 'idle',
                phase = NULL,
                detail = NULL,
                round_block = NULL,
                current_idx = NULL,
                challengers = NULL,
                leader = NULL,
                runner_up = NULL,
                gpu = NULL,
                steps = NULL,
                started_at = NULL,
                updated_at = NULL,
                completed_at = NULL
            """
        )

    run_best_effort("clear_eval_progress", _write)


def sync_fingerprint(
    fingerprint: str,
    *,
    uid: int,
    hotkey: str,
    commit_block: int,
    image: str,
    digest: str,
    registered_at: float,
) -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO content_fingerprints (
                fingerprint, uid, hotkey, commit_block, image, digest, registered_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fingerprint) DO UPDATE SET
                uid = EXCLUDED.uid,
                hotkey = EXCLUDED.hotkey,
                commit_block = EXCLUDED.commit_block,
                image = EXCLUDED.image,
                digest = EXCLUDED.digest,
                registered_at = EXCLUDED.registered_at
            """,
            (
                fingerprint,
                uid,
                hotkey,
                commit_block,
                image,
                digest,
                registered_at,
            ),
        )

    run_best_effort("sync_fingerprint", _write)


def append_leader_history(
    *,
    ts: float,
    block: int,
    new_leader_uid: int,
    new_leader_hotkey: str,
    new_leader_score: float,
    new_leader_image: str,
    new_leader_digest: str,
    overtake_threshold: float,
    prev_leader_uid: int | None = None,
    prev_leader_hotkey: str | None = None,
    prev_leader_score: float | None = None,
) -> None:
    def _write(conn) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO leader_history (
                ts, block, new_leader_uid, new_leader_hotkey, new_leader_score,
                new_leader_image, new_leader_digest, overtake_threshold,
                prev_leader_uid, prev_leader_hotkey, prev_leader_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                ts,
                block,
                new_leader_uid,
                new_leader_hotkey,
                new_leader_score,
                new_leader_image,
                new_leader_digest,
                overtake_threshold,
                prev_leader_uid,
                prev_leader_hotkey,
                prev_leader_score,
            ),
        )

    run_best_effort("append_leader_history", _write)
