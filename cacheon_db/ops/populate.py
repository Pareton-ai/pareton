"""Populate Neon Postgres from a state-mainnet directory."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

_OPS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _OPS_DIR.parent
ROOT = _PKG_DIR.parent

logger = logging.getLogger(__name__)

HISTORY_FILES = (
    "leader-history.jsonl",
    "winner-history.jsonl",
    "king-history.jsonl",
)


def _optional_field(entry: dict, *names: str):
    for name in names:
        if name in entry and entry[name] is not None:
            return entry[name]
    return None


def _normalize_history_entry(entry: dict) -> dict:
    out: dict = {
        "ts": entry.get("ts"),
        "block": entry.get("block"),
        "new_leader_uid": _optional_field(
            entry, "new_leader_uid", "new_winner_uid", "new_king_uid"
        ),
        "new_leader_hotkey": (
            entry.get("new_leader_hotkey")
            or entry.get("new_winner_hotkey")
            or entry.get("new_king_hotkey")
        ),
        "new_leader_score": _optional_field(
            entry, "new_leader_score", "new_winner_score", "new_king_score"
        ),
        "new_leader_image": (
            entry.get("new_leader_image")
            or entry.get("new_winner_image")
            or entry.get("new_king_image")
        ),
        "new_leader_digest": (
            entry.get("new_leader_digest")
            or entry.get("new_winner_digest")
            or entry.get("new_king_digest")
        ),
        "overtake_threshold": _optional_field(
            entry, "overtake_threshold", "dethrone_threshold"
        ),
    }
    prev_uid = _optional_field(
        entry, "prev_leader_uid", "prev_winner_uid", "prev_king_uid"
    )
    if prev_uid is not None:
        out["prev_leader_uid"] = prev_uid
        out["prev_leader_hotkey"] = (
            entry.get("prev_leader_hotkey")
            or entry.get("prev_winner_hotkey")
            or entry.get("prev_king_hotkey")
        )
        out["prev_leader_score"] = _optional_field(
            entry, "prev_leader_score", "prev_winner_score", "prev_king_score"
        )
    return out


def _load_history_entries(state_dir: Path) -> list[dict]:
    entries: list[dict] = []
    seen: set[tuple] = set()
    for name in HISTORY_FILES:
        path = state_dir / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            uid = _optional_field(
                entry, "new_leader_uid", "new_winner_uid", "new_king_uid"
            )
            key = (entry.get("block"), uid)
            if key in seen:
                continue
            seen.add(key)
            entries.append(_normalize_history_entry(entry))
    entries.sort(key=lambda e: e.get("ts") or 0)
    return entries


def _apply_schema(conn, schema_path: Path) -> None:
    cur = conn.cursor()
    cur.execute(schema_path.read_text())
    logger.info("Applied schema from %s", schema_path)


def _truncate_mutable_tables(cur) -> None:
    cur.execute(
        """
        TRUNCATE TABLE
            leader_history,
            evaluations,
            eval_jobs,
            precheck_failures,
            content_fingerprints
        RESTART IDENTITY
        """
    )
    cur.execute("DELETE FROM leader_state")
    logger.info("Truncated mutable tables")


def _leader_row(role: str, rec) -> tuple:
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


def _write_all(cur, state_dir: Path) -> dict[str, int]:
    from psycopg2.extras import Json

    from validator.content_fingerprint import load_fingerprint_registry
    from validator.eval_schema import EvalJob
    from validator.state import ValidatorState

    counts = {
        "evaluations": 0,
        "leader_history": 0,
        "fingerprints": 0,
        "eval_jobs": 0,
    }

    state = ValidatorState.load(state_dir)
    from cacheon_db.mirror import _VALIDATOR_META_UPSERT, _validator_meta_params

    cur.execute(_VALIDATOR_META_UPSERT, _validator_meta_params(state))

    for role, rec in (("leader", state.winner), ("runner_up", state.runner_up_record)):
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
            _leader_row(role, rec),
        )

    cur.execute("DELETE FROM precheck_failures")
    for key, reason in (state.precheck_failures or {}).items():
        hotkey, commit_block = key.rsplit(":", 1)
        cur.execute(
            """
            INSERT INTO precheck_failures (hotkey, commit_block, reason)
            VALUES (%s, %s, %s)
            """,
            (hotkey, int(commit_block), reason),
        )

    for ev in state.evaluations.values():
        per_prompt = getattr(ev, "per_prompt", None)
        cur.execute(
            """
            INSERT INTO evaluations (
                hotkey, commit_block, uid, image, digest, score,
                speed_improvement, token_match_rate, disqualified,
                disqualify_reason, evaluated_at, evaluation_block, per_prompt
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (hotkey, commit_block) DO UPDATE SET
                uid = EXCLUDED.uid,
                image = EXCLUDED.image,
                digest = EXCLUDED.digest,
                score = EXCLUDED.score,
                speed_improvement = EXCLUDED.speed_improvement,
                token_match_rate = EXCLUDED.token_match_rate,
                disqualified = EXCLUDED.disqualified,
                disqualify_reason = EXCLUDED.disqualify_reason,
                evaluated_at = EXCLUDED.evaluated_at,
                evaluation_block = EXCLUDED.evaluation_block,
                per_prompt = EXCLUDED.per_prompt
            """,
            (
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
                Json(per_prompt) if per_prompt is not None else None,
            ),
        )
        counts["evaluations"] += 1

    for entry in _load_history_entries(state_dir):
        cur.execute(
            """
            INSERT INTO leader_history (
                ts, block, new_leader_uid, new_leader_hotkey, new_leader_score,
                new_leader_image, new_leader_digest, overtake_threshold,
                prev_leader_uid, prev_leader_hotkey, prev_leader_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                float(entry["ts"]),
                int(entry["block"]),
                int(entry["new_leader_uid"]),
                str(entry["new_leader_hotkey"]),
                float(entry["new_leader_score"]),
                str(entry["new_leader_image"]),
                str(entry["new_leader_digest"]),
                float(entry["overtake_threshold"]),
                entry.get("prev_leader_uid"),
                entry.get("prev_leader_hotkey"),
                entry.get("prev_leader_score"),
            ),
        )
        counts["leader_history"] += 1

    registry = load_fingerprint_registry(state_dir)
    for fingerprint, entry in (registry.get("entries") or {}).items():
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
                int(entry["uid"]),
                str(entry["hotkey"]),
                int(entry["commit_block"]),
                str(entry["image"]),
                str(entry["digest"]),
                float(entry.get("registered_at") or 0.0),
            ),
        )
        counts["fingerprints"] += 1

    job = EvalJob.load(state_dir)
    if job is not None:
        cur.execute(
            "DELETE FROM eval_jobs WHERE block = %s AND status = 'pending'",
            (job.block,),
        )
        cur.execute(
            """
            INSERT INTO eval_jobs (
                block, block_hash, challengers, leader, runner_up, created_at, status
            ) VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            """,
            (
                job.block,
                job.block_hash,
                Json([c.to_dict() for c in job.challengers]),
                Json(job.leader.to_dict()) if job.leader else None,
                Json(job.runner_up.to_dict()) if job.runner_up else None,
                job.created_at,
            ),
        )
        counts["eval_jobs"] = 1

    from cacheon_db.mirror import _EVAL_PROGRESS_UPSERT, _eval_progress_params

    progress_path = state_dir / "eval_progress.json"
    if progress_path.is_file():
        payload = json.loads(progress_path.read_text())
        if payload.get("status") == "idle":
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
        else:
            cur.execute(_EVAL_PROGRESS_UPSERT, _eval_progress_params(payload))
    else:
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

    return counts


def _verify(url: str) -> dict:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(url)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    out: dict = {}
    cur.execute("SELECT COUNT(*) AS n FROM evaluations")
    out["evaluations"] = cur.fetchone()["n"]
    cur.execute("SELECT MAX(evaluation_block) AS b FROM evaluations")
    out["max_eval_block"] = cur.fetchone()["b"]
    cur.execute(
        "SELECT last_scan_block, last_weights_set_block FROM validator_meta WHERE id=1"
    )
    row = cur.fetchone()
    out["meta"] = dict(row) if row else None
    cur.execute("SELECT status, phase, round_block FROM eval_progress WHERE id=1")
    row = cur.fetchone()
    out["eval_progress"] = dict(row) if row else None
    cur.execute("SELECT block, status FROM eval_jobs ORDER BY block DESC LIMIT 1")
    row = cur.fetchone()
    out["latest_eval_job"] = dict(row) if row else None
    cur.execute("SELECT COUNT(*) AS n FROM leader_history")
    out["leader_history"] = cur.fetchone()["n"]
    cur.execute("SELECT uid, score FROM leader_state WHERE role='leader'")
    row = cur.fetchone()
    out["leader"] = dict(row) if row else None
    conn.close()
    return out


def populate(
    database_url: str,
    state_dir: Path,
    *,
    apply_schema: bool = False,
    schema_path: Path | None = None,
) -> None:
    import psycopg2

    schema_path = schema_path or (_OPS_DIR / "schema.sql")
    conn = psycopg2.connect(database_url)
    try:
        cur = conn.cursor()
        if apply_schema:
            _apply_schema(conn, schema_path)
        _truncate_mutable_tables(cur)
        counts = _write_all(cur, state_dir)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    verify = _verify(database_url)
    logger.info("Populate wrote: %s", counts)
    logger.info("Verify: %s", verify)


def backfill(
    database_url: str,
    state_dir: Path,
    *,
    apply_schema: bool = False,
    schema_path: Path | None = None,
) -> None:
    """Deprecated alias for :func:`populate`."""
    populate(
        database_url,
        state_dir,
        apply_schema=apply_schema,
        schema_path=schema_path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=ROOT / "state-mainnet",
        help="Path to state-mainnet directory",
    )
    parser.add_argument(
        "--database-url",
        required=True,
        help="Postgres connection URL (Neon pooler)",
    )
    parser.add_argument(
        "--apply-schema",
        action="store_true",
        help="Apply cacheon_db/ops/schema.sql before populate",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=_OPS_DIR / "schema.sql",
    )
    args = parser.parse_args()
    state_dir = args.state_dir.resolve()
    if not (state_dir / "state.json").is_file():
        raise SystemExit(f"Missing {state_dir / 'state.json'}")
    populate(
        args.database_url,
        state_dir,
        apply_schema=args.apply_schema,
        schema_path=args.schema,
    )


if __name__ == "__main__":
    main()
