"""Ephemeral eval progress tracking.

Writes ``eval_progress.json`` locally at each phase transition and mirrors
the same payload to Postgres. The local file is overwritten atomically on
every update and deleted between rounds.

All public functions swallow exceptions so a progress-write failure
never interrupts the actual evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOG_TS_RE = re.compile(r"_(\d{8})_\d{6}\.log$")


def log_cutoff() -> datetime | None:
    """Retention cutoff for timestamped log filenames, or ``None`` if disabled."""
    from . import config as validator_config

    retention = validator_config.LOG_RETENTION_DAYS
    if retention <= 0:
        return None
    return datetime.now() - timedelta(days=retention)


def log_is_stale(filename: str, cutoff: datetime) -> bool:
    """True when *filename* matches the log timestamp pattern and is before *cutoff*."""
    m = _LOG_TS_RE.search(filename)
    if not m:
        return False
    try:
        file_date = datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return False
    return file_date < cutoff


def purge_old_logs(
    state_dir: str | os.PathLike,
    *,
    remote: bool = True,
    local: bool = True,
) -> int:
    """Delete stale files in ``state/logs/`` locally and on S3.

    Returns the number of files removed (local + remote).
    """
    cutoff = log_cutoff()
    if cutoff is None:
        return 0

    removed = 0

    if remote:
        try:
            from .sync import purge_remote_logs

            removed += purge_remote_logs(cutoff)
        except Exception:
            logger.warning("S3 log purge failed", exc_info=True)

    if local:
        logs_dir = Path(state_dir) / "logs"
        if logs_dir.is_dir():
            local_removed = 0
            for p in logs_dir.iterdir():
                if not log_is_stale(p.name, cutoff):
                    continue
                try:
                    p.unlink()
                    local_removed += 1
                except OSError:
                    pass
            if local_removed:
                logger.info(
                    "🧹 Purged %d old log file(s) from %s", local_removed, logs_dir
                )
            removed += local_removed

    return removed


PROGRESS_FILE = "eval_progress.json"
"""How long a finished round stays visible in ``eval_progress.json`` before expiring."""
COMPLETE_LINGER_S = 900  # 15 minutes


def _read_progress(state_dir: str | os.PathLike) -> dict[str, Any]:
    path = Path(state_dir) / PROGRESS_FILE
    try:
        with open(path) as f:
            payload = json.load(f)
        if payload:
            return payload
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    try:
        from cacheon_db.loaders import load_eval_progress_payload

        db_payload = load_eval_progress_payload()
        if db_payload:
            return db_payload
    except Exception:
        logger.debug("Postgres eval progress read failed", exc_info=True)
    return {}


def _write_progress(state_dir: str | os.PathLike, payload: dict[str, Any]) -> None:
    from .state import _atomic_write_json

    payload["updated_at"] = time.time()
    _atomic_write_json(Path(state_dir) / PROGRESS_FILE, payload)
    try:
        from cacheon_db import sync_eval_progress

        sync_eval_progress(payload)
    except Exception:
        logger.debug("Postgres eval progress mirror failed", exc_info=True)


def seed_progress_from_eval_job(state_dir: str | os.PathLike, eval_job: Any) -> None:
    """Ensure progress has challenger rows when CPU wrote them only to Postgres."""
    if _read_progress(state_dir).get("challengers"):
        return
    challengers = [
        {"uid": c.uid, "hotkey": c.hotkey, "image": c.image}
        for c in getattr(eval_job, "challengers", [])
    ]
    leader = None
    if getattr(eval_job, "leader", None) is not None:
        leader = {
            "uid": eval_job.leader.uid,
            "hotkey": eval_job.leader.hotkey,
            "image": eval_job.leader.image,
        }
    runner_up = None
    if getattr(eval_job, "runner_up", None) is not None:
        runner_up = {
            "uid": eval_job.runner_up.uid,
            "hotkey": eval_job.runner_up.hotkey,
            "image": eval_job.runner_up.image,
        }
    update_progress(
        state_dir,
        phase="challengers_found",
        round_block=eval_job.block,
        challengers=challengers,
        leader=leader,
        runner_up=runner_up,
    )


def update_progress(
    state_dir: str | os.PathLike,
    *,
    phase: str,
    round_block: int | None = None,
    detail: str | None = None,
    challengers: list[dict[str, Any]] | None = None,
    leader: dict[str, Any] | None = None,
    runner_up: dict[str, Any] | None = None,
    gpu: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Write or update the progress file.

    When *phase* is ``"challengers_found"``, a fresh file is created with
    the provided *challengers* list (all set to ``"pending"``).  Otherwise
    the existing file is read-modify-written to preserve accumulated state.
    """
    try:
        now = time.time()
        step: dict[str, Any] = {"ts": now, "phase": phase}
        step.update(extra)

        if phase == "challengers_found":
            entries = [
                {
                    "idx": i,
                    "uid": c.get("uid"),
                    "hotkey": c.get("hotkey"),
                    "image": c.get("image"),
                    "status": "pending",
                }
                for i, c in enumerate(challengers or [])
            ]

            def _incumbent_entry(data: dict[str, Any] | None) -> dict[str, Any] | None:
                if data is None:
                    return None
                return {**data, "status": data.get("status", "pending")}

            payload: dict[str, Any] = {
                "round_block": round_block,
                "status": "running",
                "phase": phase,
                "detail": detail,
                "current_idx": None,
                "challengers": entries,
                "leader": _incumbent_entry(leader),
                "runner_up": _incumbent_entry(runner_up),
                "gpu": None,
                "steps": [step],
                "started_at": now,
            }
        else:
            payload = _read_progress(state_dir)
            if not payload:
                payload = {
                    "round_block": round_block,
                    "status": "running",
                    "phase": phase,
                    "detail": detail,
                    "current_idx": None,
                    "challengers": [],
                    "gpu": None,
                    "steps": [],
                    "started_at": now,
                }
            payload["phase"] = phase
            payload["detail"] = detail
            payload["status"] = "running"
            payload.pop("completed_at", None)
            if round_block is not None:
                payload["round_block"] = round_block
            payload.setdefault("steps", []).append(step)

        if gpu is not None:
            payload["gpu"] = gpu

        _write_progress(state_dir, payload)
    except Exception:
        logger.debug("Failed to update eval progress", exc_info=True)


def update_challenger_status(
    state_dir: str | os.PathLike,
    idx: int,
    *,
    status: str,
    score: float | None = None,
    dq_reason: str | None = None,
    detail: str | None = None,
) -> None:
    """Advance a single challenger's status in the progress file."""
    try:
        payload = _read_progress(state_dir)
        if not payload:
            return

        challengers = payload.get("challengers") or []
        if idx < 0 or idx >= len(challengers):
            return

        challengers[idx]["status"] = status
        if score is not None:
            challengers[idx]["score"] = score
        if dq_reason is not None:
            challengers[idx]["dq_reason"] = dq_reason

        payload["current_idx"] = idx
        payload["phase"] = "challenger_eval"
        payload["detail"] = detail

        step: dict[str, Any] = {
            "ts": time.time(),
            "phase": "challenger_eval",
            "uid": challengers[idx].get("uid"),
            "status": status,
        }
        payload.setdefault("steps", []).append(step)

        _write_progress(state_dir, payload)
    except Exception:
        logger.debug("Failed to update challenger status", exc_info=True)


def update_incumbent_status(
    state_dir: str | os.PathLike,
    role: str,
    *,
    status: str,
    score: float | None = None,
    dq_reason: str | None = None,
    detail: str | None = None,
) -> None:
    """Advance leader or runner-up status in the progress file."""
    if role not in ("leader", "runner_up"):
        return
    try:
        payload = _read_progress(state_dir)
        if not payload:
            return

        incumbent = payload.get(role)
        if not incumbent:
            return

        incumbent["status"] = status
        if score is not None:
            incumbent["score"] = score
        if dq_reason is not None:
            incumbent["dq_reason"] = dq_reason

        payload[role] = incumbent
        if detail is not None:
            payload["detail"] = detail

        step: dict[str, Any] = {
            "ts": time.time(),
            "phase": f"{role}_eval",
            "uid": incumbent.get("uid"),
            "status": status,
        }
        if score is not None:
            step["score"] = score
        payload.setdefault("steps", []).append(step)

        _write_progress(state_dir, payload)
    except Exception:
        logger.debug("Failed to update incumbent status", exc_info=True)


def complete_progress(state_dir: str | os.PathLike) -> None:
    """Mark the current round finished but keep the file for dashboard linger."""
    try:
        payload = _read_progress(state_dir)
        if not payload:
            return
        now = time.time()
        payload["status"] = "complete"
        payload["phase"] = "eval_complete"
        payload["completed_at"] = now
        payload.setdefault("steps", []).append({"ts": now, "phase": "eval_complete"})
        _write_progress(state_dir, payload)
    except Exception:
        logger.debug("Failed to complete eval progress", exc_info=True)


def clear_progress(state_dir: str | os.PathLike) -> None:
    """Delete the local progress file and clear Postgres."""
    path = Path(state_dir) / PROGRESS_FILE
    try:
        if path.exists():
            os.unlink(path)
    except OSError:
        logger.debug("Failed to delete local eval progress file", exc_info=True)

    try:
        from cacheon_db import clear_eval_progress

        clear_eval_progress()
    except Exception:
        logger.debug("Postgres eval progress clear failed", exc_info=True)
