"""Eval progress read helpers for the monitor API (no validator imports)."""

import time

COMPLETE_LINGER_S = 900  # keep in sync with validator.eval_progress


def progress_expired(payload: dict) -> bool:
    """True when a ``complete`` progress file is past ``COMPLETE_LINGER_S``."""
    if payload.get("status") != "complete":
        return False
    completed_at = payload.get("completed_at") or payload.get("updated_at") or 0
    return time.time() - completed_at > COMPLETE_LINGER_S
