"""Worker-side release coordination: claim gate + shared activity lock.

Stage-2 spec section 5.1. Every claim cycle takes the shared activity lock
first, then reads the gate from the release state; only an open gate may
claim. The shared lock is held for the whole task — claim, bench, result
writes, and cleanup — so a deploy holding the exclusive lock can never
update the shared environment under live work.

Production units set PARETON_COORDINATION=1 with explicit lock/state paths
(spec 4.1). Without it (plain ``python -m worker.main``) there is no gate:
local runs must not require a writable /run (spec B3). With it, a missing
or corrupt state file parks the worker: it keeps running, keeps
heartbeating, emits a structured error event periodically, and never
claims — exiting would just loop Restart=on-failure every 10s, which is
not an alert (spec 4.1).
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from observability.events import _emit

GATE_POLL_S = 5.0
PARK_EVENT_INTERVAL_S = 30.0

# Keep in sync with ops/release.py evaluate_gate(); tests parametrize over
# the same spec 4.5 matrix.
_PHASES = (
    "idle",
    "draining",
    "quiescing",
    "applying",
    "verifying",
    "verified",
)


class ClaimAborted(Exception):
    """The cycle ended without a claim: drain signal or closed gate (--once)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def coordination_enabled() -> bool:
    return os.environ.get("PARETON_COORDINATION") == "1"


def activity_lock_path() -> Path:
    return Path(os.environ.get("PARETON_ACTIVITY_LOCK", "/run/pareton-activity.lock"))


def state_path() -> Path:
    return Path(
        os.environ.get(
            "PARETON_RELEASE_STATE", "/var/lib/pareton-deploy/release-state.json"
        )
    )


def gate_open() -> tuple[bool, str]:
    """Evaluate the claim gate against the spec 4.5 matrix."""
    try:
        state = json.loads(state_path().read_text())
    except (OSError, ValueError):
        return False, "state-corrupt"
    if not isinstance(state, dict) or state.get("schema_version") != 2:
        return False, "state-corrupt"
    phase = state.get("phase")
    if phase not in _PHASES or state.get("scope") not in ("full", "vector-only"):
        return False, "state-corrupt"
    if state.get("scope") == "vector-only":
        return True, "vector-only"
    if phase in ("idle", "verified"):
        return True, phase
    if phase == "verifying" and state.get("startup_complete") is True:
        return True, "verifying-started"
    return False, f"phase={phase}"


@contextmanager
def claim_guard(should_abort: Callable[[], bool] | None = None, once: bool = False):
    """Hold the shared activity lock across one whole claim+task cycle.

    ``should_abort`` (the worker's drain flag) cuts the closed-gate wait
    short so SIGTERM still exits promptly. ``once`` mirrors ``--once``: a
    closed gate aborts with ``gate-closed`` instead of waiting, per spec.
    """
    if not coordination_enabled():
        yield
        return
    fd = _acquire_open_gate(should_abort, once)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _acquire_open_gate(should_abort: Callable[[], bool] | None, once: bool) -> int:
    last_event = 0.0
    while True:
        # O_CREAT: the file may not exist yet on a fresh boot where workers
        # start before any deploy tick has taken the exclusive side.
        fd = os.open(str(activity_lock_path()), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_SH)
        open_, detail = gate_open()
        if open_:
            return fd
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        if detail == "state-corrupt":
            if once:
                raise ClaimAborted("gate-closed")
            # Park: stay runnable, heartbeat, report periodically, never claim.
            _park(should_abort)
            continue
        if once:
            raise ClaimAborted("gate-closed")
        if time.monotonic() - last_event >= PARK_EVENT_INTERVAL_S:
            _emit("deployment_wait", reason=detail)
            last_event = time.monotonic()
        _wait_abortable(GATE_POLL_S, should_abort)


def _wait_abortable(seconds: float, should_abort: Callable[[], bool] | None) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if should_abort and should_abort():
            raise ClaimAborted("drain")
        time.sleep(1)


def _park(should_abort: Callable[[], bool] | None) -> None:
    _emit("release_state_error", reason="state-corrupt-or-missing")
    _wait_abortable(PARK_EVENT_INTERVAL_S, should_abort)
