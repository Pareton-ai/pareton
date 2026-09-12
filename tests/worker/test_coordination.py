"""Offline tests for worker release coordination (stage-2 spec 5.1, B3)."""

import fcntl
import json
import os
import threading
import time
from pathlib import Path

import pytest

from worker import coordination


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PARETON_COORDINATION", "1")
    lock = tmp_path / "activity.lock"
    state = tmp_path / "release-state.json"
    monkeypatch.setenv("PARETON_ACTIVITY_LOCK", str(lock))
    monkeypatch.setenv("PARETON_RELEASE_STATE", str(state))
    monkeypatch.setattr(coordination, "GATE_POLL_S", 0.05)
    monkeypatch.setattr(coordination, "PARK_EVENT_INTERVAL_S", 0.05)
    return {"lock": lock, "state": state, "tmp": tmp_path}


def write_state(env, **overrides):
    state = {
        "schema_version": 2,
        "op_id": "op-1",
        "phase": "idle",
        "scope": "full",
        "direction": "forward",
        "from_commit": "c1",
        "target_commit": "c1",
        "verified_commit": "c1",
        "updated_at": "2026-09-12T00:00:00Z",
    }
    state.update(overrides)
    env["state"].write_text(json.dumps(state))


def test_disabled_coordination_is_transparent(monkeypatch, tmp_path):
    monkeypatch.delenv("PARETON_COORDINATION", raising=False)
    ran = []
    with coordination.claim_guard():
        ran.append(1)
    assert ran == [1]


def test_open_gate_holds_shared_lock_through_task(env):
    write_state(env)
    saw_shared = []
    with coordination.claim_guard():
        # Another process must not be able to take the exclusive lock while
        # the guarded task runs (spec 5.1: lock spans the whole task).
        fd = os.open(str(env["lock"]), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            blocked = False
        except BlockingIOError:
            blocked = True
        finally:
            os.close(fd)
        saw_shared.append(blocked)
    assert saw_shared == [True]
    # Released afterwards.
    fd = os.open(str(env["lock"]), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_closed_gate_waits_then_proceeds(env):
    write_state(env, phase="draining")

    def opener():
        time.sleep(0.15)
        write_state(env, phase="idle")

    threading.Thread(target=opener).start()
    ran = []
    with coordination.claim_guard():
        ran.append(1)
    assert ran == [1]


def test_once_with_closed_gate_aborts(env):
    write_state(env, phase="applying")
    with pytest.raises(coordination.ClaimAborted) as info:
        with coordination.claim_guard(once=True):
            pytest.fail("must not claim")
    assert info.value.reason == "gate-closed"


def test_corrupt_state_parks_instead_of_claiming(env, monkeypatch):
    env["state"].write_text("{not json")
    park_events = []
    monkeypatch.setattr(
        coordination, "_emit", lambda event, **kw: park_events.append(event)
    )
    stop = threading.Event()

    def opener():
        time.sleep(0.2)
        write_state(env)

    threading.Thread(target=opener).start()
    ran = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        # Park keeps the process alive without claiming; once the state is
        # repaired it resumes.
        try:
            with coordination.claim_guard(should_abort=stop.is_set):
                ran.append(1)
                break
        except coordination.ClaimAborted:
            break
    assert ran == [1]
    assert "release_state_error" in park_events


def test_drain_cuts_the_closed_gate_wait(env):
    write_state(env, phase="draining")
    stop = threading.Event()
    stop.set()
    with pytest.raises(coordination.ClaimAborted) as info:
        with coordination.claim_guard(should_abort=stop.is_set):
            pytest.fail("must not claim")
    assert info.value.reason == "drain"


def test_gate_matrix_matches_spec(env):
    cases = [
        ("idle", "full", None, True),
        ("draining", "full", None, False),
        ("quiescing", "full", None, False),
        ("applying", "full", None, False),
        ("verifying", "full", True, True),
        ("verifying", "full", False, False),
        ("verified", "full", None, True),
        ("applying", "vector-only", None, True),
    ]
    for phase, scope, startup, open_ in cases:
        overrides = {"phase": phase, "scope": scope}
        if startup is not None:
            overrides["startup_complete"] = startup
        write_state(env, **overrides)
        assert coordination.gate_open() == (
            open_,
            coordination.gate_open()[1],
        ), f"{phase}/{scope}/{startup}"
