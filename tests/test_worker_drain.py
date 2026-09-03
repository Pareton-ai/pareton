"""Worker drain and deploy lock behavior."""

import fcntl
import threading
import time
from contextlib import contextmanager

import pytest

from worker.activity import shared_activity_lock
from worker.main import _run_loop


def test_idle_sleep_wakes_promptly_on_drain():
    drain = threading.Event()
    calls = []

    def cycle():
        calls.append(1)
        return False

    threading.Timer(0.1, drain.set).start()
    start = time.monotonic()
    _run_loop(cycle, drain, poll_interval_s=60.0)
    assert time.monotonic() - start < 5.0
    assert calls == [1]


def test_no_cycle_when_already_drained():
    drain = threading.Event()
    drain.set()
    calls = []

    _run_loop(lambda: calls.append(1), drain, poll_interval_s=60.0)
    assert calls == []


def test_busy_cycles_run_until_drained():
    drain = threading.Event()
    calls = []

    def cycle():
        calls.append(1)
        if len(calls) == 3:
            drain.set()
        return True

    _run_loop(cycle, drain, poll_interval_s=60.0)
    assert len(calls) == 3

    _run_loop(cycle, drain, poll_interval_s=60.0)


def test_cycle_is_inside_activity_lock():
    drain = threading.Event()
    held = False

    @contextmanager
    def activity_lock(_drain):
        nonlocal held
        held = True
        try:
            yield True
        finally:
            held = False

    def cycle():
        assert held is True
        drain.set()
        return True

    _run_loop(cycle, drain, poll_interval_s=60.0, activity_lock=activity_lock)
    assert held is False


def test_shared_activity_lock_blocks_exclusive_deploy_lock(tmp_path):
    lock_path = tmp_path / "worker.lock"
    drain = threading.Event()

    with shared_activity_lock(drain, path=lock_path) as acquired:
        assert acquired is True
        with (
            lock_path.open("a+") as deploy_lock,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(deploy_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_activity_lock_wait_stops_on_drain(tmp_path):
    lock_path = tmp_path / "worker.lock"
    drain = threading.Event()

    with lock_path.open("a+") as deploy_lock:
        fcntl.flock(deploy_lock.fileno(), fcntl.LOCK_EX)
        threading.Timer(0.1, drain.set).start()
        start = time.monotonic()
        with shared_activity_lock(
            drain, path=lock_path, poll_interval_s=0.01
        ) as acquired:
            assert acquired is False
        assert time.monotonic() - start < 5.0


def test_scan_chain_flag_is_accepted_noop(monkeypatch):
    """Old systemd units still pass --scan-chain; it must not crash argparse."""
    import worker.main as wm

    monkeypatch.setattr(wm, "_heartbeat_loop", lambda *_args: None)
    monkeypatch.setattr(wm, "run_once", lambda **_: False)
    monkeypatch.setattr(wm, "shared_activity_lock", _always_acquired_lock)
    assert wm.main(["--scan-chain", "--once"]) == 0


@contextmanager
def _always_acquired_lock(_drain):
    yield True
