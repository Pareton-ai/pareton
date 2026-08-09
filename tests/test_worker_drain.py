"""Worker drain loop: SIGTERM must finish the in-flight job, then exit."""

import threading
import time

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
