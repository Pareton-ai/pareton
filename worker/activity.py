"""Host lock shared by worker activity and pull-based deploys."""

from __future__ import annotations

import fcntl
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import config

LOCK_POLL_INTERVAL_S = 0.25


@contextmanager
def shared_activity_lock(
    drain: threading.Event,
    *,
    path: Path | None = None,
    poll_interval_s: float = LOCK_POLL_INTERVAL_S,
) -> Iterator[bool]:
    """Hold the worker side of the deploy lock, or stop waiting on drain.

    Workers use a shared lock so accidental multi-worker deployments retain
    their existing concurrency. The deploy script takes an exclusive lock and
    therefore cannot mutate the checkout while any worker is building,
    benchmarking, provisioning, or cleaning up.

    Acquisition is nonblocking with a short drain-aware poll. A worker waiting
    behind a deploy must still exit promptly when systemd sends SIGTERM as part
    of that deploy.
    """
    lock_path = path or config.WORKER_ACTIVITY_LOCK_PATH
    lock_file = lock_path.open("a+")
    acquired = False
    try:
        while not drain.is_set():
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                drain.wait(poll_interval_s)
        yield acquired
    finally:
        if acquired:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
