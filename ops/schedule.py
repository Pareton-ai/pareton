"""Run one maintenance command at a time; drain the active run on SIGTERM."""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import threading
import time

from observability.events import _emit

logger = logging.getLogger(__name__)


def run_schedule(command, *, interval, initial_delay, stop, run=subprocess.run):
    if stop.wait(initial_delay):
        return
    while not stop.is_set():
        started = time.monotonic()
        try:
            result = run(command, check=False)
            _emit("maintenance_completed", command=command, exit_code=result.returncode)
        except OSError:
            logger.exception("maintenance command could not start")
        # Start-to-start cadence, with no overlapping runs. A stop request
        # leaves an active reaper/cleanup alone and prevents another run.
        if stop.wait(max(0, interval - (time.monotonic() - started))):
            return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--initial-delay", type=float, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.interval <= 0 or args.initial_delay < 0 or not command:
        parser.error("positive interval, nonnegative delay, and command required")
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    run_schedule(
        command, interval=args.interval, initial_delay=args.initial_delay, stop=stop
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
