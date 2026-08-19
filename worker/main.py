"""Stage 0 worker: claim jobs and run the gate + bench pipeline.

Chain ingest is a separate process: ``python -m worker.watcher``.

Usage:
    PARETON_DATABASE_URL=... python -m worker.main --mock-build
    PARETON_ALLOW_MOCK_BENCH=1 PARETON_DATABASE_URL=... python -m worker.main --mock-bench
    PARETON_DATABASE_URL=... python -m worker.main --once
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

import config
from campaign.store import claim_next_job, count_pending_jobs
from observability.events import heartbeat as _heartbeat
from worker.pipeline import process_submission

logger = logging.getLogger(__name__)

# Heartbeats must continue while a long gates/build/bench job blocks the main
# loop, otherwise the 15-minute heartbeat-absent monitor pages on healthy work.
HEARTBEAT_INTERVAL_S = 300.0


def _queue_depth() -> int | None:
    """Pending job count, or None if the read fails.

    A database hiccup must not stop the beat: losing one field is cheap,
    whereas a dead heartbeat thread pages heartbeat-absent as though the
    whole worker had died.
    """
    try:
        return count_pending_jobs()
    except Exception:
        logger.warning("queue depth unavailable; heartbeat omits it", exc_info=True)
        return None


def _heartbeat_loop(
    stop: threading.Event, interval_s: float = HEARTBEAT_INTERVAL_S
) -> None:
    while not stop.is_set():
        _heartbeat(queue_depth=_queue_depth())
        stop.wait(interval_s)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def run_once(
    *,
    mock_build: bool,
    mock_bench: bool,
    mock_correctness_fail: bool,
    registered_hotkeys: list[str] | None,
) -> bool:
    row = claim_next_job()
    if row is not None:
        # Ingest already filtered to metagraph members (chain.watcher).
        # A row in submissions is the registration proof; re-reading the
        # metagraph hours later would reject a paid submit that later
        # deregistered. --registered-hotkey remains for local/tests.
        keys = registered_hotkeys if registered_hotkeys is not None else [row["hotkey"]]
        logger.info(
            "processing gates job submission %s patch=%s", row["id"], row["patch_hash"]
        )
        result = process_submission(row, registered_hotkeys=keys, mock_build=mock_build)
        logger.info(
            "submission %s -> ok=%s state=%s reason=%s",
            row["id"],
            result.ok,
            result.state,
            result.reason,
        )
        return True

    # TODO(PAR-83): rounds replace the per-submission bench job. The worker
    # claims a pending round here once round/store.py and worker/round_job.py
    # land; mock_bench and mock_correctness_fail feed that runner.
    return False


def _run_loop(cycle, drain: threading.Event, poll_interval_s: float) -> None:
    """Run work cycles until drain is set; idle sleep wakes early on drain."""
    while not drain.is_set():
        did = cycle()
        if not did:
            drain.wait(poll_interval_s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pareton Stage 0 gate + bench worker")
    p.add_argument(
        "--once", action="store_true", help="Process at most one job and exit"
    )
    p.add_argument(
        "--mock-build",
        action="store_true",
        help="Skip Docker/GHCR; write local mock build artifact",
    )
    p.add_argument(
        "--mock-bench",
        action="store_true",
        help="Run bench in-process with mock engines (requires PARETON_ALLOW_MOCK_BENCH=1)",
    )
    p.add_argument(
        "--mock-correctness-fail",
        action="store_true",
        help="With --mock-bench, make one candidate emit garbage the scorer fails",
    )
    p.add_argument(
        "--registered-hotkey",
        action="append",
        default=None,
        help="Hotkey treated as registered (repeatable). Default: accept job hotkey.",
    )
    p.add_argument(
        "--scan-chain",
        action="store_true",
        help="Deprecated no-op. Chain ingest is python -m worker.watcher.",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    _configure_logging(args.verbose)

    if args.mock_bench and not config.ALLOW_MOCK_BENCH:
        print(
            "error: --mock-bench requires PARETON_ALLOW_MOCK_BENCH=1 "
            "(mock bench is non-authoritative)",
            file=sys.stderr,
        )
        return 2

    if args.scan_chain:
        logger.warning("--scan-chain is ignored; run python -m worker.watcher")

    registered_hotkeys = args.registered_hotkey

    threading.Thread(
        target=_heartbeat_loop, args=(threading.Event(),), daemon=True
    ).start()

    # A killed worker strands its claimed job in 'running' forever (only
    # 'pending' jobs are re-claimed) and orphans any rented GPU pod, so on
    # SIGTERM/SIGINT finish the in-flight job before exiting. systemd allows
    # this up to TimeoutStopSec, then SIGKILLs. The drain flag is defined
    # before _cycle so a signal arriving mid-job stops the worker before it
    # claims a fresh one.
    drain = threading.Event()

    def _cycle() -> bool:
        if drain.is_set():
            return False
        return run_once(
            mock_build=args.mock_build,
            mock_bench=args.mock_bench,
            mock_correctness_fail=args.mock_correctness_fail,
            registered_hotkeys=registered_hotkeys,
        )

    def _request_drain(signum, _frame):
        logger.info("signal %d received; finishing current job before exit", signum)
        drain.set()

    signal.signal(signal.SIGTERM, _request_drain)
    signal.signal(signal.SIGINT, _request_drain)

    if args.once:
        _cycle()
        return 0

    _run_loop(_cycle, drain, config.POLL_INTERVAL_S)
    logger.info("drain complete; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
