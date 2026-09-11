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
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import config
from campaign.store import claim_next_job, count_pending_jobs
from observability import probe as obs_probe
from observability.events import heartbeat as _heartbeat
from round.store import claim_pending_round
from worker import coordination
from worker.pipeline import process_submission
from worker.round_job import process_round

logger = logging.getLogger(__name__)

# Heartbeats must continue while a long gates/build/bench job blocks the main
# loop, otherwise the 15-minute heartbeat-absent monitor pages on healthy work.
HEARTBEAT_INTERVAL_S = 300.0
# The heartbeat thread doubles as the deployment-probe poller (stage-2 spec
# 7.2): it wakes every 5 seconds to check the probe file, while heartbeats
# themselves stay on the 300-second cadence.
PROBE_POLL_S = 5.0
# A hung DB read must not stall the probe loop: bound the queue-depth fetch
# and omit the field on timeout, same as on error (spec 7.2).
QUEUE_DEPTH_TIMEOUT_S = 8.0

# Single worker thread so a stuck query serializes instead of piling up.
_depth_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hb-db")


def worker_unit_name() -> str:
    return os.environ.get("PARETON_UNIT_NAME", "pareton-worker")


def _queue_depth() -> int | None:
    """Pending job count, or None if the bounded read fails or hangs.

    A database hiccup must not stop the beat: losing one field is cheap,
    whereas a dead heartbeat thread pages heartbeat-absent as though the
    whole worker had died.
    """
    try:
        return _depth_executor.submit(count_pending_jobs).result(
            timeout=QUEUE_DEPTH_TIMEOUT_S
        )
    except FutureTimeout:
        logger.warning("queue depth timed out; heartbeat omits it")
        return None
    except Exception:
        logger.warning("queue depth unavailable; heartbeat omits it", exc_info=True)
        return None


def _heartbeat_loop(
    stop: threading.Event, interval_s: float = HEARTBEAT_INTERVAL_S
) -> None:
    unit = worker_unit_name()
    next_beat = 0.0
    last_probe: str | None = None
    while not stop.is_set():
        now = time.monotonic()
        if now >= next_beat:
            _heartbeat(queue_depth=_queue_depth())
            next_beat = now + interval_s
        last_probe = obs_probe.poll_once(unit, last_probe)
        # Wake for whichever comes first: the next probe poll or the next
        # beat (short test intervals must not be stretched to PROBE_POLL_S).
        stop.wait(min(PROBE_POLL_S, max(0.001, next_beat - time.monotonic())))


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
    queue: str = "all",
) -> bool:
    claimed = claim_pending_round() if queue != "submissions" else None
    if claimed is not None:
        logger.info(
            "processing round %s ordinal=%s campaign=%s",
            claimed["id"],
            claimed["ordinal"],
            claimed["campaign_id"],
        )
        outcome = process_round(
            claimed,
            mock_bench=mock_bench,
            mock_correctness_fail=mock_correctness_fail,
        )
        logger.info("round %s -> %s", claimed["id"], outcome)
        return True
    if queue == "rounds":
        return False

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
        "--queue",
        choices=("all", "submissions", "rounds"),
        default="all",
        help="Production runs submissions and rounds in separate services",
    )
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

    def _guarded_cycle() -> bool:
        if drain.is_set():
            return False
        # The claim guard holds the shared activity lock for the whole
        # claim+task cycle, so a deploy can never update the environment
        # under live work (stage-2 spec 5.1). Without PARETON_COORDINATION=1
        # (local runs) it is a no-op.
        with coordination.claim_guard(should_abort=drain.is_set, once=args.once):
            if drain.is_set():
                return False
            return run_once(
                mock_build=args.mock_build,
                mock_bench=args.mock_bench,
                mock_correctness_fail=args.mock_correctness_fail,
                registered_hotkeys=registered_hotkeys,
                queue=args.queue,
            )

    def _cycle() -> bool:
        try:
            return _guarded_cycle()
        except coordination.ClaimAborted as aborted:
            if args.once and aborted.reason == "gate-closed":
                raise
            logger.info("claim aborted (%s); exiting", aborted.reason)
            return False

    def _request_drain(signum, _frame):
        logger.info("signal %d received; finishing current job before exit", signum)
        drain.set()

    signal.signal(signal.SIGTERM, _request_drain)
    signal.signal(signal.SIGINT, _request_drain)

    if args.once:
        try:
            _guarded_cycle()
        except coordination.ClaimAborted as aborted:
            if aborted.reason == "gate-closed":
                logger.error("--once refused: claim gate closed")
                return 3
            logger.info("--once skipped: %s", aborted.reason)
        return 0

    _run_loop(_cycle, drain, config.POLL_INTERVAL_S)
    logger.info("drain complete; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
