"""Chain-watch process: poll SN10 commitments and enqueue submissions.

Usage:
    PARETON_DATABASE_URL=... python -m worker.watcher
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading

import config
from chain.watcher import scan_chain
from worker.main import _configure_logging, _run_loop

logger = logging.getLogger(__name__)


def _connect_subtensor():
    import bittensor as bt

    return bt.Subtensor(network=config.SUBTENSOR_NETWORK)


def scan_cycle(subtensor, drain: threading.Event):
    """One watch cycle. Returns the live subtensor, or None to reconnect.

    Always safe to call while a worker job is running: this process never
    claims jobs. A failed scan drops the client so the next cycle reconnects
    instead of retrying a dead websocket forever.
    """
    if drain.is_set():
        return subtensor
    try:
        if subtensor is None:
            logger.info(
                "connecting to subtensor network=%s netuid=%d",
                config.SUBTENSOR_NETWORK,
                config.NETUID,
            )
            subtensor = _connect_subtensor()
        created, _hotkeys = scan_chain(
            subtensor, config.NETUID, network=config.SUBTENSOR_NETWORK
        )
        if created:
            logger.info("chain scan enqueued %d new submission(s)", len(created))
        return subtensor
    except Exception:
        logger.exception("chain scan failed; will reconnect next cycle")
        # Each bt.Subtensor holds a websocket; GC does not reliably reap it.
        try:
            if subtensor is not None:
                subtensor.close()
        except Exception:
            pass
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pareton chain watcher")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    _configure_logging(args.verbose)

    drain = threading.Event()

    def _request_drain(signum, _frame):
        logger.info("signal %d received; exiting after current scan", signum)
        drain.set()

    signal.signal(signal.SIGTERM, _request_drain)
    signal.signal(signal.SIGINT, _request_drain)

    subtensor = None

    def _cycle() -> bool:
        nonlocal subtensor
        subtensor = scan_cycle(subtensor, drain)
        # Always sleep between scans. True would busy-loop after ingest.
        return False

    _run_loop(_cycle, drain, config.POLL_INTERVAL_S)
    logger.info("drain complete; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
