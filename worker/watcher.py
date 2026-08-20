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
from observability import events as obs
from round.create import create_due_rounds
from round.store import VOID_HEARTBEAT_STALE, reap_stale_rounds
from worker.main import _configure_logging, _run_loop

logger = logging.getLogger(__name__)


def _connect_subtensor():
    import bittensor as bt

    return bt.Subtensor(network=config.SUBTENSOR_NETWORK)


def scan_cycle(subtensor, drain: threading.Event):
    """One watch cycle. Returns the live subtensor, or None to reconnect.

    Always safe to call while a worker job is running: this process never
    claims jobs. Successful scans emit ``chain_scanned`` here, so a long
    build on ``pareton-worker`` does not look like a stalled scanner. A
    failed scan drops the client so the next cycle reconnects instead of
    retrying a dead websocket forever.

    Chain work and round work fail apart. Reaping a dead round needs no chain
    at all, so a broken websocket must not leave a stale round holding the
    campaign's live-round slot; and a round that cannot be created must not
    cost the process its subtensor.
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
    except Exception:
        logger.exception("subtensor connect failed; will retry next cycle")
        subtensor = None

    if subtensor is not None:
        try:
            created, _hotkeys = scan_chain(
                subtensor, config.NETUID, network=config.SUBTENSOR_NETWORK
            )
            if created:
                logger.info("chain scan enqueued %d new submission(s)", len(created))
        except Exception:
            logger.exception("chain scan failed; will reconnect next cycle")
            # Each bt.Subtensor holds a websocket; GC does not reliably reap it.
            try:
                subtensor.close()
            except Exception:
                pass
            subtensor = None

    # Reap first: voiding a dead round frees the live-round slot, so a new
    # round can start in this same cycle.
    try:
        for row in reap_stale_rounds(config.ROUND_STALE_S):
            logger.warning(
                "voided round %s (campaign %s): heartbeat stale",
                row["ordinal"],
                row["campaign_id"],
            )
            obs.round_voided(
                round_id=str(row["id"]),
                campaign_id=str(row["campaign_id"]),
                void_reason=VOID_HEARTBEAT_STALE,
            )
    except Exception:
        logger.exception("round reaping failed; will retry next cycle")

    if subtensor is not None:
        try:
            create_due_rounds(subtensor)
        except Exception:
            logger.exception("round creation failed; will retry next cycle")
    return subtensor


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
