"""Cadence process: ``python -m weights``.

Usage:
    PARETON_DATABASE_URL=... python -m weights
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading

import config
from chain.rpc import fetch_metagraph
from worker.main import _configure_logging
from weights.loop import WeightsProcess, cycle_due, last_computed_block

logger = logging.getLogger(__name__)


def _connect_subtensor():
    import bittensor as bt

    return bt.Subtensor(network=config.SUBTENSOR_NETWORK)


def _load_wallet():
    import bittensor as bt

    return bt.Wallet(name=config.WALLET_NAME, hotkey=config.WALLET_HOTKEY)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pareton weight-setting cadence")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--once",
        action="store_true",
        help="Run at most one cycle and exit, even if the cadence is not due",
    )
    args = p.parse_args(argv)
    _configure_logging(args.verbose)

    drain = threading.Event()

    def _request_drain(signum, _frame):
        logger.info("signal %d received; exiting after current cycle", signum)
        drain.set()

    signal.signal(signal.SIGTERM, _request_drain)
    signal.signal(signal.SIGINT, _request_drain)

    process = WeightsProcess(
        last_block=last_computed_block(),
        cadence=config.WEIGHTS_CADENCE_BLOCKS,
    )
    wallet = None
    subtensor = None

    def _cycle() -> None:
        nonlocal subtensor, wallet
        if drain.is_set():
            return
        try:
            if subtensor is None:
                logger.info(
                    "connecting to subtensor network=%s netuid=%d",
                    config.SUBTENSOR_NETWORK,
                    config.NETUID,
                )
                subtensor = _connect_subtensor()
            head = int(subtensor.block)
        except Exception:
            logger.exception("subtensor connect or head read failed; will retry")
            subtensor = None
            return

        if not args.once and not cycle_due(process.last_block, head, process.cadence):
            return

        try:
            meta, head, _hash = fetch_metagraph(
                subtensor,
                config.NETUID,
                network=config.SUBTENSOR_NETWORK,
            )
        except Exception:
            logger.exception("metagraph fetch failed; will reconnect")
            try:
                subtensor.close()
            except Exception:
                pass
            subtensor = None
            return

        if config.WEIGHTS_ENABLED and wallet is None:
            wallet = _load_wallet()

        outcome = process.tick(
            head=head,
            subtensor=subtensor,
            wallet=wallet,
            meta=meta,
            enabled=config.WEIGHTS_ENABLED,
            force=args.once,
        )
        logger.info("weights cycle %s at block %d", outcome, head)

    if args.once:
        _cycle()
        return 0

    while not drain.is_set():
        _cycle()
        drain.wait(config.POLL_INTERVAL_S)
    logger.info("drain complete; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
