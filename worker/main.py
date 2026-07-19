"""Single-process Stage 0 worker: optional chain scan + gate + bench pipeline.

Usage:
    PARETON_DATABASE_URL=... python -m worker.main --mock-build
    PARETON_ALLOW_MOCK_BENCH=1 PARETON_DATABASE_URL=... python -m worker.main --mock-bench
    PARETON_DATABASE_URL=... python -m worker.main --once
    PARETON_DATABASE_URL=... python -m worker.main --scan-chain
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import config
from campaign.store import claim_next_job
from worker.bench_job import process_bench_job
from worker.pipeline import process_submission

logger = logging.getLogger(__name__)


def _connect_subtensor():
    import bittensor as bt

    return bt.subtensor(network=config.SUBTENSOR_NETWORK)


def scan_chain_once(subtensor) -> tuple[list[str], list[str]]:
    """One chain scan: enqueue new submissions, return (submission_ids, registered_hotkeys)."""
    from chain.watcher import scan_chain

    created, hotkeys = scan_chain(subtensor, config.NETUID)
    if created:
        logger.info("chain scan enqueued %d new submission(s)", len(created))
    return created, hotkeys


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def run_once(
    *,
    mock_build: bool,
    mock_bench: bool,
    mock_tampered_candidate: bool,
    registered_hotkeys: list[str] | None,
) -> bool:
    row = claim_next_job(kind="gates")
    if row is not None:
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

    row = claim_next_job(kind="bench")
    if row is None:
        return False
    logger.info(
        "processing bench job submission %s patch=%s", row["id"], row["patch_hash"]
    )
    outcome = process_bench_job(
        row,
        mock_bench=mock_bench,
        mock_tampered_candidate=mock_tampered_candidate,
    )
    logger.info("submission %s bench -> %s", row["id"], outcome)
    return True


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
        "--mock-tampered-candidate",
        action="store_true",
        help="With --mock-bench, offset candidate logprobs (adversarial fail)",
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
        help="Poll SN10 revealed commitments each cycle and enqueue new submissions. "
        "Registered hotkeys come from the live metagraph.",
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

    subtensor = None
    registered_hotkeys = args.registered_hotkey
    if args.scan_chain:
        logger.info(
            "connecting to subtensor network=%s netuid=%d",
            config.SUBTENSOR_NETWORK,
            config.NETUID,
        )
        subtensor = _connect_subtensor()

    def _cycle() -> bool:
        nonlocal registered_hotkeys
        if subtensor is not None:
            try:
                _created, hotkeys = scan_chain_once(subtensor)
                if hotkeys:
                    registered_hotkeys = hotkeys
            except Exception:
                logger.exception("chain scan failed; will retry next cycle")
        return run_once(
            mock_build=args.mock_build,
            mock_bench=args.mock_bench,
            mock_tampered_candidate=args.mock_tampered_candidate,
            registered_hotkeys=registered_hotkeys,
        )

    if args.once:
        _cycle()
        return 0

    while True:
        did = _cycle()
        if not did:
            time.sleep(config.POLL_INTERVAL_S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
