"""Relay Pareton's published weight vector to Bittensor."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chain.rpc import fetch_metagraph
from chain.weights import set_weights

API_URL = "https://api.pareton.ai/v1/weights"
NETWORK = "finney"
NETUID = 10
SUCCESS_BLOCKS = 360
RETRY_BLOCKS = 36
POLL_SECONDS = 12

logger = logging.getLogger("pareton-auditor")


def fetch_weights(client: Any = requests) -> dict[str, Any]:
    """Fetch and validate the fields needed for an on-chain submission."""
    logger.info("fetching latest weights from %s", API_URL)
    response = client.get(API_URL, timeout=30)
    response.raise_for_status()
    body = response.json()

    version_key = body.get("version_key")
    burn_uid = body.get("burn_uid")
    raw_weights = body.get("weights")
    if isinstance(version_key, bool) or not isinstance(version_key, int):
        raise TypeError("weights response has no integer version_key")
    if isinstance(burn_uid, bool) or not isinstance(burn_uid, int):
        raise TypeError("weights response has no integer burn_uid")
    if not isinstance(raw_weights, list) or not raw_weights:
        raise ValueError("weights response has no non-empty weights list")

    weights: list[float] = []
    for value in raw_weights:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("weights response contains a non-numeric weight")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weights response contains an invalid weight")
        weights.append(weight)
    if math.fsum(weights) <= 0:
        raise ValueError("weights response contains an all-zero vector")

    logger.info(
        "fetched %d weights computed at block %s with version_key=%d",
        len(weights),
        body.get("computed_at_block"),
        version_key,
    )

    return {
        "computed_at_block": body.get("computed_at_block"),
        "version_key": version_key,
        "burn_uid": burn_uid,
        "weights": weights,
    }


def submit_latest(
    subtensor: Any,
    wallet: Any,
    *,
    network: str = NETWORK,
    netuid: int = NETUID,
    client: Any = requests,
) -> None:
    """Fetch and submit the latest published vector exactly once."""
    body = fetch_weights(client)
    weights = body["weights"]
    logger.info("fetching metagraph for network=%s netuid=%d", network, netuid)
    meta, _block, _hash = fetch_metagraph(
        subtensor,
        netuid,
        network=network,
        attempts=1,
    )
    logger.info(
        "submitting %d weights to network=%s netuid=%d",
        len(weights),
        network,
        netuid,
    )
    set_weights(
        subtensor,
        wallet,
        meta,
        netuid=netuid,
        uids=range(len(weights)),
        weights=weights,
        version_key=body["version_key"],
        burn_uid=body["burn_uid"],
        attempts=1,
    )
    logger.info(
        "submitted weights computed at block %s",
        body["computed_at_block"],
    )


def run(
    *,
    coldkey: str,
    hotkey: str,
    network: str = NETWORK,
    netuid: int = NETUID,
    once: bool = False,
) -> bool:
    """Submit forever, using chain height as the retry clock."""
    import bittensor as bt

    logger.info("loading wallet coldkey=%s hotkey=%s", coldkey, hotkey)
    wallet = bt.Wallet(name=coldkey, hotkey=hotkey)
    logger.info("connecting to network=%s netuid=%d", network, netuid)
    subtensor = bt.Subtensor(network=network)
    next_attempt = 0

    try:
        while True:
            try:
                head = int(subtensor.block)
                if head < next_attempt:
                    time.sleep(POLL_SECONDS)
                    continue

                try:
                    submit_latest(
                        subtensor,
                        wallet,
                        network=network,
                        netuid=netuid,
                    )
                except Exception:
                    delay = RETRY_BLOCKS
                    if once:
                        logger.exception("weight setting failed")
                    else:
                        logger.exception(
                            "weight setting failed; retrying in %d blocks", delay
                        )
                    succeeded = False
                else:
                    delay = SUCCESS_BLOCKS
                    if once:
                        logger.info("weight setting succeeded")
                    else:
                        logger.info(
                            "weight setting succeeded; waiting %d blocks", delay
                        )
                    succeeded = True

                if once:
                    logger.info("one-shot run complete; exiting")
                    return succeeded

                try:
                    head = int(subtensor.block)
                except Exception:
                    logger.exception(
                        "could not refresh chain head; using block %d", head
                    )
                next_attempt = head + delay
                logger.info(
                    "sleeping %d blocks from block %d until block %d",
                    delay,
                    head,
                    next_attempt,
                )
            except Exception:
                logger.exception("could not read the chain head; retrying poll")
                if once:
                    return False
            time.sleep(POLL_SECONDS)
    finally:
        logger.info("closing chain connection")
        try:
            subtensor.close()
        except Exception:
            logger.exception("failed to close chain connection cleanly")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotkey", required=True, help="local validator hotkey name")
    parser.add_argument("--coldkey", required=True, help="local validator wallet name")
    parser.add_argument("--network", default=NETWORK, help="Bittensor network")
    parser.add_argument("--netuid", type=int, default=NETUID, help="subnet netuid")
    parser.add_argument(
        "--once",
        action="store_true",
        help="attempt one weight submission and exit",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        succeeded = run(
            coldkey=args.coldkey,
            hotkey=args.hotkey,
            network=args.network,
            netuid=args.netuid,
            once=args.once,
        )
    except KeyboardInterrupt:
        logger.info("stopped")
        return 0
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
