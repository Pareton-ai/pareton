"""Relay Pareton's published weight vector to Bittensor.

Standalone by design. It needs only `requests` and `bittensor`, and imports
nothing from the Pareton repository, so a validator can copy this one file
anywhere and run it. Every decision about what the vector contains belongs to
`GET /v1/weights`; this process only signs and submits what that endpoint
publishes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import time
from typing import Any

import requests

API_URL = "https://api.pareton.ai/v1/weights"
NETWORK = "finney"
NETUID = 10
SUCCESS_BLOCKS = 360
RETRY_BLOCKS = 36
POLL_SECONDS = 12

logger = logging.getLogger("pareton-auditor")


class ValidatorPermitError(RuntimeError):
    """Raised when the signing hotkey may not set weights on the netuid."""


class WeightSetError(RuntimeError):
    """Raised when the chain rejected the weight vector."""


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


def fetch_metagraph() -> Any:
    """Read the metagraph. Only the signing hotkey's permit flag is used."""

    async def _fetch() -> Any:
        import bittensor as bt
        import bittensor.metagraph as mg

        async with bt.Client(NETWORK) as client:
            return await mg.fetch(client, NETUID, commitments=False)

    meta = asyncio.run(_fetch())
    if meta is None:
        raise RuntimeError(f"subnet {NETUID} does not exist on network {NETWORK}")
    return meta


def assert_validator_permit(meta: Any, hotkey: str) -> int:
    """Return the signing hotkey's uid, or raise before anything is signed.

    An unregistered hotkey or one without a permit fails inside the extrinsic
    as an opaque substrate error, so check the metagraph first and say which
    of the two it was.
    """
    neuron = meta.by_hotkey(hotkey)
    if neuron is None:
        raise ValidatorPermitError(
            f"signing hotkey is not registered on netuid {NETUID}"
        )
    if not getattr(neuron, "validator_permit", False):
        raise ValidatorPermitError(
            f"signing hotkey (uid {neuron.uid}) holds no validator permit "
            f"on netuid {NETUID}"
        )
    return int(neuron.uid)


def _outcome(result: Any) -> tuple[bool, str]:
    """`(ok, reason)` out of the SDK's ExtrinsicResult.

    A result missing ``success`` reads as a failure, so an SDK that changes
    this shape raises loudly rather than reporting a silent win.
    """
    ok = bool(getattr(result, "success", False))
    reason = getattr(result, "message", None) or getattr(result, "error", None)
    return ok, str(reason) if reason else ""


def set_weights(
    subtensor: Any,
    wallet: Any,
    meta: Any,
    *,
    uids: Any,
    weights: Any,
    version_key: int,
) -> None:
    """Sign and submit `(uids, weights)` once. `weights[i]` is UID `uids[i]`."""
    uids = list(uids)
    weights = [float(weight) for weight in weights]
    if len(uids) != len(weights):
        raise WeightSetError(
            f"uids and weights differ in length: {len(uids)} vs {len(weights)}"
        )
    uid = assert_validator_permit(meta, wallet.hotkey.ss58_address)

    import bittensor as bt

    intent = bt.SetWeights(
        netuid=NETUID,
        uids=uids,
        weights=weights,
        version_key=version_key,
    )
    try:
        result = subtensor.execute(
            intent,
            wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    except Exception as exc:
        raise WeightSetError(f"set_weights failed: {exc}") from exc
    ok, reason = _outcome(result)
    if not ok:
        raise WeightSetError(reason or "chain rejected the weight vector")
    logger.info(
        "set_weights accepted from uid %d: %d uids, version_key=%d",
        uid,
        len(uids),
        version_key,
    )


def submit_latest(subtensor: Any, wallet: Any, *, client: Any = requests) -> None:
    """Fetch and submit the latest published vector exactly once."""
    body = fetch_weights(client)
    weights = body["weights"]
    burn_uid = body["burn_uid"]
    burn_share = weights[burn_uid] if 0 <= burn_uid < len(weights) else 0.0
    logger.info("fetching metagraph for network=%s netuid=%d", NETWORK, NETUID)
    meta = fetch_metagraph()
    logger.info(
        "submitting %d weights to network=%s netuid=%d with burn_share=%.4f",
        len(weights),
        NETWORK,
        NETUID,
        burn_share,
    )
    set_weights(
        subtensor,
        wallet,
        meta,
        uids=range(len(weights)),
        weights=weights,
        version_key=body["version_key"],
    )
    logger.info("submitted weights computed at block %s", body["computed_at_block"])


def run(*, coldkey: str, hotkey: str, once: bool = False) -> bool:
    """Submit forever, using chain height as the retry clock."""
    import bittensor as bt

    logger.info("loading wallet coldkey=%s hotkey=%s", coldkey, hotkey)
    wallet = bt.Wallet(name=coldkey, hotkey=hotkey)
    logger.info("connecting to network=%s netuid=%d", NETWORK, NETUID)
    subtensor = bt.Subtensor(network=NETWORK)
    next_attempt = 0

    try:
        while True:
            try:
                head = int(subtensor.block)
                if head < next_attempt:
                    time.sleep(POLL_SECONDS)
                    continue

                try:
                    submit_latest(subtensor, wallet)
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The wallet settings also read environment variables:\n"
            "  PARETON_WALLET_NAME    --coldkey\n"
            "  PARETON_WALLET_HOTKEY  --hotkey\n"
            "A command-line flag overrides the environment variable.\n"
            f"Network, netuid and the API URL are fixed: {NETWORK}, "
            f"netuid {NETUID}, {API_URL}."
        ),
    )
    parser.add_argument("--hotkey", help="local validator hotkey name")
    parser.add_argument("--coldkey", help="local validator wallet name")
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

    coldkey = args.coldkey or os.environ.get("PARETON_WALLET_NAME", "").strip()
    hotkey = args.hotkey or os.environ.get("PARETON_WALLET_HOTKEY", "").strip()
    missing = [
        name
        for name, value in (
            ("--coldkey / PARETON_WALLET_NAME", coldkey),
            ("--hotkey / PARETON_WALLET_HOTKEY", hotkey),
        )
        if not value
    ]
    if missing:
        parser.error(f"missing wallet settings: {', '.join(missing)}")

    try:
        succeeded = run(coldkey=coldkey, hotkey=hotkey, once=args.once)
    except KeyboardInterrupt:
        logger.info("stopped")
        return 0
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
