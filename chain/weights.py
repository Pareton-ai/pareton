"""Set the subnet weight vector on chain (bittensor 11.x SDK).

The first validator-side signing path in the repo: the rest of `chain/` only
reads. Everything here takes an already-built sparse vector, so building it,
storing it, and deciding when to set it live elsewhere.

A set that fails every attempt raises and writes nothing else. The previous
on-chain vector stays standing and keeps paying the miners it named, which is
the correct failure mode: an outage of ours must never stop paying miners.
There is deliberately no fallback that submits a different vector.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import config

logger = logging.getLogger(__name__)


class WeightSetError(RuntimeError):
    """Raised when the weight vector could not be set, with the last reason."""


class ValidatorPermitError(WeightSetError):
    """Raised when the signing hotkey may not set weights on the netuid."""


def dense_to_sparse(dense: Sequence[float]) -> tuple[list[int], list[float]]:
    """Non-zero entries of a dense vector as `(uids, weights)`, ascending by uid.

    The single dense-to-sparse conversion point. A dense vector is an internal
    representation and never goes on the wire.
    """
    pairs = [(uid, float(weight)) for uid, weight in enumerate(dense) if weight]
    return [uid for uid, _ in pairs], [weight for _, weight in pairs]


def _outcome(result: Any) -> tuple[bool, str]:
    """`(ok, reason)` out of whatever the SDK handed back.

    11.x returns an ExtrinsicResult (`success` / `message`); other releases
    return a bare bool or an `(ok, reason)` tuple. Normalise all three so the
    retry loop reads one shape.
    """
    if isinstance(result, bool):
        return result, ""
    if isinstance(result, tuple):
        ok = bool(result[0])
        return ok, str(result[1]) if len(result) > 1 else ""
    ok = bool(getattr(result, "success", False))
    reason = getattr(result, "message", None) or getattr(result, "error", None)
    return ok, str(reason) if reason else ""


def assert_validator_permit(meta: Any, hotkey: str, netuid: int) -> int:
    """Return the signing hotkey's uid, or raise before anything is signed.

    An unregistered hotkey or one without a permit fails inside the extrinsic
    as an opaque substrate error, so check the metagraph first and say which
    of the two it was. The uid, not the hotkey, goes in the message.
    """
    neuron = meta.by_hotkey(hotkey)
    if neuron is None:
        raise ValidatorPermitError(
            f"signing hotkey is not registered on netuid {netuid}"
        )
    if not getattr(neuron, "validator_permit", False):
        raise ValidatorPermitError(
            f"signing hotkey (uid {neuron.uid}) holds no validator permit "
            f"on netuid {netuid}"
        )
    return int(neuron.uid)


def set_weights(
    subtensor: Any,
    wallet: Any,
    meta: Any,
    *,
    netuid: int,
    uids: Sequence[int],
    weights: Sequence[float],
    version_key: int = config.VERSION_KEY,
    burn_uid: int = config.BURN_UID,
    attempts: int = config.CHAIN_RETRY_ATTEMPTS,
    delay_s: float = config.CHAIN_RETRY_DELAY_S,
) -> None:
    """Sign and submit `(uids, weights)`, retrying a rejection.

    `uids` and `weights` are the sparse form: non-zero entries only, ascending
    by uid. They are submitted as given. Raises `WeightSetError` after the last
    attempt so the caller can log it and keep its loop running.
    """
    if len(uids) != len(weights):
        raise WeightSetError(
            f"uids and weights differ in length: {len(uids)} vs {len(weights)}"
        )
    assert_validator_permit(meta, wallet.hotkey.ss58_address, netuid)

    import bittensor as bt

    intent = bt.SetWeights(
        netuid=netuid,
        uids=list(uids),
        weights=[float(weight) for weight in weights],
        version_key=version_key,
    )
    burn_share = next(
        (float(weight) for uid, weight in zip(uids, weights) if uid == burn_uid), 0.0
    )
    last_reason = ""
    for attempt in range(1, attempts + 1):
        try:
            ok, reason = _outcome(
                subtensor.execute(
                    intent,
                    wallet,
                    wait_for_inclusion=True,
                    wait_for_finalization=True,
                )
            )
        except Exception as exc:
            ok, reason = False, str(exc)
        if ok:
            logger.info(
                "set_weights accepted on attempt %d/%d: %d uids, "
                "version_key=%d, burn_share=%.4f",
                attempt,
                attempts,
                len(uids),
                version_key,
                burn_share,
            )
            return
        last_reason = reason or "chain rejected the weight vector"
        logger.warning(
            "set_weights failed (attempt %d/%d): %s", attempt, attempts, last_reason
        )
        if attempt < attempts:
            time.sleep(delay_s)
    raise WeightSetError(f"set_weights failed after {attempts} attempts: {last_reason}")
