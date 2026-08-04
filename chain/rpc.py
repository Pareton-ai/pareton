"""Bittensor RPC helpers for Pareton (bittensor 11.x SDK).

11.x split the SDK: sync low-level client (`bt.Subtensor`: block info, extrinsic
submit) and async graph reads (`bittensor.metagraph.fetch` / `fetch_commitments`
over `bt.Client`). The worker loop is sync, so graph reads go through
`asyncio.run` with a short-lived async client per attempt.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ChainError(RuntimeError):
    """Raised when a chain RPC fails after all retries."""


def _retry(
    fn: Callable[[], Any],
    *,
    label: str,
    attempts: int,
    delay_s: float,
) -> Any:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "%s failed (attempt %d/%d): %s",
                label,
                i + 1,
                attempts,
                exc,
            )
            if i < attempts - 1:
                time.sleep(delay_s)
    raise ChainError(f"{label} failed after {attempts} attempts: {last_exc}")


def _commitment_entries(commitment: Any) -> list[tuple[int, str]]:
    """Map an 11.x NeuronCommitment to (block, data_str) entries.

    Plaintext commitments contribute their current data at the commit block;
    timelock-revealed payloads contribute their reveal history. Still-sealed
    (encrypted) entries are skipped until the chain reveals them.
    """
    entries = [
        (int(block), str(data))
        for block, data in (getattr(commitment, "revealed", None) or [])
    ]
    if not getattr(commitment, "encrypted", False):
        data = getattr(commitment, "data", None)
        if data:
            entries.append((int(commitment.block), str(data)))
    return entries


async def _graph_and_commitments(network: str, netuid: int) -> tuple[Any, Any]:
    import bittensor as bt
    import bittensor.metagraph as mg

    async with bt.Client(network) as client:
        meta = await mg.fetch(client, netuid, commitments=False)
        commitments = await mg.fetch_commitments(client, netuid)
    return meta, commitments


def fetch_chain_view(
    subtensor: Any,
    netuid: int,
    *,
    network: str = "finney",
    attempts: int = 3,
    delay_s: float = 30.0,
) -> tuple[Any, dict[str, list[tuple[int, str]]], int, str | None]:
    """One chain read: metagraph + revealed commitments + block + block hash."""

    def _inner() -> tuple[Any, dict[str, list[tuple[int, str]]], int, str | None]:
        meta, commitments = asyncio.run(_graph_and_commitments(network, netuid))
        if meta is None:
            raise ChainError(f"subnet {netuid} does not exist")
        current_block = int(subtensor.block)
        try:
            block_hash = subtensor.block_info().hash
        except Exception as exc:
            logger.warning(
                "Block hash lookup failed: %s — continuing with block_hash=None.",
                exc,
            )
            block_hash = None
        revealed: dict[str, list[tuple[int, str]]] = {}
        for hotkey, commitment in commitments.items():
            entries = _commitment_entries(commitment)
            if entries:
                revealed[str(hotkey)] = entries
        return meta, revealed, current_block, block_hash

    return _retry(
        _inner,
        label="fetch_chain_view",
        attempts=attempts,
        delay_s=delay_s,
    )


def fetch_metagraph(
    subtensor: Any,
    netuid: int,
    *,
    network: str = "finney",
    attempts: int = 3,
    delay_s: float = 30.0,
) -> tuple[Any, int, str | None]:
    """Metagraph + current block + block hash."""
    meta, _revealed, block, block_hash = fetch_chain_view(
        subtensor, netuid, network=network, attempts=attempts, delay_s=delay_s
    )
    return meta, block, block_hash


def fetch_revealed_commitments(
    subtensor: Any,
    netuid: int,
    *,
    network: str = "finney",
    attempts: int = 3,
    delay_s: float = 30.0,
) -> dict[str, list[tuple[int, str]]]:
    """Return `{hotkey: [(block, data_str), ...]}` for the subnet."""
    _meta, revealed, _block, _hash = fetch_chain_view(
        subtensor, netuid, network=network, attempts=attempts, delay_s=delay_s
    )
    return revealed
