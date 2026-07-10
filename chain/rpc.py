"""Bittensor RPC helpers for Pareton (inlined from legacy Cacheon validator.chain)."""

from __future__ import annotations

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


def fetch_metagraph(
    subtensor: Any,
    netuid: int,
    *,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> tuple[Any, int, str | None]:
    """Fetch metagraph + current block + block hash."""

    def _inner() -> tuple[Any, int, str | None]:
        metagraph = subtensor.metagraph(netuid)
        current_block = int(subtensor.block)
        try:
            block_hash = subtensor.substrate.get_block_hash(current_block)
        except Exception as exc:
            logger.warning(
                "Block hash lookup failed: %s — continuing with block_hash=None.",
                exc,
            )
            block_hash = None
        return metagraph, current_block, block_hash

    return _retry(
        _inner,
        label="fetch_metagraph",
        attempts=attempts,
        delay_s=delay_s,
    )


def _decode_raw_commitment(raw: str | bytes, *, _depth: int = 0) -> str:
    """Normalize a raw on-chain commitment value to a plain JSON string."""
    if _depth > 3:
        return str(raw)

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    s = str(raw)

    if s.startswith("0x"):
        try:
            decoded = bytes.fromhex(s[2:])
        except ValueError:
            return s
        text = decoded.decode("utf-8", errors="replace")
        idx_brace = text.find("{")
        idx_0x = text.find("0x")
        if idx_0x >= 0 and (idx_brace < 0 or idx_0x < idx_brace):
            return _decode_raw_commitment(text[idx_0x:], _depth=_depth + 1)
        return text[idx_brace:] if idx_brace >= 0 else text

    idx = s.find("{")
    if idx > 0:
        return s[idx:]
    return s


def _fetch_commitments_raw_substrate(
    subtensor: Any,
    netuid: int,
) -> dict[str, list[tuple[int, str]]]:
    """Fallback: query the substrate storage map directly."""
    result = subtensor.substrate.query_map(
        module="Commitments",
        storage_function="RevealedCommitments",
        params=[netuid],
    )
    out: dict[str, list[tuple[int, str]]] = {}
    for key, value in result:
        hotkey = str(key)
        entries = []
        for entry in value or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                data_raw, block = entry[0], entry[1]
            else:
                continue
            entries.append((int(block), _decode_raw_commitment(data_raw)))
        if entries:
            out[hotkey] = entries
    return out


def fetch_revealed_commitments(
    subtensor: Any,
    netuid: int,
    *,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> dict[str, list[tuple[int, str]]]:
    """Return `{hotkey: [(block, data_str), ...]}` for the subnet."""

    def _inner() -> dict[str, list[tuple[int, str]]]:
        for method_name in (
            "get_all_revealed_commitments",
            "get_revealed_commitments",
        ):
            fn = getattr(subtensor, method_name, None)
            if callable(fn):
                try:
                    return fn(netuid) or {}
                except ValueError as exc:
                    if "fromhex" in str(exc) or "hexadecimal" in str(exc):
                        logger.warning(
                            "SDK %s hit hex decode error, falling back "
                            "to raw substrate query: %s",
                            method_name,
                            exc,
                        )
                        return _fetch_commitments_raw_substrate(subtensor, netuid)
                    raise
        raise RuntimeError(
            "subtensor has no get_all_revealed_commitments / "
            "get_revealed_commitments method -- bittensor version mismatch?"
        )

    return _retry(
        _inner,
        label="fetch_revealed_commitments",
        attempts=attempts,
        delay_s=delay_s,
    )
