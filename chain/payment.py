"""Verify the submission-fee transfer referenced by a patch commitment.

A miner pays the fee as a normal coldkey-signed transfer and puts the
payment's `(block, extrinsic_index)` in the commitment. Everything here works
on already-decoded extrinsics so the checks stay unit-testable without a chain:
`fetch_block_extrinsics` is the only function that talks to the network.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

RAO_PER_TAO = 1_000_000_000
# bt.Transfer wraps these; both move free balance to `dest`.
TRANSFER_FUNCTIONS = frozenset({"transfer_keep_alive", "transfer_allow_death"})


@dataclass(frozen=True)
class PaymentCheck:
    """Outcome of verifying one fee proof."""

    ok: bool
    reason: str = ""
    amount_rao: int = 0

    @classmethod
    def reject(cls, reason: str) -> PaymentCheck:
        return cls(ok=False, reason=reason)


def fee_rao(fee_tao: float | str) -> int:
    """Fee as integer RAO. Amounts are only ever compared as integers."""
    return int(Decimal(str(fee_tao)) * RAO_PER_TAO)


def _ss58(value: Any) -> str | None:
    """ss58 address out of a decoded AccountId / MultiAddress."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("Id", "id", "value"):
            inner = value.get(key)
            if isinstance(inner, str):
                return inner
    return None


def _rao(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _call_args(call: dict) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for arg in call.get("call_args") or []:
        if isinstance(arg, dict) and "name" in arg:
            args[str(arg["name"])] = arg.get("value")
    return args


def extract_transfer(extrinsic: Any) -> tuple[str, str, int] | None:
    """`(signer, dest, amount_rao)` for a balance transfer, else None."""
    if not isinstance(extrinsic, dict):
        return None
    call = extrinsic.get("call")
    if not isinstance(call, dict):
        return None
    if call.get("call_module") != "Balances":
        return None
    if call.get("call_function") not in TRANSFER_FUNCTIONS:
        return None
    signer = _ss58(extrinsic.get("address"))
    args = _call_args(call)
    dest = _ss58(args.get("dest"))
    amount = _rao(args.get("value"))
    if signer is None or dest is None or amount is None:
        return None
    return signer, dest, amount


def verify_payment(
    *,
    extrinsics: Sequence[Any],
    extrinsic_index: int,
    recipient: str,
    min_amount_rao: int,
    hotkey: str,
    coldkey: str = "",
) -> PaymentCheck:
    """Check that the referenced extrinsic really paid the fee.

    The signer must be the submitting hotkey or the coldkey that owns it per
    the metagraph, so a miner cannot point at somebody else's transfer.
    """
    if extrinsic_index < 0 or extrinsic_index >= len(extrinsics):
        return PaymentCheck.reject("payment_index_out_of_range")
    transfer = extract_transfer(extrinsics[extrinsic_index])
    if transfer is None:
        return PaymentCheck.reject("payment_not_a_transfer")
    signer, dest, amount_rao = transfer
    if dest != recipient:
        return PaymentCheck.reject("payment_recipient_mismatch")
    if amount_rao < min_amount_rao:
        return PaymentCheck.reject("payment_below_fee")
    owners = {key for key in (hotkey, coldkey) if key}
    if signer not in owners:
        return PaymentCheck.reject("payment_signer_not_miner")
    return PaymentCheck(ok=True, amount_rao=amount_rao)


def fetch_block_extrinsics(subtensor: Any, block: int) -> list[Any] | None:
    """Decoded extrinsics of one block, or None when it cannot be read.

    `subtensor.block_info` decodes fully; the `chain.block_info` read returns
    only `module.function` summaries, which cannot carry dest or amount.
    """
    try:
        info = subtensor.block_info(block)
    except Exception as exc:
        logger.warning("block_info(%d) failed: %s", block, exc)
        return None
    if info is None:
        return None
    return list(getattr(info, "extrinsics", None) or [])
