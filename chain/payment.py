"""Verify the submission-fee transfer referenced by a patch commitment.

A miner pays the fee as a normal coldkey-signed transfer and puts the
payment's `(block, extrinsic_index)` in the commitment. Everything here works
on already-decoded extrinsics/events so the checks stay unit-testable without a
chain: `fetch_block_payment_view` is the only function that talks to the network.
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


@dataclass(frozen=True)
class BlockPaymentView:
    """Decoded extrinsics and System.Events for one block."""

    extrinsics: list[Any]
    events: list[Any]


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


def extrinsic_index_of(event: Any) -> int | None:
    """Block extrinsic index for a System.Events record, if any."""
    if not isinstance(event, dict):
        return None
    idx = event.get("extrinsic_idx")
    if isinstance(idx, int) and not isinstance(idx, bool):
        return idx
    if isinstance(idx, str) and idx.isdigit():
        return int(idx)
    phase = event.get("phase")
    if isinstance(phase, dict):
        applied = phase.get("ApplyExtrinsic")
        if isinstance(applied, int) and not isinstance(applied, bool):
            return applied
        if isinstance(applied, str) and applied.isdigit():
            return int(applied)
    return None


def _event_ids(event: Any) -> tuple[str, str] | None:
    if not isinstance(event, dict):
        return None
    body = event.get("event")
    if not isinstance(body, dict):
        return None
    module = body.get("module_id")
    event_id = body.get("event_id")
    if isinstance(module, str) and isinstance(event_id, str):
        return module, event_id
    return None


def extrinsic_succeeded(events: Sequence[Any], extrinsic_index: int) -> bool | None:
    """Whether the extrinsic dispatched successfully.

    Failed Substrate extrinsics stay in the block with intact call data, so a
    fee check that only reads call args would accept unpaid proofs. Returns
    True/False when a System success/fail event is present for the index, else
    None when the outcome cannot be determined.
    """
    saw_success = False
    saw_failure = False
    for event in events:
        if extrinsic_index_of(event) != extrinsic_index:
            continue
        ids = _event_ids(event)
        if ids is None:
            continue
        module, event_id = ids
        if module != "System":
            continue
        if event_id == "ExtrinsicSuccess":
            saw_success = True
        elif event_id == "ExtrinsicFailed":
            saw_failure = True
    if saw_failure:
        return False
    if saw_success:
        return True
    return None


def verify_payment(
    *,
    extrinsics: Sequence[Any],
    events: Sequence[Any],
    extrinsic_index: int,
    recipient: str,
    min_amount_rao: int,
    hotkey: str,
    coldkey: str = "",
) -> PaymentCheck:
    """Check that the referenced extrinsic really paid the fee.

    The signer must be the submitting hotkey or the coldkey that owns it per
    the metagraph, so a miner cannot point at somebody else's transfer. The
    extrinsic must also have dispatched successfully on-chain.
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
    outcome = extrinsic_succeeded(events, extrinsic_index)
    if outcome is False:
        return PaymentCheck.reject("payment_extrinsic_failed")
    if outcome is not True:
        return PaymentCheck.reject("payment_outcome_unknown")
    return PaymentCheck(ok=True, amount_rao=amount_rao)


def fetch_block_payment_view(subtensor: Any, block: int) -> BlockPaymentView | None:
    """Decoded extrinsics + events for one block, or None when unreadable.

    `subtensor.block_info` decodes fully; the `chain.block_info` read returns
    only `module.function` summaries, which cannot carry dest or amount.
    Events come from `System.Events` so failed transfers can be rejected.
    """
    try:
        info = subtensor.block_info(block)
    except Exception as exc:
        logger.warning("block_info(%d) failed: %s", block, exc)
        return None
    if info is None:
        return None
    try:
        raw_events = subtensor.query(("System", "Events"), block=block)
    except Exception as exc:
        logger.warning("System.Events at block %d failed: %s", block, exc)
        return None
    if hasattr(raw_events, "value"):
        raw_events = raw_events.value
    events = list(raw_events or [])
    return BlockPaymentView(
        extrinsics=list(getattr(info, "extrinsics", None) or []),
        events=events,
    )
