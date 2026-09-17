"""Exact campaign fees and block-effective history."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

TRUSTED_PAYMENT_RECIPIENT = "5CiieAa5nzSMbw4LPkh2hqv9rfMPZX9ZfEcSjh3SYWNBzk3K"
MAX_RAO = 2**64 - 1

RAO_PER_TAO = Decimal(1_000_000_000)
SUBMISSION_FEE_KEYS = frozenset({"amount_tao", "recipient"})


def validate_submission_fee(value: dict[str, Any]) -> dict[str, str]:
    """Return canonical fee terms suitable for storage and payment checks."""
    if not isinstance(value, dict):
        raise ValueError("submission_fee must be an object")
    unknown = set(value) - SUBMISSION_FEE_KEYS
    if unknown:
        raise ValueError(
            f"submission_fee has unknown keys: {sorted(unknown)} "
            f"(allowed: {sorted(SUBMISSION_FEE_KEYS)})"
        )

    raw_amount = value.get("amount_tao")
    if not isinstance(raw_amount, (str, int, Decimal)) or isinstance(raw_amount, bool):
        raise ValueError("submission_fee.amount_tao must be a non-negative TAO amount")
    try:
        amount = Decimal(str(raw_amount).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(
            "submission_fee.amount_tao must be a non-negative TAO amount"
        ) from None
    if not amount.is_finite() or amount < 0:
        raise ValueError("submission_fee.amount_tao must be a non-negative TAO amount")
    if amount != 0 and amount.adjusted() < -9:
        raise ValueError("submission_fee.amount_tao must resolve to a whole RAO")
    if amount > Decimal(MAX_RAO) / RAO_PER_TAO:
        raise ValueError("submission_fee.amount_tao exceeds the chain balance range")
    with localcontext() as ctx:
        ctx.prec = max(40, len(amount.as_tuple().digits) + 10)
        amount_rao = amount * RAO_PER_TAO
    if amount_rao != amount_rao.to_integral_value():
        raise ValueError("submission_fee.amount_tao must resolve to a whole RAO")

    recipient = value.get("recipient")
    if not isinstance(recipient, str) or not recipient.strip():
        raise ValueError("submission_fee.recipient must be a non-empty address")
    recipient = recipient.strip()
    if any(char.isspace() for char in recipient):
        raise ValueError("submission_fee.recipient must not contain whitespace")

    return {
        "amount_tao": "0" if amount == 0 else format(amount.normalize(), "f"),
        "recipient": recipient,
    }


def submission_fee_rao(value: dict[str, Any]) -> int:
    """Convert validated campaign fee terms to the exact integer chain unit."""
    fee = validate_submission_fee(value)
    return int(Decimal(fee["amount_tao"]) * RAO_PER_TAO)


def validate_fee_history(value: Any) -> list[dict[str, Any]]:
    """Require a genesis entry and strictly increasing whole block heights."""
    if not isinstance(value, list) or not value:
        raise ValueError("submission_fee_history must be a non-empty array")
    result = []
    previous = -1
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {
            "effective_from_block",
            "amount_tao",
            "recipient",
        }:
            raise ValueError("invalid submission_fee_history entry")
        block = entry["effective_from_block"]
        if type(block) is not int or block <= previous or block > 2**63 - 1:
            raise ValueError(
                "fee history blocks must be increasing non-negative integers"
            )
        fee = validate_submission_fee({k: entry[k] for k in SUBMISSION_FEE_KEYS})
        result.append({**fee, "effective_from_block": block})
        previous = block
    if result[0]["effective_from_block"] != 0:
        raise ValueError("fee history must start at block zero")
    return result


def fee_at_block(history: Any, block: int) -> dict[str, str]:
    if type(block) is not int or block < 0:
        raise ValueError("fee lookup requires a non-negative integer block")
    entries = validate_fee_history(history)
    entry = next(e for e in reversed(entries) if e["effective_from_block"] <= block)
    return {k: entry[k] for k in SUBMISSION_FEE_KEYS}
