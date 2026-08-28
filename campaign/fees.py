"""Campaign-pinned submission fee terms."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

RAO_PER_TAO = Decimal(1_000_000_000)
SUBMISSION_FEE_KEYS = frozenset({"amount_tao", "recipient"})


def validate_submission_fee(value: dict[str, Any]) -> dict[str, str]:
    """Return canonical fee terms suitable for hashing and payment checks."""
    if not isinstance(value, dict):
        raise ValueError("submission_fee must be an object")
    unknown = set(value) - SUBMISSION_FEE_KEYS
    if unknown:
        raise ValueError(
            f"submission_fee has unknown keys: {sorted(unknown)} "
            f"(allowed: {sorted(SUBMISSION_FEE_KEYS)})"
        )

    raw_amount = value.get("amount_tao")
    if isinstance(raw_amount, bool) or raw_amount is None:
        raise ValueError("submission_fee.amount_tao must be a non-negative TAO amount")
    try:
        amount = Decimal(str(raw_amount).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(
            "submission_fee.amount_tao must be a non-negative TAO amount"
        ) from None
    if not amount.is_finite() or amount < 0:
        raise ValueError("submission_fee.amount_tao must be a non-negative TAO amount")
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
        "amount_tao": format(amount.normalize(), "f"),
        "recipient": recipient,
    }


def submission_fee_rao(value: dict[str, Any]) -> int:
    """Convert validated campaign fee terms to the exact integer chain unit."""
    fee = validate_submission_fee(value)
    return int(Decimal(fee["amount_tao"]) * RAO_PER_TAO)
