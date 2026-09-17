"""Schedule an append-only campaign fee change: python -m campaign.set_fee."""

from __future__ import annotations

import argparse
from uuid import UUID

from psycopg2.extras import Json

import config
from campaign.fees import (
    TRUSTED_PAYMENT_RECIPIENT,
    validate_fee_history,
    validate_submission_fee,
)
from db.connection import db_connection


def schedule_fee(
    campaign_id: str, amount_tao: str, effective_from_block: int, *, current_block: int
) -> dict:
    # Publish changes ahead of activation. Old payment proofs retain their terms.
    if effective_from_block < current_block + 100:
        raise ValueError(
            "effective_from_block must be at least 100 blocks in the future"
        )
    fee = validate_submission_fee(
        {"amount_tao": amount_tao, "recipient": TRUSTED_PAYMENT_RECIPIENT}
    )
    entry = {**fee, "effective_from_block": effective_from_block}
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT submission_fee_history FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("campaign not found")
            history = validate_fee_history([*row[0], entry])
            cur.execute(
                "UPDATE campaigns SET submission_fee_history = %s, updated_at = now() WHERE id = %s",
                (Json(history), campaign_id),
            )
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True, type=UUID)
    parser.add_argument("--amount-tao", required=True)
    parser.add_argument("--effective-from-block", required=True, type=int)
    args = parser.parse_args(argv)
    import bittensor as bt

    with bt.Subtensor(network=config.SUBTENSOR_NETWORK) as subtensor:
        current_block = int(subtensor.block)
    entry = schedule_fee(
        str(args.campaign_id),
        args.amount_tao,
        args.effective_from_block,
        current_block=current_block,
    )
    print(
        f"Scheduled {entry['amount_tao']} TAO from block {entry['effective_from_block']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
