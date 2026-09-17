"""Publish an immediate campaign fee change: python -m campaign.set_fee."""

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


def set_fee(campaign_id: str, amount_tao: str, *, current_block: int) -> dict:
    if type(current_block) is not int or not 0 < current_block <= 2**63 - 1:
        raise ValueError("current_block must be a positive integer chain height")
    fee = validate_submission_fee(
        {"amount_tao": amount_tao, "recipient": TRUSTED_PAYMENT_RECIPIENT}
    )
    entry = {**fee, "effective_from_block": current_block}
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT submission_fee_history FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("campaign not found")
            previous = validate_fee_history(row[0])
            if current_block <= previous[-1]["effective_from_block"]:
                raise ValueError(
                    "fee history already reaches the current block; retry after the chain advances"
                )
            history = validate_fee_history([*previous, entry])
            cur.execute(
                "UPDATE campaigns SET submission_fee_history = %s, updated_at = now() WHERE id = %s",
                (Json(history), campaign_id),
            )
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True, type=UUID)
    parser.add_argument("--amount-tao", required=True)
    args = parser.parse_args(argv)
    import bittensor as bt

    with bt.Subtensor(network=config.SUBTENSOR_NETWORK) as subtensor:
        current_block = int(subtensor.block)
    entry = set_fee(
        str(args.campaign_id),
        args.amount_tao,
        current_block=current_block,
    )
    print(
        f"Published {entry['amount_tao']} TAO from block {entry['effective_from_block']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
