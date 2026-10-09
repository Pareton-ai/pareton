"""Update campaign disclosure terms without changing its hash or signoff."""

import argparse
import json
from uuid import UUID

from psycopg2.extras import Json

from campaign.visibility import validate_patch_visibility
from db.connection import db_connection


def set_patch_visibility(campaign_id: str, policy: dict) -> dict:
    policy = validate_patch_visibility(policy)
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE campaigns SET patch_visibility = %s, updated_at = now()
                   WHERE id = %s RETURNING id""",
            (Json(policy), str(campaign_id)),
        )
        if cur.fetchone() is None:
            raise ValueError("campaign not found")
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True, type=UUID)
    parser.add_argument(
        "--mode", required=True, choices=("private", "public_after_reveal")
    )
    parser.add_argument("--reveal-delay-s", type=int)
    args = parser.parse_args(argv)
    policy = {"mode": args.mode}
    if args.reveal_delay_s is not None:
        policy["reveal_delay_s"] = args.reveal_delay_s
    try:
        result = set_patch_visibility(str(args.campaign_id), policy)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
