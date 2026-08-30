"""Waive a campaign hotkey disqualification for future submissions.

Usage:
    PARETON_DATABASE_URL=... python -m campaign.waive \
      --campaign-id UUID --hotkey SS58 --reason TEXT \
      --evidence-ref PAR-123 --operator NAME

UIDs are resolved from the live metagraph and are never stored.
"""

from __future__ import annotations

import argparse
import sys

import config
from campaign.disqualify import _resolve_uid
from round.store import waive_campaign_hotkey


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Waive a campaign hotkey disqualification for future submissions"
    )
    parser.add_argument("--campaign-id", required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--hotkey")
    identity.add_argument("--uid", type=int)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence-ref", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--network", default=config.SUBTENSOR_NETWORK)
    parser.add_argument("--netuid", type=int, default=config.NETUID)
    args = parser.parse_args(argv)

    try:
        hotkey = args.hotkey
        if hotkey is None:
            hotkey = _resolve_uid(args.uid, network=args.network, netuid=args.netuid)
            print(f"resolved uid {args.uid} to hotkey {hotkey}")
        result = waive_campaign_hotkey(
            args.campaign_id,
            hotkey,
            reason=args.reason,
            evidence_ref=args.evidence_ref,
            waived_by=args.operator,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    action = "created" if result.created else "already existed"
    print(f"campaign waiver: {action}")
    print(f"campaign: {result.campaign_id}")
    print(f"hotkey: {result.hotkey}")
    print("historical verdicts changed: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
