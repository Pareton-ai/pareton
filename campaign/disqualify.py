"""Disqualify one hotkey from one campaign.

Usage:
    PARETON_DATABASE_URL=... python -m campaign.disqualify \
      --campaign-id UUID --hotkey SS58 --reason TEXT \
      --evidence-ref PAR-123 --operator NAME

UIDs are resolved from the live metagraph and are never stored.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import config
from chain.rpc import fetch_metagraph
from round.store import disqualify_campaign_hotkey


def hotkey_for_uid(meta: Any, uid: int) -> str:
    """Resolve one live UID through authoritative ``by_hotkey`` records."""
    if uid < 0:
        raise ValueError("uid must be non-negative")
    matches: list[str] = []
    for value in getattr(meta, "hotkeys", []):
        hotkey = str(value)
        neuron = meta.by_hotkey(hotkey)
        if neuron is not None and int(neuron.uid) == uid:
            matches.append(hotkey)
    if len(matches) != 1:
        raise ValueError(
            f"uid {uid} resolved to {len(matches)} hotkeys on the live metagraph"
        )
    return matches[0]


def _resolve_uid(uid: int, *, network: str, netuid: int) -> str:
    import bittensor as bt

    subtensor = bt.Subtensor(network=network)
    meta, _block, _hash = fetch_metagraph(
        subtensor,
        netuid,
        network=network,
    )
    return hotkey_for_uid(meta, uid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Disqualify a hotkey from one campaign"
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
        result = disqualify_campaign_hotkey(
            args.campaign_id,
            hotkey,
            reason=args.reason,
            evidence_ref=args.evidence_ref,
            disqualified_by=args.operator,
            epsilon=config.OVERTAKE_EPSILON,
        )
    # This is the CLI boundary. Convert SDK and database failures into a clean
    # operator-facing error instead of emitting a traceback.
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    action = "created" if result.created else "already existed"
    print(f"campaign exclusion: {action}")
    print(f"campaign: {result.campaign_id}")
    print(f"hotkey: {result.hotkey}")
    print(f"submissions disqualified: {result.submissions_disqualified}")
    print(f"pending jobs stopped: {result.pending_jobs_stopped}")
    print(f"leader vacated: {'yes' if result.leader_vacated else 'no'}")
    if result.replacement_submission_id is None:
        print("replacement leader: none")
    else:
        print(
            "replacement leader: "
            f"{result.replacement_hotkey} ({result.replacement_submission_id})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
