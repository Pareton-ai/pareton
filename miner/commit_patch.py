#!/usr/bin/env python3
"""Commit a Pareton patch on-chain (Stage 0).

Flow:
  1. Request Pareton-presigned S3 upload (or reuse --retrieval-url)
  2. PUT patch bytes
  3. Commitments.set_commitment with v2 patch payload (plaintext Raw fields)

Usage:
    python miner/commit_patch.py \\
      --campaign-id <uuid> \\
      --patch ./my.patch \\
      --api-base http://localhost:8000 \\
      --wallet-name <wallet> --wallet-hotkey <hotkey> \\
      --network finney --netuid 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chain.commitment import encode_patch_commitment, fetch_metagraph  # noqa: E402
from storage.s3 import patch_url_hotkey  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _http_json(method: str, url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _put_bytes(url: str, data: bytes) -> None:
    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"Content-Type": "text/plain"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()


def _plaintext_fields(payload: str) -> list[dict]:
    """Split payload into Data::Raw chunks (each variant holds <=128 bytes).

    Finney's CommitmentInfo fields is a BoundedVec with MaxFields=3: a 4th
    field makes validate_transaction trap (wasm unreachable) instead of
    erroring cleanly, so fail fast here instead.
    """
    raw = payload.encode()
    fields = [
        {f"Raw{len(chunk)}": "0x" + chunk.hex()}
        for chunk in (raw[i : i + 128] for i in range(0, len(raw), 128))
    ]
    if len(fields) > 3:
        raise ValueError(
            f"commitment payload is {len(raw)} bytes ({len(fields)} Raw fields); "
            "finney MaxFields=3 caps plaintext commitments at 384 bytes"
        )
    return fields


def _timelock_fields(payload: str) -> tuple[list[dict], int]:
    from datetime import datetime, timedelta, timezone

    from bittensor import timelock

    sealed = timelock.encrypt(
        payload,
        reveal_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    field = {
        "TimelockEncrypted": {
            "encrypted": sealed.ciphertext,
            "reveal_round": sealed.reveal_round,
        }
    }
    return [field], sealed.reveal_round


def _await_visible(
    network: str, netuid: int, hotkey: str, payload: str, *, timeout_s: float = 90.0
) -> str | None:
    """Poll until the chain exposes our payload, or return None on timeout."""
    import asyncio
    import time

    import bittensor as bt
    import bittensor.metagraph as mg

    async def _poll() -> str | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            async with bt.Client(network) as client:
                commitments = await mg.fetch_commitments(client, netuid)
            c = commitments.get(hotkey)
            if c is not None:
                if (getattr(c, "data", None) or "") == payload:
                    return f"plaintext visible at block {c.block}"
                if any(str(data) == payload for _block, data in (c.revealed or [])):
                    return f"revealed at block {c.block}"
            await asyncio.sleep(6)
        return None

    return asyncio.run(_poll())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Commit a Pareton patch (Stage 0)")
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--patch", required=True, type=Path, help="Unified git diff file")
    p.add_argument("--api-base", default="http://127.0.0.1:8000")
    p.add_argument("--retrieval-url", default=None, help="Skip upload; use this URL")
    p.add_argument("--wallet-name", required=True)
    p.add_argument("--wallet-hotkey", default="default")
    p.add_argument("--network", default="finney")
    p.add_argument("--netuid", type=int, default=10)
    p.add_argument(
        "--timelock",
        action="store_true",
        help="Seal payload with drand timelock instead of plaintext "
        "(experimental: finney reveal has not produced entries in ~52 days)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commitment payload without submitting on-chain",
    )
    args = p.parse_args(argv)

    if not args.patch.is_file():
        print(f"error: patch not found: {args.patch}", file=sys.stderr)
        return 1

    patch_bytes = args.patch.read_bytes()
    patch_hash = _sha256_file(args.patch)

    import bittensor as bt

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    hotkey = wallet.hotkey.ss58_address

    # Fetch campaign for baseline_commit
    try:
        campaign = _http_json(
            "GET", f"{args.api_base.rstrip('/')}/v1/campaigns/{args.campaign_id}"
        )
    except Exception as exc:
        print(f"error: failed to fetch campaign: {exc}", file=sys.stderr)
        return 1
    baseline_commit = campaign["baseline_commit"]

    if args.retrieval_url:
        retrieval_url = args.retrieval_url
        if patch_url_hotkey(retrieval_url) != hotkey:
            print(
                f"error: --retrieval-url path hotkey must match wallet hotkey {hotkey}",
                file=sys.stderr,
            )
            return 1
    else:
        try:
            presign = _http_json(
                "POST",
                f"{args.api_base.rstrip('/')}/v1/uploads/patch",
                {"campaign_id": args.campaign_id, "hotkey": hotkey},
            )
        except Exception as exc:
            print(f"error: presign failed: {exc}", file=sys.stderr)
            return 1
        try:
            _put_bytes(presign["upload_url"], patch_bytes)
        except urllib.error.URLError as exc:
            print(f"error: upload failed: {exc}", file=sys.stderr)
            return 1
        retrieval_url = presign["retrieval_url"]
        print(f"uploaded patch to {retrieval_url}")

    try:
        payload = encode_patch_commitment(
            campaign_id=args.campaign_id,
            baseline_commit=baseline_commit,
            patch_hash=patch_hash,
            retrieval_url=retrieval_url,
        )
        if args.timelock:
            fields, reveal_round = _timelock_fields(payload)
            mode = f"timelock reveal_round={reveal_round}"
        else:
            fields = _plaintext_fields(payload)
            mode = "plaintext"
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"commitment payload ({len(payload.encode())} bytes):")
    print(payload)

    if args.dry_run:
        print("dry-run: not submitting on-chain")
        return 0

    subtensor = bt.Subtensor(network=args.network)
    meta = fetch_metagraph(subtensor, args.netuid, network=args.network)
    if meta.by_hotkey(hotkey) is None:
        print(
            f"error: hotkey {hotkey} not registered on netuid {args.netuid}",
            file=sys.stderr,
        )
        return 1

    try:
        call = bt.calls.Commitments.set_commitment(
            netuid=args.netuid,
            info={"fields": fields},
        )
        result = bt.Subtensor(
            network=args.network, policy=bt.Policy(allow_raw_calls=True)
        ).submit_call(call, wallet, signer="hotkey")
    except Exception as exc:
        print(f"error: commitment failed: {exc}", file=sys.stderr)
        return 1
    if not getattr(result, "success", False):
        print(
            f"error: commitment rejected on-chain: "
            f"{getattr(result, 'message', None) or getattr(result, 'error', result)}",
            file=sys.stderr,
        )
        return 1

    print(f"committed patch_hash={patch_hash} on netuid {args.netuid} ({mode})")

    # Verify is informational only: the commitment already landed, so a poll
    # failure must not turn a successful commit into a non-zero exit.
    try:
        status = _await_visible(args.network, args.netuid, hotkey, payload)
    except Exception as exc:
        print(
            f"warning: on-chain verify failed ({exc}); commitment is submitted",
            file=sys.stderr,
        )
    else:
        if status:
            print(f"verified on-chain: {status}")
        else:
            print(
                "warning: commitment not visible within 90s; check chain "
                "propagation before expecting the worker to pick it up",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
