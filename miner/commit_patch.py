#!/usr/bin/env python3
"""Commit a Pareton patch on-chain (Stage 0).

Flow:
  1. Request Pareton-presigned S3 upload (or reuse --retrieval-url)
  2. PUT patch bytes
  3. set_reveal_commitment with v1 patch payload

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

from chain.commitment import encode_patch_commitment  # noqa: E402


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
        campaign = _http_json("GET", f"{args.api_base.rstrip('/')}/v1/campaigns/{args.campaign_id}")
    except Exception as exc:
        print(f"error: failed to fetch campaign: {exc}", file=sys.stderr)
        return 1
    baseline_commit = campaign["baseline_commit"]

    if args.retrieval_url:
        retrieval_url = args.retrieval_url
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

    payload = encode_patch_commitment(
        campaign_id=args.campaign_id,
        baseline_commit=baseline_commit,
        patch_hash=patch_hash,
        retrieval_url=retrieval_url,
    )
    print("commitment payload:")
    print(json.dumps(json.loads(payload), indent=2))

    if args.dry_run:
        print("dry-run: not submitting on-chain")
        return 0

    subtensor = bt.Subtensor(network=args.network)
    if not subtensor.is_hotkey_registered(netuid=args.netuid, hotkey_ss58=hotkey):
        print(
            f"error: hotkey {hotkey} not registered on netuid {args.netuid}",
            file=sys.stderr,
        )
        return 1

    try:
        subtensor.set_reveal_commitment(
            wallet=wallet,
            netuid=args.netuid,
            data=payload,
            blocks_until_reveal=1,
        )
    except Exception as exc:
        print(f"error: commitment failed: {exc}", file=sys.stderr)
        return 1

    print(f"committed patch_hash={patch_hash} on netuid {args.netuid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
