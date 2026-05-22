#!/usr/bin/env python3
"""Commit a Docker image reference on-chain for Cacheon evaluation.

Usage:
    python miner/commit.py \
        --image docker.io/myuser/cacheon-miner:v1 \
        --digest sha256:abc123... \
        --wallet-name my-miner --wallet-hotkey default \
        --network finney --netuid 14 \
        --fee 0.1

Every submission requires a submission fee sent to PAYMENT_ADDRESS:
  1. Submits a Utility.batch_all extrinsic (signed by the coldkey) containing:
       - Balances.transfer_keep_alive -> payment address, amount=fee_rao
       - System.remark -> b"cacheon:<hotkey_ss58>:<digest>"
  2. Records the resulting tx hash + block hash.
  3. Commits image, digest, payment_tx, and payment_block on-chain.

Environment variable fallback when --fee is omitted:
    CACHEON_SUBMISSION_FEE_RAO  (fee in RAO; 1 TAO = 1_000_000_000 RAO)

If payment succeeded but the commitment step failed, retry with the printed
``--payment-tx`` and ``--payment-block`` flags to avoid paying twice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validator.chain import (  # noqa: E402
    DIGEST_RE,
    extract_payment_fields,
    is_valid_docker_image,
)
from validator.config import PAYMENT_ADDRESS  # noqa: E402

_TAO_TO_RAO = 1_000_000_000
_TAO_TX_BASE = "https://tao.app/tx/"


def _tao_tx_url(tx_hash: str) -> str:
    return f"{_TAO_TX_BASE}{tx_hash}"


def _log_section(title: str) -> None:
    print()
    print(title)
    print("-" * min(len(title), 60))


def _log_kv(key: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key:<14} {value}")


def _resolve_fee(fee_tao: float | None, default_fee_rao: int) -> int:
    """Return fee in RAO or raise ValueError."""
    if fee_tao is not None:
        if fee_tao <= 0:
            raise ValueError(f"--fee must be positive TAO, got: {fee_tao}")
        fee_rao = round(fee_tao * _TAO_TO_RAO)
    elif default_fee_rao > 0:
        fee_rao = default_fee_rao
    else:
        raise ValueError(
            "--fee is required in TAO (e.g. 0.1), or set CACHEON_SUBMISSION_FEE_RAO"
        )

    if fee_rao <= 0:
        raise ValueError("submission fee must be greater than 0")

    return fee_rao


def _parse_payment_reuse(
    payment_tx: str | None,
    payment_block: str | None,
) -> tuple[str, str] | None:
    """Return validated (payment_tx, payment_block) or None if both omitted."""
    if payment_tx is None and payment_block is None:
        return None
    if not payment_tx or not payment_block:
        raise ValueError("--payment-tx and --payment-block must be provided together")
    parsed = extract_payment_fields(
        json.dumps({"payment_tx": payment_tx, "payment_block": payment_block})
    )
    if parsed is None:
        raise ValueError(
            "invalid --payment-tx or --payment-block (expected 0x + 64 hex chars each)"
        )
    return parsed


def _commit_retry_command(args, payment_tx: str, payment_block: str) -> str:
    return (
        "python miner/commit.py \\\n"
        f"  --image {json.dumps(args.image)} \\\n"
        f"  --digest {json.dumps(args.digest)} \\\n"
        f"  --wallet-name {json.dumps(args.wallet_name)} \\\n"
        f"  --wallet-hotkey {json.dumps(args.wallet_hotkey)} \\\n"
        f"  --network {json.dumps(args.network)} \\\n"
        f"  --netuid {args.netuid} \\\n"
        f"  --payment-tx {json.dumps(payment_tx)} \\\n"
        f"  --payment-block {json.dumps(payment_block)}"
    )


def _submit_payment_batch(
    subtensor,
    wallet,
    payment_address: str,
    fee_rao: int,
    hotkey_ss58: str,
    digest: str,
) -> tuple[str, str]:
    """Submit Utility.batch_all([transfer, remark]) signed by the coldkey.

    Returns (extrinsic_hash, block_hash) on success.
    Raises RuntimeError on failure.

    Uses batch_all so both inner calls succeed atomically; if either fails,
    the whole extrinsic is rolled back and receipt.is_success is False.

    The batch contains:
      - Balances.transfer_keep_alive to payment_address for fee_rao RAO.
      - System.remark with b"cacheon:<hotkey_ss58>:<digest>" tying this
        payment to this specific image submission.
    """
    substrate = subtensor.substrate

    remark_bytes = f"cacheon:{hotkey_ss58}:{digest}".encode()

    transfer_call = substrate.compose_call(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={"dest": payment_address, "value": fee_rao},
    )
    remark_call = substrate.compose_call(
        call_module="System",
        call_function="remark",
        call_params={"remark": remark_bytes},
    )
    batch_call = substrate.compose_call(
        call_module="Utility",
        call_function="batch_all",
        call_params={"calls": [transfer_call, remark_call]},
    )

    extrinsic = substrate.create_signed_extrinsic(
        call=batch_call,
        keypair=wallet.coldkey,
    )
    receipt = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    if not receipt.is_success:
        raise RuntimeError(f"payment batch failed on-chain: {receipt.error_message}")

    return str(receipt.extrinsic_hash).lower(), str(receipt.block_hash).lower()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Commit a Docker image reference on-chain (submission fee required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image",
        required=True,
        help="Docker image reference (e.g. docker.io/myuser/cacheon-miner:v1)",
    )
    p.add_argument(
        "--digest",
        required=True,
        help="Image manifest digest (e.g. sha256:abc123...)",
    )
    p.add_argument("--wallet-name", required=True)
    p.add_argument("--wallet-hotkey", default="default")
    p.add_argument("--network", default="test", help="Bittensor network: finney | test")
    p.add_argument("--netuid", type=int, default=14, help="Subnet UID")
    p.add_argument(
        "--fee",
        type=float,
        default=None,
        help="Submission fee in TAO (e.g. 0.1), or CACHEON_SUBMISSION_FEE_RAO in env.",
    )
    p.add_argument(
        "--payment-tx",
        default=None,
        help="Reuse an existing fee payment extrinsic hash (requires --payment-block).",
    )
    p.add_argument(
        "--payment-block",
        default=None,
        help="Block hash where --payment-tx was included (requires --payment-tx).",
    )

    args = p.parse_args(argv)

    try:
        payment_reuse = _parse_payment_reuse(args.payment_tx, args.payment_block)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not is_valid_docker_image(args.image):
        print(f"error: invalid Docker image reference: {args.image}", file=sys.stderr)
        return 1

    if not DIGEST_RE.match(args.digest):
        print(
            f"error: digest must be sha256:<64 hex chars>, got: {args.digest}",
            file=sys.stderr,
        )
        return 1

    default_fee_rao = 0
    raw_fee_rao = os.environ.get("CACHEON_SUBMISSION_FEE_RAO", "").strip()
    if raw_fee_rao:
        try:
            default_fee_rao = int(raw_fee_rao)
        except ValueError:
            print(
                f"error: CACHEON_SUBMISSION_FEE_RAO must be an integer, got: {raw_fee_rao}",
                file=sys.stderr,
            )
            return 1

    try:
        if payment_reuse is None:
            fee_rao = _resolve_fee(args.fee, default_fee_rao)
        else:
            fee_rao = 0
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    import bittensor as bt

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.Subtensor(network=args.network)
    hotkey_ss58: str = wallet.hotkey.ss58_address

    if not subtensor.is_hotkey_registered(netuid=args.netuid, hotkey_ss58=hotkey_ss58):
        print(
            f"error: hotkey {hotkey_ss58} is not registered on netuid {args.netuid} "
            f"({args.network}). Register before paying or committing:\n\n"
            f"  btcli subnet register --netuid {args.netuid} --network {args.network}\n\n"
            f"See https://docs.learnbittensor.org/miners",
            file=sys.stderr,
        )
        return 1

    fee_tao = fee_rao / _TAO_TO_RAO if payment_reuse is None else 0.0
    remark = f"cacheon:{hotkey_ss58}:{args.digest}"

    _log_section("Cacheon miner commit")
    _log_kv("network", args.network)
    _log_kv("netuid", str(args.netuid))
    _log_kv("wallet", f"{args.wallet_name} / {args.wallet_hotkey}")
    _log_kv("hotkey", hotkey_ss58)
    _log_kv("image", args.image)
    _log_kv("digest", args.digest)

    if payment_reuse is None:
        _log_section("Step 1/2  Submission fee")
        _log_kv("amount", f"{fee_tao:.4f} TAO")
        _log_kv("recipient", PAYMENT_ADDRESS)
        _log_kv("remark", remark)
        print("  💳 Coldkey signature required (password prompt may appear).")

        try:
            payment_tx, payment_block = _submit_payment_batch(
                subtensor,
                wallet,
                payment_address=PAYMENT_ADDRESS,
                fee_rao=fee_rao,
                hotkey_ss58=hotkey_ss58,
                digest=args.digest,
            )
        except Exception as e:
            print(f"\nerror: payment failed: {e}", file=sys.stderr)
            return 1

        _log_section("Payment confirmed on-chain")
        print("  ✅ Transfer and remark included in block.")
        _log_kv("extrinsic", payment_tx)
        _log_kv("block", payment_block)
        _log_kv("explorer", _tao_tx_url(payment_tx))
    else:
        payment_tx, payment_block = payment_reuse
        _log_section("Step 1/2  Submission fee (reused)")
        _log_kv("extrinsic", payment_tx)
        _log_kv("block", payment_block)
        _log_kv("explorer", _tao_tx_url(payment_tx))
        print("  Skipping payment; reusing existing on-chain fee transfer.")

    commit_payload = {
        "image": args.image,
        "digest": args.digest,
        "payment_tx": payment_tx,
        "payment_block": payment_block,
    }
    commit_data = json.dumps(commit_payload)

    _log_section("Step 2/2  Reveal commitment")
    _log_kv("netuid", str(args.netuid))
    print("  payload:")
    for line in json.dumps(commit_payload, indent=2).splitlines():
        print(f"    {line}")

    try:
        subtensor.set_reveal_commitment(
            wallet=wallet,
            netuid=args.netuid,
            data=commit_data,
            blocks_until_reveal=1,
        )
    except Exception as e:
        print(f"\nerror: commitment failed: {e}", file=sys.stderr)
        print(
            "\nPayment already submitted. Do not run the script again without "
            "reusing these flags or you will pay twice:\n",
            file=sys.stderr,
        )
        _log_kv("payment_tx", payment_tx, indent=0)
        _log_kv("payment_block", payment_block, indent=0)
        _log_kv("explorer", _tao_tx_url(payment_tx), indent=0)
        print(
            "\nRetry commitment only:\n\n"
            f"{_commit_retry_command(args, payment_tx, payment_block)}\n",
            file=sys.stderr,
        )
        return 1

    _log_section("All set")
    print(f"  ✅ Submitted commitment on netuid {args.netuid}.")
    print("  Validators will verify payment and schedule eval on the next round.")
    print(f"  View payment: {_tao_tx_url(payment_tx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
