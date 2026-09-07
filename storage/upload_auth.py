"""Canonical, expiring miner authorization for one immutable patch upload."""

import json
import time

import config


def upload_message(fields: dict) -> bytes:
    return b"pareton.patch-upload.v1\n" + json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def verify_upload_request(fields: dict, signature: str) -> int:
    """Verify with the public hotkey only; return remaining authorization seconds.

    Replays address the same upload UUID and checksum. S3 conditional writes
    make those retries idempotent without a nonce table.
    """
    from bittensor.sp_core import verify

    remaining = fields["expires_at"] - int(time.time())
    if not 0 < remaining <= config.UPLOAD_AUTH_TTL_S:
        raise ValueError("upload authorization expired or too far in the future")
    if (
        fields["network"] != config.SUBTENSOR_NETWORK
        or fields["netuid"] != config.NETUID
    ):
        raise ValueError("upload authorization is for a different subnet")
    try:
        valid = verify(
            upload_message(fields), bytes.fromhex(signature), fields["hotkey"]
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("invalid hotkey signature")
    return remaining
