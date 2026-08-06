"""Gate b: fetch patch and verify content hash."""

from __future__ import annotations

import hashlib
from typing import Callable

from gate.types import GateResult, SubmissionState
from storage.s3 import fetch_patch_bytes, is_allowed_retrieval_url, patch_url_hotkey


def hash_patch_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def check_integrity(
    *,
    retrieval_url: str,
    expected_patch_hash: str,
    hotkey: str | None = None,
    fetcher: Callable[[str], bytes] = fetch_patch_bytes,
) -> GateResult:
    if not is_allowed_retrieval_url(retrieval_url):
        return GateResult.reject(
            "retrieval_url not allowlisted",
            retrieval_url=retrieval_url,
        )
    if hotkey is not None:
        url_hotkey = patch_url_hotkey(retrieval_url)
        if url_hotkey != hotkey:
            return GateResult.reject(
                "retrieval_url hotkey mismatch",
                retrieval_url=retrieval_url,
                hotkey=hotkey,
                url_hotkey=url_hotkey,
            )
    try:
        data = fetcher(retrieval_url)
    except Exception as exc:
        return GateResult.reject(
            "fetch_failed",
            retrieval_url=retrieval_url,
            error=str(exc),
        )

    actual = hash_patch_bytes(data)
    if actual != expected_patch_hash.lower():
        return GateResult.reject(
            "patch_hash mismatch",
            expected=expected_patch_hash.lower(),
            actual=actual,
            size=len(data),
        )
    return GateResult.success(
        SubmissionState.FETCHED,
        patch_hash=actual,
        size=len(data),
        patch_bytes=data,
    )
