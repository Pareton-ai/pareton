"""Gate b: fetch patch and verify content hash."""

from __future__ import annotations

import hashlib
from typing import Callable

from gate.types import GateResult, SubmissionState
from storage.s3 import fetch_patch_bytes, is_allowed_retrieval_url, patch_url_hotkey


def hash_patch_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _strip_comments(line: bytes, in_block_comment: bool) -> tuple[bytes, bool]:
    """Remove comments outside single-line quoted strings."""
    output = bytearray()
    quote: int | None = None
    escaped = False
    index = 0

    while index < len(line):
        if in_block_comment:
            block_end = line.find(b"*/", index)
            if block_end == -1:
                return bytes(output).rstrip(), True
            in_block_comment = False
            index = block_end + 2
            continue

        byte = line[index]
        if quote is not None:
            output.append(byte)
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == quote:
                quote = None
            index += 1
            continue

        if byte in (ord("'"), ord('"'), ord("`")):
            quote = byte
            output.append(byte)
            index += 1
            continue
        if byte == ord("#") or line[index : index + 2] == b"//":
            break
        if line[index : index + 2] == b"/*":
            in_block_comment = True
            index += 2
            continue

        output.append(byte)
        index += 1

    return bytes(output).rstrip(), in_block_comment


def patch_fingerprint_bytes(data: bytes) -> str:
    """Hash a line-normalized patch for campaign-local copy detection.

    Normalization removes blank lines, trailing whitespace, and ``#``, ``//``,
    and ``/* ... */`` comments outside quoted strings. Git hunk content markers
    (``+``, ``-``, or space) are ignored when classifying a source line, while
    diff metadata remains part of the fingerprint. The policy is deliberately
    language-agnostic. Consequently, a line beginning with ``#`` inside a
    multiline Python docstring is treated as a comment and removed.
    """
    normalized: list[bytes] = []
    in_block_comment = {b"+": False, b"-": False, b" ": False, b"": False}
    is_git_diff = data.startswith(b"diff --git ") or b"\ndiff --git " in data

    for raw_line in data.splitlines():
        line = raw_line.rstrip()
        marker = b""
        content = line
        if line[:1] in (b"+", b"-", b" ") and not line.startswith((b"+++", b"---")):
            marker = line[:1]
            content = line[1:]

        if is_git_diff and not marker:
            if line:
                normalized.append(line)
            continue

        content, in_block_comment[marker] = _strip_comments(
            content, in_block_comment[marker]
        )
        if not content.strip():
            continue
        normalized.append(marker + content)

    payload = b"\n".join(normalized)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


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
