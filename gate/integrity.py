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
    (``+``, ``-``, or space) are ignored when classifying a source line. Git
    blob hashes and hunk coordinates are omitted because comment-only changes
    alter them. The policy is deliberately language-agnostic. Consequently, a
    line beginning with ``#`` inside a multiline Python docstring is treated as
    a comment and removed.
    """
    normalized: list[bytes] = []
    is_git_diff = data.startswith(b"diff --git ") or b"\ndiff --git " in data
    old_in_block_comment = False
    new_in_block_comment = False
    plain_in_block_comment = False
    in_hunk = False

    for raw_line in data.splitlines():
        line = raw_line.rstrip()
        if not is_git_diff:
            content, plain_in_block_comment = _strip_comments(
                line, plain_in_block_comment
            )
            if content.strip():
                normalized.append(content)
            continue

        if line.startswith(b"diff --git "):
            old_in_block_comment = False
            new_in_block_comment = False
            in_hunk = False
            normalized.append(line)
            continue
        if line.startswith(b"index "):
            continue
        if line.startswith(b"@@"):
            in_hunk = True
            header_end = line.find(b"@@", 2)
            suffix = line[header_end + 2 :].strip() if header_end != -1 else b""
            normalized.append(b"@@" + (b" " + suffix if suffix else b""))
            continue
        if not in_hunk or line[:1] not in (b"+", b"-", b" "):
            if line:
                normalized.append(line)
            continue

        marker = line[:1]
        content = line[1:]
        if marker == b"-":
            content, old_in_block_comment = _strip_comments(
                content, old_in_block_comment
            )
            if content.strip():
                normalized.append(marker + content)
            continue
        if marker == b"+":
            content, new_in_block_comment = _strip_comments(
                content, new_in_block_comment
            )
            if content.strip():
                normalized.append(marker + content)
            continue

        old_content, old_in_block_comment = _strip_comments(
            content, old_in_block_comment
        )
        new_content, new_in_block_comment = _strip_comments(
            content, new_in_block_comment
        )
        if old_content == new_content:
            if old_content.strip():
                normalized.append(marker + old_content)
            continue
        if old_content.strip():
            normalized.append(b"< " + old_content)
        if new_content.strip():
            normalized.append(b"> " + new_content)

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
