"""Gate b: fetch patch and verify content hash."""

from __future__ import annotations

import hashlib
from typing import Callable

from gate.types import GateResult, SubmissionState
from storage.s3 import fetch_patch_bytes, is_allowed_retrieval_url, patch_url_hotkey


def hash_patch_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


PATCH_HASH_MISMATCH = "patch_hash mismatch"

_NO_COMMENTS = (False, False)
_PYTHON_COMMENTS = (True, False)
_C_FAMILY_COMMENTS = (False, True)


def _comment_syntax_for_path(path: bytes) -> tuple[bool, bool]:
    normalized = path.split(b"\t", 1)[0].strip().strip(b'"')
    if normalized.startswith((b"a/", b"b/")):
        normalized = normalized[2:]
    path_lower = normalized.lower()
    if path_lower.endswith((b".py", b".pyi")):
        return _PYTHON_COMMENTS
    if path_lower.endswith((b".c", b".cc", b".cpp", b".h", b".hpp", b".cu", b".cuh")):
        return _C_FAMILY_COMMENTS
    return _NO_COMMENTS


def _strip_comments(
    line: bytes,
    in_block_comment: bool,
    syntax: tuple[bool, bool],
) -> tuple[bytes, bool]:
    """Remove comments outside single-line quoted strings."""
    hash_comments, slash_comments = syntax
    output = bytearray()
    quote: int | None = None
    escaped = False
    index = 0

    while index < len(line):
        if slash_comments and in_block_comment:
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
        if hash_comments and byte == ord("#"):
            break
        if slash_comments and line[index : index + 2] == b"//":
            break
        if slash_comments and line[index : index + 2] == b"/*":
            in_block_comment = True
            index += 2
            continue

        output.append(byte)
        index += 1

    return bytes(output).rstrip(), in_block_comment


def patch_fingerprint_bytes(data: bytes) -> str:
    """Hash a line-normalized patch for campaign-local copy detection.

    Normalization removes blank lines, trailing whitespace, and comments in
    Python and C-family files, in any directory. Python floor division and
    C-family preprocessor directives remain code. Git hunk content markers
    (``+``, ``-``, or space) are ignored when classifying a source line. Git
    blob hashes and hunk coordinates are omitted because comment-only changes
    alter them. A line beginning with ``#`` inside a multiline Python docstring
    is still treated as a comment and removed.
    """
    normalized: list[bytes] = []
    is_git_diff = data.startswith(b"diff --git ") or b"\ndiff --git " in data
    old_in_block_comment = False
    new_in_block_comment = False
    old_syntax = _NO_COMMENTS
    new_syntax = _NO_COMMENTS
    in_hunk = False

    for raw_line in data.splitlines():
        line = raw_line.rstrip()
        if not is_git_diff:
            if line.strip():
                normalized.append(line)
            continue

        if line.startswith(b"diff --git "):
            old_in_block_comment = False
            new_in_block_comment = False
            old_syntax = _NO_COMMENTS
            new_syntax = _NO_COMMENTS
            in_hunk = False
            normalized.append(line)
            continue
        if line.startswith(b"index "):
            continue
        if not in_hunk and line.startswith(b"--- "):
            old_syntax = _comment_syntax_for_path(line[4:])
        if not in_hunk and line.startswith(b"+++ "):
            new_syntax = _comment_syntax_for_path(line[4:])
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
                content, old_in_block_comment, old_syntax
            )
            if content.strip():
                normalized.append(marker + content)
            continue
        if marker == b"+":
            content, new_in_block_comment = _strip_comments(
                content, new_in_block_comment, new_syntax
            )
            if content.strip():
                normalized.append(marker + content)
            continue

        old_content, old_in_block_comment = _strip_comments(
            content, old_in_block_comment, old_syntax
        )
        new_content, new_in_block_comment = _strip_comments(
            content, new_in_block_comment, new_syntax
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
            PATCH_HASH_MISMATCH,
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
