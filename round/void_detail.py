"""Make a void reason's detail safe to publish.

``RoundInfraError`` carries a free-text detail built from whatever failed:
exception strings from provider APIs, local paths, and in at least one path a
workload trace URL. That text is written straight to a public column, so it is
scrubbed here before it lands rather than on the way out. Sanitizing on write
keeps credentials out of the database entirely; the worker still logs the raw
string for internal debugging.

Pure text. No HTTP, no database.
"""

from __future__ import annotations

import re

# Longest detail we keep. Long enough for a provider error plus context, short
# enough that a stack trace cannot turn a round row into a log sink.
MAX_VOID_DETAIL = 500

REDACTED = "[redacted]"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")

# A presigned URL carries its signature in the query string, so the query goes
# whole rather than key by key: one unrecognized parameter name would be enough
# to leak the credential this is here to protect.
_URL_QUERY = re.compile(r"(https?://[^\s?]*)\?[^\s]*", re.IGNORECASE)

# Bare `key=value` outside a URL, for the same secrets arriving loose in an
# exception string. Bounded to the value's own token so surrounding prose
# survives.
#
# The name is wrapped in `[\w.-]*` rather than `\b` because an underscore is a
# word character: `\bACCESS_KEY` never matches inside `AWS_ACCESS_KEY`, which
# is exactly the spelling a provider error uses. The optional scheme keeps
# `Authorization: Bearer <token>` from redacting only the word "Bearer".
_SENSITIVE_PAIR = re.compile(
    r"([\w.-]*(?:signature|credential|token|secret|password|passwd|api[_-]?key"
    r"|access[_-]?key|auth(?:orization)?)[\w.-]*)"
    r"\s*[=:]\s*"
    r"(?:(?:bearer|basic|token)\s+)?\S+",
    re.IGNORECASE,
)


def sanitize_void_detail(detail: str | None, *, limit: int = MAX_VOID_DETAIL) -> str:
    """Scrub a void detail for public display.

    Strips terminal escapes and control characters, flattens the string to one
    line, redacts URL query strings and loose credential pairs, then truncates.
    Returns "" for nothing worth showing, so a caller can store NULL.
    """
    if not detail:
        return ""
    text = _ANSI.sub("", str(detail))
    text = _CONTROL.sub(" ", text)
    text = _URL_QUERY.sub(rf"\1?{REDACTED}", text)
    text = _SENSITIVE_PAIR.sub(rf"\1={REDACTED}", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if len(text) > limit:
        # Cut to the limit including the marker, so the column bound holds.
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return text
