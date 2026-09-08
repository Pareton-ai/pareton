"""Make a void reason's detail safe to publish.

``RoundInfraError`` carries a free-text detail built from whatever failed:
exception strings from provider APIs, local paths, and in at least one path a
workload trace URL. That text is written straight to a public column, so it is
scrubbed here before it lands rather than on the way out. Sanitizing on write
keeps credentials out of the database entirely; the worker still logs the raw
string for internal debugging.

The value after a credential name is delimited by a hand-written scanner rather
than by regex alternatives. A regex has to describe every shape a value can
take, and a shape it fails to describe falls through to a narrower alternative
that keeps part of the credential. The input is a truncated provider body, so
malformed shapes are normal: the 300-character cut in ``gpu/providers/*._req``
lands mid-value, mid-escape, or on a dangling backslash. The scanner has no
failure mode, because every path either finds a delimiter or consumes the rest
of the text, so a shape nobody anticipated over-redacts instead of leaking.

Pure text. No HTTP, no database.
"""

from __future__ import annotations

import re

# Longest detail we keep. Long enough for a provider error plus context, short
# enough that a stack trace cannot turn a round row into a log sink.
MAX_VOID_DETAIL = 500

REDACTED = "[redacted]"

# Work on a bounded prefix so a multi-megabyte stack trace cannot make the
# scans quadratic. Redaction only ever extends to the right, so dropping text
# far past the limit cannot change what survives inside it.
_SCAN_HEADROOM = 8

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")

# A presigned URL carries its signature in the query string, so the query goes
# whole rather than key by key: one unrecognized parameter name would be enough
# to leak the credential this is here to protect.
_URL_QUERY = re.compile(r"(https?://[^\s?]*)\?[^\s]*", re.IGNORECASE)

# The words that make a name a credential name. The surrounding name is found
# by walking outwards from the word rather than by padding this pattern with
# `[\w.-]*` on both sides: that padding is ambiguous, and on a long run of name
# characters with no separator after it the engine retries every split, which
# is cubic. A word alone matches in one pass.
#
# It is a substring search, not `\b`-anchored, because an underscore is a word
# character: `\bACCESS_KEY` never matches inside `AWS_ACCESS_KEY`, which is
# exactly the spelling a provider error uses.
_SENSITIVE_WORD = re.compile(
    r"signature|credential|token|secret|password|passwd"
    r"|api[_-]?key|access[_-]?key|auth(?:orization)?",
    re.IGNORECASE,
)

# Characters that continue a name on either side of the word above.
_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)

# What has to follow the name for it to be a pair: an optional closing quote,
# then `=` or `:`. Provider adapters dump JSON and Python-dict bodies into
# ProvisionError, so the name may be quoted (`"api_key": "secret"`,
# `{'Authorization': 'Bearer secret'}`) and the quotes may themselves be
# encoded to any depth by nesting (`\"api_key\"`, `\\\"api_key\\\"`), hence
# `\\*` rather than a fixed number of backslashes.
_SEPARATOR = re.compile(r"((?:\\*[\"'])?)\s*[=:]\s*")

# An opening quote, at any encoding depth.
_QUOTE = re.compile(r"\\*[\"']")

# `Authorization: Bearer <token>` puts a word between the separator and the
# credential; without this the redaction would cover only "Bearer".
_SCHEME = re.compile(r"(?:bearer|basic|token)[ \t]+", re.IGNORECASE)

# Only these end an unquoted value. Everything else, including raw control
# bytes (flattened to spaces further down), is part of the credential, so an
# injected byte cannot cut the redaction short.
_UNQUOTED_END = " \t\r\n"

# Second pass, independent of the name: credentials that announce their own
# shape are redacted wherever they appear, so an unrecognized (or structurally
# mangled) name is not the only thing standing between a token and the column.
_SECRET_SHAPE = re.compile(
    # An auth scheme plus a credential. The digit is what keeps prose such as
    # "basic authentication failed" out of the redaction.
    r"(?:bearer|basic)[ \t]+(?=[^\s\"']*\d)[^\s\"']{8,}"
    r"|eyJ[\w-]{4,}\.[\w-]+\.[\w-]*"  # JWT
    r"|(?:sk|pk|rk)-[\w-]{8,}"  # OpenAI-style key
    r"|AKIA[0-9A-Z]{12,}"  # AWS access key id
    r"|gh[pousr]_[A-Za-z0-9]{16,}"  # GitHub token
    r"|xox[abprs]-[A-Za-z0-9-]{8,}",  # Slack token
    re.IGNORECASE,
)


def _quoted_value_end(text: str, start: int, quote: str) -> int:
    """Index just past the string whose content starts at ``start``.

    Stops at the first ``quote`` preceded by an even number of backslashes. In
    unnested text that is the value's own closing quote. In text that has been
    JSON-encoded into an enclosing string every quote of the value is escaped,
    so the first unescaped one closes the *enclosing* string and the span still
    covers the whole credential. Over-redacting at depth costs a few trailing
    characters of structure; stopping early would cost the token.

    Returns ``len(text)`` when no such quote exists, the unterminated value
    the provider's 300-character cut leaves behind.
    """
    n = len(text)
    i = start
    backslashes = 0
    while i < n:
        char = text[i]
        if char == "\\":
            backslashes += 1
            i += 1
            continue
        if char == quote and backslashes % 2 == 0:
            return i + 1
        backslashes = 0
        i += 1
    return n


def _value_end(text: str, start: int) -> int:
    """Index just past the credential value beginning at ``start``."""
    quote = _QUOTE.match(text, start)
    if quote:
        return _quoted_value_end(text, quote.end(), quote.group()[-1])
    scheme = _SCHEME.match(text, start)
    i = scheme.end() if scheme else start
    quote = _QUOTE.match(text, i)
    if quote:
        return _quoted_value_end(text, quote.end(), quote.group()[-1])
    n = len(text)
    while i < n and text[i] not in _UNQUOTED_END:
        i += 1
    return i


def _name_start(text: str, word_start: int) -> int:
    """Index of the name, with its opening quote, around a sensitive word."""
    i = word_start
    while i > 0 and text[i - 1] in _NAME_CHARS:
        i -= 1
    if i > 0 and text[i - 1] in "\"'":
        i -= 1
        while i > 0 and text[i - 1] == "\\":
            i -= 1
    return i


def _name_end(text: str, word_end: int) -> int:
    n = len(text)
    i = word_end
    while i < n and text[i] in _NAME_CHARS:
        i += 1
    return i


def _redact_pairs(text: str) -> str:
    """Replace the value of every credential pair, keeping the name."""
    out: list[str] = []
    pos = 0
    for word in _SENSITIVE_WORD.finditer(text):
        if word.start() < pos:
            # Inside a span already redacted as another pair's value.
            continue
        name_end = _name_end(text, word.end())
        separator = _SEPARATOR.match(text, name_end)
        if separator is None:
            continue
        value_end = _value_end(text, separator.end())
        # `completion_tokens=41` is a count, not a credential: no credential
        # name is plural and no credential value is a bare integer. Without
        # this the harness's own stream errors lose the number that matters.
        if (
            text[word.end() : name_end].lower().startswith("s")
            and text[separator.end() : value_end].strip().isdigit()
        ):
            continue
        name_start = max(_name_start(text, word.start()), pos)
        name = text[name_start:name_end] + separator.group(1)
        out.append(text[pos:name_start])
        out.append(f"{name}={REDACTED}")
        pos = value_end
    out.append(text[pos:])
    return "".join(out)


def sanitize_void_detail(detail: str | None, *, limit: int = MAX_VOID_DETAIL) -> str:
    """Scrub a void detail for public display.

    Strips terminal escapes, redacts URL query strings, credential pairs
    (quoted or not, at any encoding depth) and self-describing tokens, then
    flattens to one line and truncates.
    Returns "" for nothing worth showing, so a caller can store NULL.
    """
    if not detail:
        return ""
    text = _ANSI.sub("", str(detail))[: max(limit, 0) * _SCAN_HEADROOM]
    text = _URL_QUERY.sub(rf"\1?{REDACTED}", text)
    text = _redact_pairs(text)
    text = _SECRET_SHAPE.sub(REDACTED, text)
    # After redaction: a raw control byte inside a value must not have been
    # read as the whitespace that ends it.
    text = _CONTROL.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if len(text) > limit:
        # Cut to the limit including the marker, so the column bound holds.
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return text
