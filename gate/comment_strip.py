"""Remove comments from whole source files without changing what the code does.

The LLM review (d2) reads comment-free code, so a patch cannot talk to the
reviewer through comments. Stripping must never hide or invent code, so it
works on whole files (never diff fragments) with a real lexer per language,
and every result is checked before it is used:

- Python (PyTorch, Triton, TileLang, CuTe DSL): comments are the COMMENT
  tokens from ``tokenize``. The result must parse to exactly the same AST as
  the original under this interpreter, or the file is left as it is.
- C, C++ and CUDA (FlashInfer, DeepGEMM, CUTLASS, sgl-kernel): a lexer that
  follows translation phases 1-3: line splices, ``//`` and ``/* */``
  comments, string and character literals with encoding prefixes, C++ raw
  strings, ``#include <...>`` header names and digit separators. The result
  must lex to exactly the same token sequence with no comments left, or the
  file is left as it is.

Any other file type, a file that does not tokenize or parse, or a file that
fails its check is returned unchanged. Leaving comments in is always safe;
removing something that is not a comment is not.

Line numbers are preserved: a removed comment leaves its newlines behind, so
hunk headers in the review diff point at real lines of the patched file.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass

PY_SUFFIXES = (".py", ".pyi")
C_SUFFIXES = (".c", ".h")
CXX_SUFFIXES = (
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".hh",
    ".hpp",
    ".hxx",
    ".h++",
    ".cu",
    ".cuh",
    ".inl",
    ".ipp",
    ".tcc",
    ".tpp",
)

STRIPPED = "stripped"  # comments removed and the result verified
NO_COMMENTS = "no_comments"  # nothing to remove
UNSUPPORTED = "unsupported"  # not a language this module lexes
UNPARSEABLE = "unparseable"  # the original did not tokenize/parse; kept as is
CHECK_FAILED = "check_failed"  # the verification failed; kept as is


@dataclass(frozen=True)
class StripResult:
    text: str
    status: str
    comments: int = 0


def language_of(path: str) -> str | None:
    lower = path.lower()
    if lower.endswith(PY_SUFFIXES):
        return "python"
    if lower.endswith(C_SUFFIXES):
        # .h is shared by C and C++/CUDA projects; lexing it as C++ only
        # differs on raw strings, which a C header cannot contain.
        return "cxx" if lower.endswith(".h") else "c"
    if lower.endswith(CXX_SUFFIXES):
        return "cxx"
    return None


def strip_comments(path: str, text: str) -> StripResult:
    """Return ``text`` without comments, or unchanged when that isn't provably safe."""
    lang = language_of(path)
    if lang == "python":
        return _strip_python(text)
    if lang in ("c", "cxx"):
        return _strip_c_family(text, cxx=lang == "cxx")
    return StripResult(text, UNSUPPORTED)


# -- Python ------------------------------------------------------------------


def _ast_dump(text: str) -> str:
    return ast.dump(ast.parse(text))


def _strip_python(text: str) -> StripResult:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        original_ast = _ast_dump(text)
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError):
        return StripResult(text, UNPARSEABLE)
    comments = [t for t in tokens if t.type == tokenize.COMMENT]
    if not comments:
        return StripResult(text, NO_COMMENTS)
    # Split exactly as tokenize did (str.splitlines also breaks on \f, \x1c,
    # \u2028 and others, which would shift the rows).
    lines = io.StringIO(text).readlines()
    for tok in comments:
        (row, col), (end_row, end_col) = tok.start, tok.end
        if row != end_row or row > len(lines):
            return StripResult(text, CHECK_FAILED, len(comments))
        line = lines[row - 1]
        # A comment runs to the end of its physical line; keep the newline.
        lines[row - 1] = line[:col].rstrip(" \t\f") + line[end_col:]
    stripped = "".join(lines)
    try:
        same = _ast_dump(stripped) == original_ast
        tokenize_ok = not any(
            t.type == tokenize.COMMENT
            for t in tokenize.generate_tokens(io.StringIO(stripped).readline)
        )
    except (SyntaxError, ValueError, tokenize.TokenError, RecursionError):
        same = tokenize_ok = False
    if not (same and tokenize_ok):
        return StripResult(text, CHECK_FAILED, len(comments))
    return StripResult(stripped, STRIPPED, len(comments))


# -- C, C++, CUDA ------------------------------------------------------------

_IDENT_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$")
_IDENT_CHARS = _IDENT_START | frozenset("0123456789")
_DIGITS = frozenset("0123456789")
_HSPACE = frozenset(" \t\f\v\r")
_STRING_PREFIXES = frozenset(("u8", "u", "U", "L"))
_RAW_PREFIXES = frozenset(("R", "u8R", "uR", "UR", "LR"))
_HEADER_DIRECTIVES = frozenset(("include", "include_next", "import"))


class _LexError(Exception):
    pass


@dataclass
class _Lexed:
    out: str
    tokens: list[str]
    comments: int


def _is_ident_char(ch: str) -> bool:
    return ch in _IDENT_CHARS or ord(ch) > 127


class _CLexer:
    """Phases 1-3 of C/C++ translation, enough to find comments exactly."""

    def __init__(self, src: str, *, cxx: bool) -> None:
        self.s = src
        self.n = len(src)
        self.cxx = cxx

    def splice_len(self, j: int) -> int:
        """Length of a line splice at j (backslash, optional blanks, newline)."""
        s = self.s
        if j >= self.n or s[j] != "\\":
            return 0
        k = j + 1
        # GCC accepts (and warns about) blanks between the backslash and newline.
        while k < self.n and s[k] in " \t\f\v":
            k += 1
        if k < self.n and s[k] == "\r":
            k += 1
        if k < self.n and s[k] == "\n":
            return k + 1 - j
        return 0

    def skip_splices(self, j: int) -> int:
        while True:
            k = self.splice_len(j)
            if not k:
                return j
            j += k

    def char_at(self, j: int) -> tuple[str, int]:
        """The logical character at j (after splices) and the index after it."""
        j = self.skip_splices(j)
        if j >= self.n:
            return "", j
        return self.s[j], j + 1

    def lex(self, *, strip: bool) -> _Lexed:
        s, n = self.s, self.n
        out: list[str] = []
        tokens: list[str] = []
        comments = 0
        i = 0
        line_start = True  # only whitespace/comments since the last newline
        in_directive = False
        directive_name_pending = False
        header_name_allowed = False
        while i < n:
            k = self.splice_len(i)
            if k:
                out.append(s[i : i + k])
                i += k
                continue
            ch = s[i]
            if ch == "\n":
                out.append(ch)
                i += 1
                line_start = True
                in_directive = directive_name_pending = header_name_allowed = False
                continue
            if ch in _HSPACE:
                out.append(ch)
                i += 1
                continue
            if ch == "/":
                nxt, after = self.char_at(i + 1)
                if nxt == "/":
                    end = self._line_comment_end(after)
                    comments += 1
                    span = s[i:end]
                    if strip:
                        while out and out[-1] in (" ", "\t", "\f", "\v"):
                            out.pop()
                        # The comment ran to the end of the logical line, so
                        # its spliced newlines can become plain newlines.
                        out.append("\n" * span.count("\n"))
                    else:
                        out.append(span)
                    i = end
                    continue
                if nxt == "*":
                    end = self._block_comment_end(after)
                    comments += 1
                    span = s[i:end]
                    if strip:
                        out.append(self._block_replacement(span, end, in_directive))
                    else:
                        out.append(span)
                    i = end
                    continue
            if ch == "#" and line_start:
                in_directive = directive_name_pending = True
                line_start = False
                out.append(ch)
                tokens.append(ch)
                i += 1
                continue
            if ch == "<" and header_name_allowed:
                end = self._header_name_end(i)
                tok = s[i:end]
                out.append(tok)
                tokens.append(tok)
                header_name_allowed = False
                i = end
                continue
            line_start = False
            header_name_allowed = False
            if ch in _IDENT_START or ord(ch) > 127:
                end = self._ident_end(i)
                ident = _unsplice(s[i:end])
                q, _ = self.char_at(end)
                if self.cxx and ident in _RAW_PREFIXES and q == '"':
                    end = self._raw_string_end(self.skip_splices(end))
                elif ident in _STRING_PREFIXES and q in ('"', "'"):
                    end = self._quoted_end(self.skip_splices(end))
                elif directive_name_pending:
                    header_name_allowed = ident in _HEADER_DIRECTIVES
                directive_name_pending = False
                tok = s[i:end]
                out.append(tok)
                tokens.append(_unsplice(tok))
                i = end
                continue
            directive_name_pending = False
            if ch in _DIGITS or (ch == "." and self.char_at(i + 1)[0] in _DIGITS):
                end = self._pp_number_end(i)
            elif ch in ('"', "'"):
                end = self._quoted_end(i)
            else:
                end = i + 1
            tok = s[i:end]
            out.append(tok)
            tokens.append(_unsplice(tok))
            i = end
        return _Lexed("".join(out), tokens, comments)

    def _line_comment_end(self, j: int) -> int:
        """End of a // comment: the first newline that is not part of a splice."""
        while j < self.n:
            k = self.splice_len(j)
            if k:
                j += k
                continue
            if self.s[j] == "\n":
                return j
            j += 1
        return j

    def _block_comment_end(self, j: int) -> int:
        while True:
            ch, j = self.char_at(j)
            if not ch:
                raise _LexError("unterminated block comment")
            if ch == "*":
                nxt, after = self.char_at(j)
                if nxt == "/":
                    return after

    def _block_replacement(self, span: str, end: int, in_directive: bool) -> str:
        newlines = span.count("\n")
        if not newlines:
            return " "
        # A block comment is one space. Outside a directive a newline is just
        # whitespace too, unless it would put a '#' at the start of a line and
        # make a directive out of it. Otherwise keep the logical line intact
        # with splices.
        eol = self.s.find("\n", end)
        rest = self.s[end : eol if eol >= 0 else self.n].lstrip(" \t\f\v\r")
        if not in_directive and not rest.startswith(("#", "%:")):
            return " " + "\n" * newlines
        return " " + "\\\n" * newlines

    def _header_name_end(self, j: int) -> int:
        k = j + 1
        while k < self.n:
            ch = self.s[k]
            if ch == ">":
                return k + 1
            if ch == "\n":
                break
            k += 1
        # Not a header name after all: just the '<' punctuator.
        return j + 1

    def _ident_end(self, j: int) -> int:
        while True:
            k = self.skip_splices(j)
            if k < self.n and _is_ident_char(self.s[k]):
                j = k + 1
            else:
                return j

    def _pp_number_end(self, j: int) -> int:
        """A preprocessing number, including C++14/C23 digit separators."""
        _, j = self.char_at(j)
        prev = ""
        while True:
            ch, after = self.char_at(j)
            if not ch:
                return j
            if ch in "+-" and prev in ("e", "E", "p", "P"):
                pass
            elif ch == "'":
                nxt, _ = self.char_at(after)
                if not nxt or not _is_ident_char(nxt):
                    return j
            elif not (_is_ident_char(ch) or ch == "."):
                return j
            prev = ch
            j = after

    def _quoted_end(self, j: int) -> int:
        """End of a string or character literal starting at the quote at j.

        An unterminated literal ends at the newline, as in GCC: the rest of the
        line is part of the bad token, never a comment.
        """
        quote = self.s[j]
        j += 1
        while True:
            ch, after = self.char_at(j)
            if not ch:
                return self.n
            if ch == "\n":
                return self.skip_splices(j)
            if ch == "\\":
                esc, after2 = self.char_at(after)
                if not esc or esc == "\n":
                    return after
                j = after2
                continue
            j = after
            if ch == quote:
                return j

    def _raw_string_end(self, j: int) -> int:
        """End of R"delim( ... )delim" starting at the quote at j.

        Splices are not processed inside raw strings, so this works on raw text.
        """
        s = self.s
        open_paren = s.find("(", j + 1, j + 18)
        if open_paren < 0:
            raise _LexError("bad raw string delimiter")
        delim = s[j + 1 : open_paren]
        # d-chars: anything but space, parentheses, backslash and whitespace
        # controls; '"' is allowed (CUTLASS uses R"""( ... )""").
        if any(c in delim for c in " ()\\\t\v\f\n"):
            raise _LexError("bad raw string delimiter")
        close = s.find(")" + delim + '"', open_paren + 1)
        if close < 0:
            raise _LexError("unterminated raw string")
        return close + len(delim) + 2


def _unsplice(tok: str) -> str:
    if "\\" not in tok or "\n" not in tok:
        return tok
    out: list[str] = []
    i = 0
    lexer = _CLexer(tok, cxx=True)
    while i < len(tok):
        k = lexer.splice_len(i)
        if k:
            i += k
            continue
        out.append(tok[i])
        i += 1
    return "".join(out)


def lex_c_family(text: str, *, cxx: bool = True) -> tuple[list[str], int]:
    """Tokens (without whitespace or comments) and the comment count."""
    lexed = _CLexer(text, cxx=cxx).lex(strip=False)
    return lexed.tokens, lexed.comments


def _strip_c_family(text: str, *, cxx: bool) -> StripResult:
    try:
        original = _CLexer(text, cxx=cxx).lex(strip=False)
    except _LexError:
        return StripResult(text, UNPARSEABLE)
    if original.out != text:
        return StripResult(text, CHECK_FAILED, original.comments)
    if not original.comments:
        return StripResult(text, NO_COMMENTS)
    try:
        stripped = _CLexer(text, cxx=cxx).lex(strip=True)
        relexed = _CLexer(stripped.out, cxx=cxx).lex(strip=False)
    except _LexError:
        return StripResult(text, CHECK_FAILED, original.comments)
    if (
        relexed.comments
        or relexed.tokens != original.tokens
        or stripped.out.count("\n") != text.count("\n")
    ):
        return StripResult(text, CHECK_FAILED, original.comments)
    return StripResult(stripped.out, STRIPPED, original.comments)
