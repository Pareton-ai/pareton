"""Gate d1: reject added code that hides what it does.

Optional per campaign (bench.obfuscation), pinned in the manifest like every
other acceptance rule. A deterministic scan of the lines a patch adds. It does not try to prove a
patch honest; it rejects the constructs that make a patch hard to review, so
that a dtype change has to be written plainly where the LLM review (d2) and a
human can see it. Unlike the LLM review it gives the same answer every time,
so resubmitting a tweaked patch is not a fresh roll; a miner can only get past
it by finding a construct the rules don't cover, which the rules can then add.

Rules apply per file type. Python covers PyTorch, Triton, TileLang and CuTe
DSL code; the C family covers CUDA/C++ (FlashInfer, DeepGEMM, CUTLASS). The
general rules apply to every added line in every file, including templates.

Only added lines are scanned, so existing upstream code that uses these
constructs is not penalised. A legitimate patch that needs one of them has to
be written another way; that trade-off is deliberate.
"""

from __future__ import annotations

import bisect
import io
import re
import tokenize
import unicodedata
from dataclasses import dataclass

from gate.comment_strip import C_SUFFIXES as _C_ONLY
from gate.comment_strip import CXX_SUFFIXES, NO_COMMENTS, STRIPPED, strip_comments
from gate.types import GateResult, SubmissionState

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

_PY_SUFFIXES = (".py", ".pyi", ".pyx", ".pxd")
_C_SUFFIXES = _C_ONLY + CXX_SUFFIXES

# Rule sets a campaign can pin in bench.obfuscation.rules.
RULE_SETS = ("v1",)

_MAX_FINDINGS = 50
_SNIPPET_CHARS = 160


def _rules(*pairs: tuple[str, str]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple((name, re.compile(pattern)) for name, pattern in pairs)


# Every added line, any file.
_GENERAL_RULES = _rules(
    # Long base64 or hex runs are payloads, not code.
    ("encoded_blob", r"[A-Za-z0-9+/]{120,}={0,2}"),
    ("encoded_blob", r"(?:0x)?[0-9a-fA-F]{128,}"),
    (
        "reviewer_directed_text",
        r"(?i)\b(?:ignore|disregard)\s+(?:all\s+|any\s+|the\s+)?"
        r"(?:previous|prior|above|earlier)\s+(?:instructions?|rules?|prompts?)",
    ),
    (
        "reviewer_directed_text",
        r"(?i)\byou\s+are\s+(?:an?\s+)?(?:ai|llm|language\s+model|assistant"
        r"|reviewer|grader|validator)\b",
    ),
    (
        "reviewer_directed_text",
        r"(?i)\b(?:llm|ai|automated|model)[\s_-]+(?:review(?:er)?|grader)\b",
    ),
    ("reviewer_directed_text", r"(?i)\b(?:note\s+to\s+)?(?:reviewer|grader)\s*:"),
    ("reviewer_directed_text", r"(?i)\"verdict\"\s*:"),
    ("reviewer_directed_text", r"(?i)\bprecision[\s-]+neutral\b"),
)

# A fragment of an environment variable name that is about precision, e.g.
# SGLANG_KV_CACHE_DTYPE, NVIDIA_TF32_OVERRIDE, SGLANG_ENABLE_FP8_GEMM.
_DTYPE_ENV = (
    r"(?i:dtype|precision|fp4|fp8|fp16|fp32|bf16|bfloat16|float16|half|tf32"
    r"|quant|e4m3|e5m2|nvfp4|mxfp)"
)

# __import__("os").environ.get( or __import__("os").getenv(, after the "(".
_ENV_READ_IMPORT = (
    r"\s*[rbu]?[\"']os[\"']\s*\)\s*\.\s*(?:environ\s*\.\s*get|getenv)\s*\("
)

# An attribute name that is not a plain string literal.
_NON_LITERAL_ATTR = r"\s*,(?!\s*[\"'][\w.]*[\"']\s*[,)])"

_PY_RULES = _rules(
    # Code built or loaded at runtime.
    ("dynamic_code", r"(?<![\w.])(?<!def )(?:exec|eval|compile)\s*\("),
    # __import__("os").environ.get(...) / .getenv(...) only reads the
    # environment; any other __import__ still matches.
    ("dynamic_code", r"\b__import__\s*\((?!" + _ENV_READ_IMPORT + r")"),
    (
        "dynamic_code",
        # import_module is allowed when its argument starts with a literal
        # module path, as in import_module(f"sglang.srt.layers.{name}").
        r"\bimportlib\s*\.\s*(?:import_module\s*\((?!\s*[rbuf]*[\"'][A-Za-z_][\w.]*)"
        r"|reload|util\s*\.\s*(?:spec_from|module_from))"
        r"|\bfrom\s+importlib(?:\.util)?\s+import\b",
    ),
    # Unpickling a variable is ordinary IPC (scheduler <-> workers); unpickling
    # a literal, a decoded blob, or an alias of loads still matches.
    ("dynamic_code", r"\bmarshal\s*\.\s*loads?\b"),
    (
        "dynamic_code",
        r"\b(?:pickle|dill|cloudpickle)\s*\.\s*loads?\b"
        r"(?!\s*\(\s*[A-Za-z_][\w.]*\s*(?:,[^()]*)?\))",
    ),
    ("dynamic_code", r"\btypes\s*\.\s*(?:CodeType|FunctionType)\b"),
    ("dynamic_code", r"\b(?:import\s+builtins|from\s+builtins\s+import)\b"),
    ("dynamic_code", r"\b__builtins__\b"),
    # Payloads decoded at runtime.
    (
        "encoded_payload",
        r"\b(?:b64decode|b32decode|b16decode|b85decode|a85decode|decodebytes"
        r"|unhexlify|a2b_\w+)\b",
    ),
    ("encoded_payload", r"\b(?:zlib|lzma|bz2|gzip)\s*\.\s*decompress\b"),
    ("encoded_payload", r"\bcodecs\s*\.\s*(?:decode|encode|getdecoder)\b"),
    ("encoded_payload", r"\bbytes\s*\.\s*fromhex\b"),
    ("encoded_payload", r"(?i)['\"]rot.?13['\"]"),
    # Reflection that hides which attribute or module is touched. A plain
    # string literal names the attribute, so getattr(config, "x") is fine.
    (
        "reflection",
        r"\bgetattr\s*\(\s*(?:torch|sys|os|builtins|config)" + _NON_LITERAL_ATTR,
    ),
    (
        "reflection",
        r"\bsetattr\s*\(\s*(?:torch|sys|os|builtins|config|envs|server_args"
        r"|self\s*\.\s*(?:server_args|model_config|config))" + _NON_LITERAL_ATTR,
    ),
    ("reflection", r"\bsetattr\s*\(\s*(?:torch|sys|os|builtins)\b"),
    (
        "reflection",
        r"\b(?:globals|locals)\s*\(\s*\)\s*(?:\[|\.\s*(?:update|setdefault)\b)",
    ),
    ("reflection", r"\bvars\s*\(\s*(?:torch|sys|os|builtins)\b"),
    (
        "reflection",
        r"\bsys\s*\.\s*modules\s*(?:\[|\.\s*(?:update|setdefault|pop)\b)",
    ),
    ("reflection", r"\.__(?:code|globals)__\s*=(?!=)|\.__globals__\b"),
    # A literal key names the attribute, like obj.name = ...; a computed key
    # hides it.
    ("reflection", r"\.__dict__\s*\[(?!\s*[rbu]?[\"'][\w.]*[\"']\s*\])"),
    ("reflection", r"\boperator\s*\.\s*(?:attrgetter|methodcaller)\b"),
    # Native code compiled or loaded at runtime, outside the hermetic build.
    (
        "runtime_native_code",
        r"\b(?:load_inline|cpp_extension|cffi|nvrtc|RawKernel|RawModule)\b",
    ),
    (
        "runtime_native_code",
        r"\bctypes\s*\.\s*(?:CDLL|cdll|PyDLL|pydll|CFUNCTYPE|memmove|string_at)\b",
    ),
    ("runtime_native_code", r"\btorch\s*\.\s*ops\s*\.\s*load_library\b"),
    # Environment flips can select a dtype far from the code that uses it.
    # Only variables whose name is about precision; NCCL, logging and other
    # tuning variables are ordinary performance work.
    (
        "env_mutation",
        r"\bos\s*\.\s*environ\s*\[\s*[rbuf]*[\"'][\w.-]*"
        + _DTYPE_ENV
        + r"[\w.-]*[\"']\s*\]\s*=(?!=)",
    ),
    (
        "env_mutation",
        r"\bos\s*\.\s*(?:environ\s*\.\s*(?:update|setdefault)|putenv)\s*\("
        r"[^)]*?[\"'][\w.-]*" + _DTYPE_ENV,
    ),
    # Assigning into a framework module, e.g. torch.bfloat16 = torch.float8_e4m3fn
    # or tl.float32 = tl.float16, changes every use of it far from this line.
    # Anchored at the start of a statement so annotations (x: torch.Tensor = ...)
    # don't match. torch._dynamo.config and torch._inductor.config are compiler
    # tuning knobs, not numerics; TF32 flags under torch.backends still match.
    (
        "dtype_rebinding",
        r"^\s*(?:torch(?!\s*\.\s*_(?:dynamo|inductor)\s*\.\s*config\b)"
        r"|tl|triton\s*\.\s*language|T|tilelang\s*\.\s*language"
        r"|cute|cutlass|flashinfer|deep_gemm)\s*\.\s*[\w.]+\s*(?:\[[^\]]*\])?\s*"
        r"(?:=(?!=)|[-+*/|&^%@]=)",
    ),
)

_C_RULES = _rules(
    ("dynamic_code", r"\b(?:dlopen|dlsym|popen|execve?|system)\s*\("),
    # Trigraphs.
    ("encoded_payload", r"\?\?[=/'()!<>\-]"),
    # A line splice inside a token (an identifier or number split across
    # lines) reads as two words to a person and as one to the compiler.
    ("token_splice", r"[A-Za-z0-9_]\\[ \t]*$"),
    # Redefining a built-in or CUDA floating-point type name, so that code
    # which says half or float gets a different type.
    (
        "dtype_rebinding",
        r"^\s*#\s*(?:define|undef)\s+(?:half2?|__half2?|float|double|"
        r"(?:__)?nv_bfloat162?|__nv_fp8\w*|__nv_fp4\w*|bfloat16(?:_t)?|half_t|"
        r"float_e\dm\d_t|tfloat32_t|float32_t|float16_t|__float2half\w*|"
        r"__float2bfloat16\w*|__half2float|__bfloat162float)\b",
    ),
    (
        "dtype_rebinding",
        r"\b(?:using\s+(?:half2?|__half2?|(?:__)?nv_bfloat162?|bfloat16_t|half_t"
        r"|float_e\dm\d_t)\s*=|typedef\b[^;]*\b(?:half2?|__half2?|"
        r"(?:__)?nv_bfloat162?|bfloat16_t|half_t|float_e\dm\d_t)\s*;)",
    ),
)

# Universal character names outside string and character literals smuggle
# identifiers past a reader; inside a literal ("µs") they are just text.
_C_UCN_RE = re.compile(r"\\[uU][0-9a-fA-F]{4,8}")
_C_LITERAL_RE = re.compile(r"\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'")
# Python string literals within one line, for the per-line fallback when the
# patched file can't be tokenized.
_PY_LITERAL_RE = re.compile(r"[rRbBuUfF]{0,2}(\"\"\"|'''|\"|')(?:\\.|(?!\1).)*?\1")

# The same rules, for matching across the lines of one statement (a call
# split by a line splice, a comment or a newline). token_splice is about a
# single physical line, so it stays per-line only.
_STATEMENT_RULES = {
    lang: tuple(
        (name, re.compile(p.pattern, p.flags | re.MULTILINE))
        for name, p in rules
        if name != "token_splice"
    )
    for lang, rules in (
        ("py", _GENERAL_RULES + _PY_RULES),
        ("c", _GENERAL_RULES + _C_RULES),
    )
}
# Line splices as each language joins them. GCC, like the comment lexer
# (_CLexer.splice_len), also accepts blanks between the backslash and the
# newline; Python does not.
_SPLICE_RE = {
    "py": re.compile(r"\\\r?\n"),
    "c": re.compile(r"\\[ \t\f\v]*\r?\n"),
}

# Names that run code from data. Checked on the token stream of the patched
# file, so an alias (f = eval), a call split across lines or a reference
# without parentheses is caught too.
_PY_DYNAMIC_NAMES = frozenset(
    {"exec", "eval", "compile", "__import__", "__builtins__", "breakpoint"}
)


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    rule: str
    snippet: str

    def as_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "rule": self.rule,
            "snippet": self.snippet,
        }


def _char_findings(text: str) -> list[str]:
    rules: list[str] = []
    for ch in text:
        if ord(ch) < 128:
            continue
        cat = unicodedata.category(ch)
        # Bidi controls, zero-width characters, BOMs (Trojan Source).
        if cat in ("Cf", "Co", "Cs", "Cn"):
            rules.append("invisible_or_bidi_char")
        # Look-alike letters (homoglyphs). Callers pass only code outside
        # string literals, so prose like "γ=7" in a docstring is not a hit.
        elif cat.startswith("L"):
            rules.append("non_ascii_letter")
    return rules


def _stripped_lines(path: str, text: str | None) -> list[str] | None:
    """The patched file's lines with comments removed, or None when there is
    no file text or stripping it isn't verified safe. Stripping keeps every
    newline, so index i is line i + 1 of the patched file."""
    if text is None:
        return None
    res = strip_comments(path, text)
    if res.status not in (STRIPPED, NO_COMMENTS):
        return None
    return res.text.split("\n")


def _py_tokens(text: str | None) -> list[tokenize.TokenInfo] | None:
    if text is None:
        return None
    try:
        return list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None


def _non_ascii_code(path: str, code: str) -> bool:
    """Whether a line of Python or C has a non-ASCII letter outside string
    literals. Other files (JSON, docs) carry text, not identifiers."""
    if path.endswith(_PY_SUFFIXES):
        code = _PY_LITERAL_RE.sub('""', code)
    elif path.endswith(_C_SUFFIXES):
        code = _C_LITERAL_RE.sub('""', code)
    else:
        return False
    return "non_ascii_letter" in _char_findings(code)


def _is_env_read_import(toks: list[tokenize.TokenInfo], i: int) -> bool:
    """toks[i] is __import__ in __import__("os").environ.get( or .getenv(."""
    rest = [t.string for t in toks[i + 1 : i + 9]]
    if len(rest) < 4 or rest[0] != "(" or rest[2] != ")" or rest[3] != ".":
        return False
    if toks[i + 2].type != tokenize.STRING or toks[i + 2].string.strip(
        "rbuRBU"
    ) not in (
        '"os"',
        "'os'",
    ):
        return False
    return rest[4:6] == ["getenv", "("] or rest[4:8] == ["environ", ".", "get", "("]


def _python_token_findings(
    path: str, toks: list[tokenize.TokenInfo], added: dict[int, str]
) -> list[Finding]:
    """On added lines: bare references to the builtins in _PY_DYNAMIC_NAMES,
    and identifiers with non-ASCII letters (homoglyphs)."""
    significant = [
        t
        for t in toks
        if t.type
        not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)
    ]
    out: list[Finding] = []
    for i, tok in enumerate(significant):
        if tok.type != tokenize.NAME or tok.start[0] not in added:
            continue
        if "non_ascii_letter" in _char_findings(tok.string):
            out.append(
                Finding(
                    path,
                    tok.start[0],
                    "non_ascii_letter",
                    added[tok.start[0]].strip()[:_SNIPPET_CHARS],
                )
            )
        if tok.string not in _PY_DYNAMIC_NAMES:
            continue
        prev = significant[i - 1].string if i else ""
        nxt = significant[i + 1].string if i + 1 < len(significant) else ""
        # model.eval(), torch.compile(...), def eval(self), eval=False as a
        # keyword argument or parameter.
        if prev in (".", "def", "class"):
            continue
        if nxt == "=" and prev in ("(", ","):
            continue
        if tok.string == "__import__" and _is_env_read_import(significant, i):
            continue
        out.append(
            Finding(
                path,
                tok.start[0],
                "dynamic_code",
                added[tok.start[0]].strip()[:_SNIPPET_CHARS],
            )
        )
    return out


def _statement_findings(
    path: str, text: str, added: dict[int, str], lang: str
) -> list[Finding]:
    """Rule matches over the whole patched file that touch an added line.

    Comments are stripped (when that is verified safe; stripping keeps every
    newline) and line splices are joined as the compiler joins them, so a
    match can span the lines of a statement. Offsets are mapped back to line
    numbers in the patched file.
    """
    stripped = strip_comments(path, text)
    if stripped.status in (STRIPPED, NO_COMMENTS):
        text = stripped.text
    parts = _SPLICE_RE[lang].split(text)
    # Offset in the joined text at which each splice was removed.
    splices: list[int] = []
    pos = 0
    for part in parts[:-1]:
        pos += len(part)
        splices.append(pos)
    text = "".join(parts)
    newlines = [m.start() for m in re.finditer("\n", text)]

    def line_of(offset: int) -> int:
        return (
            1
            + bisect.bisect_left(newlines, offset)
            + bisect.bisect_right(splices, offset)
        )

    out: list[Finding] = []
    for name, pattern in _STATEMENT_RULES[lang]:
        for m in pattern.finditer(text):
            # A leading \s* can reach back over blank lines; start at the
            # first character of the match that is not whitespace.
            start = m.start() + len(m.group()) - len(m.group().lstrip())
            first = line_of(start)
            last = line_of(max(start, m.end() - 1))
            for line_no in range(first, last + 1):
                if line_no in added:
                    snippet = added[line_no].strip()[:_SNIPPET_CHARS]
                    out.append(Finding(path, line_no, name, snippet))
                    break
    return out


def added_lines_from_patch(patch_text: str) -> dict[str, dict[int, str]]:
    """Added lines per file and new line number, parsed from a git diff.

    The worker does not use this: it takes added lines from the source view,
    which git computed from the files it actually applied. This parser is for
    tests and tools that have only a patch.
    """
    added: dict[str, dict[int, str]] = {}
    path = ""
    new_line = 0
    in_hunk = False
    for raw in patch_text.splitlines():
        m = _DIFF_GIT_RE.match(raw)
        if m:
            path, in_hunk = m.group(2), False
            continue
        h = _HUNK_RE.match(raw)
        if h:
            new_line, in_hunk = int(h.group(1)), True
            continue
        if not in_hunk or not raw:
            continue
        if raw[0] in "-\\":
            # Removed lines, and "\ No newline at end of file", which is not
            # a line of either side.
            continue
        if raw[0] != "+":
            new_line += 1
            continue
        added.setdefault(path, {})[new_line] = raw[1:]
        new_line += 1
    return added


def scan_added(
    added: dict[str, dict[int, str]], post_texts: dict[str, str] | None = None
) -> list[Finding]:
    """Scan added lines; ``post_texts`` (patched file contents) enables the
    checks that span lines: the rules matched over whole statements, and the
    token-level Python checks.

    With the patched file at hand, the rules read each added line with its
    comments stripped, the same code the LLM review sees, so a comment that
    mentions ``compile`` or uses a Greek letter is not a finding. The
    invisible/bidi character check always reads the raw line.
    """
    post_texts = post_texts or {}
    findings: list[Finding] = []
    for path, lines in added.items():
        rules: list[tuple[str, re.Pattern[str]]] = list(_GENERAL_RULES)
        if path.endswith(_PY_SUFFIXES):
            rules += _PY_RULES
        elif path.endswith(_C_SUFFIXES):
            rules += _C_RULES
        stripped = _stripped_lines(path, post_texts.get(path))
        # When the patched Python file tokenizes, its identifiers are checked
        # on the token stream below; a docstring's inner lines carry no quotes
        # for the per-line check to see.
        py_toks = (
            _py_tokens(post_texts.get(path)) if path.endswith(_PY_SUFFIXES) else None
        )
        for line_no, raw in lines.items():
            code = raw
            if stripped is not None and line_no <= len(stripped):
                code = stripped[line_no - 1]
            hits = {name for name, pattern in rules if pattern.search(code)}
            if path.endswith(_C_SUFFIXES) and _C_UCN_RE.search(
                _C_LITERAL_RE.sub('""', code)
            ):
                hits.add("encoded_payload")
            if py_toks is None and _non_ascii_code(path, code):
                hits.add("non_ascii_letter")
            hits.update(r for r in _char_findings(raw) if r != "non_ascii_letter")
            snippet = raw.strip()[:_SNIPPET_CHARS]
            findings.extend(
                Finding(path, line_no, rule, snippet) for rule in sorted(hits)
            )
    if post_texts:
        seen = {(f.file, f.line, f.rule) for f in findings}
        for path, lines in added.items():
            text = post_texts.get(path)
            if text is None:
                continue
            if path.endswith(_PY_SUFFIXES):
                extra = _statement_findings(path, text, lines, "py")
                toks = _py_tokens(text)
                if toks is not None:
                    extra += _python_token_findings(path, toks, lines)
            elif path.endswith(_C_SUFFIXES):
                extra = _statement_findings(path, text, lines, "c")
            else:
                continue
            for f in extra:
                if (f.file, f.line, f.rule) not in seen:
                    seen.add((f.file, f.line, f.rule))
                    findings.append(f)
    findings.sort(key=lambda f: (f.file, f.line, f.rule))
    return findings


def scan_patch(
    patch_text: str, post_texts: dict[str, str] | None = None
) -> list[Finding]:
    return scan_added(added_lines_from_patch(patch_text), post_texts)


def validate_obfuscation(spec: dict | None) -> dict:
    """Check bench.obfuscation, so a bad value fails at seed, not in the gate.

    Returns {} when the scan is off; campaigns without it keep their hashes.
    ``rules`` names the rule set, so a later change to the rules is a new
    value a campaign opts into rather than a change under a live campaign.
    """
    if not spec:
        return {}
    if not isinstance(spec, dict):
        raise ValueError("obfuscation must be an object")
    unknown = set(spec) - {"rules"}
    if unknown:
        raise ValueError(f"unknown obfuscation keys: {sorted(unknown)}")
    if spec.get("rules") not in RULE_SETS:
        raise ValueError(f"obfuscation.rules must be one of {list(RULE_SETS)}")
    return {"rules": spec["rules"]}


def check_obfuscation(
    *,
    patch_bytes: bytes,
    post_texts: dict[str, str] | None = None,
    added_lines: dict[str, dict[int, str]] | None = None,
) -> GateResult:
    """Reject the patch if any added line matches an obfuscation rule.

    Mandatory for every campaign. ``post_texts`` maps each touched path to its
    patched contents (from the source view) for the token-level checks.
    ``added_lines`` (also from the source view) are the lines git added, per
    file; without them the patch text is parsed instead.
    """
    try:
        patch_text = patch_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return GateResult.reject("obfuscation_invalid_patch")
    if added_lines is None:
        added_lines = added_lines_from_patch(patch_text)
    findings = scan_added(added_lines, post_texts)
    if findings:
        return GateResult.reject(
            "obfuscation_detected",
            rules=sorted({f.rule for f in findings}),
            findings=[f.as_dict() for f in findings[:_MAX_FINDINGS]],
            finding_count=len(findings),
        )
    return GateResult.success(SubmissionState.SURFACE_OK)
