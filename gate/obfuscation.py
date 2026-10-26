"""Gate d1: reject added code that hides what it does.

A deterministic scan of the lines a patch adds. It does not try to prove a
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

import re
import unicodedata
from dataclasses import dataclass

from gate.types import GateResult, SubmissionState

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

_PY_SUFFIXES = (".py", ".pyi", ".pyx", ".pxd")
_C_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".cu",
    ".cuh",
    ".inl",
    ".tcc",
)

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

# An attribute name that is not a plain string literal.
_NON_LITERAL_ATTR = r"\s*,(?!\s*[\"'][\w.]*[\"']\s*[,)])"

_PY_RULES = _rules(
    # Code built or loaded at runtime.
    ("dynamic_code", r"(?<![\w.])(?:exec|eval|compile)\s*\("),
    ("dynamic_code", r"\b__import__\s*\("),
    (
        "dynamic_code",
        r"\bimportlib\s*\.\s*(?:import_module|reload|util\s*\.\s*(?:spec_from|"
        r"module_from))|\bfrom\s+importlib(?:\.util)?\s+import\b",
    ),
    ("dynamic_code", r"\b(?:marshal|pickle|dill|cloudpickle)\s*\.\s*loads?\b"),
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
    ("reflection", r"\.__dict__\s*\["),
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
    ("env_mutation", r"\bos\s*\.\s*environ\s*\[[^\]]+\]\s*=(?!=)"),
    ("env_mutation", r"\bos\s*\.\s*(?:environ\s*\.\s*(?:update|setdefault)|putenv)\b"),
)

_C_RULES = _rules(
    ("dynamic_code", r"\b(?:dlopen|dlsym|popen|execve?|system)\s*\("),
    # Universal character names smuggle identifiers past a reader.
    ("encoded_payload", r"\\[uU][0-9a-fA-F]{4,8}"),
    # Trigraphs.
    ("encoded_payload", r"\?\?[=/'()!<>\-]"),
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
        # Look-alike letters in identifiers or strings (homoglyphs).
        elif cat.startswith("L"):
            rules.append("non_ascii_letter")
    return rules


def scan_patch(patch_text: str) -> list[Finding]:
    findings: list[Finding] = []
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
        if raw[0] == "-":
            continue
        if raw[0] != "+":
            new_line += 1
            continue
        code = raw[1:]
        rules: list[tuple[str, re.Pattern[str]]] = list(_GENERAL_RULES)
        if path.endswith(_PY_SUFFIXES):
            rules += _PY_RULES
        elif path.endswith(_C_SUFFIXES):
            rules += _C_RULES
        hits = {name for name, pattern in rules if pattern.search(code)}
        hits.update(_char_findings(code))
        snippet = code.strip()[:_SNIPPET_CHARS]
        findings.extend(Finding(path, new_line, rule, snippet) for rule in sorted(hits))
        new_line += 1
    return findings


def check_obfuscation(*, patch_bytes: bytes) -> GateResult:
    """Reject the patch if any added line matches an obfuscation rule."""
    try:
        patch_text = patch_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return GateResult.reject("obfuscation_invalid_patch")
    findings = scan_patch(patch_text)
    if findings:
        return GateResult.reject(
            "obfuscation_detected",
            rules=sorted({f.rule for f in findings}),
            findings=[f.as_dict() for f in findings[:_MAX_FINDINGS]],
            finding_count=len(findings),
        )
    return GateResult.success(SubmissionState.SURFACE_OK)
