"""Whole-file comment stripping for the review diff (Python, C, C++, CUDA)."""

from __future__ import annotations

import ast
import shutil
import subprocess

import pytest

from gate.comment_strip import (
    CHECK_FAILED,
    NO_COMMENTS,
    STRIPPED,
    UNPARSEABLE,
    UNSUPPORTED,
    language_of,
    lex_c_family,
    strip_comments,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("path", "lang"),
    [
        ("python/sglang/srt/layers/attention/triton_ops/decode.py", "python"),
        ("tilelang/kernels/x.pyi", "python"),
        ("csrc/a.c", "c"),
        ("csrc/a.h", "cxx"),
        ("sgl-kernel/csrc/attention/k.cu", "cxx"),
        ("include/flashinfer/attention/k.cuh", "cxx"),
        ("include/cutlass/gemm.hpp", "cxx"),
        ("include/cute/x.inl", "cxx"),
        ("README.md", None),
        ("setup.cfg", None),
    ],
)
def test_language_of(path, lang):
    assert language_of(path) == lang


# -- Python ------------------------------------------------------------------


def test_python_comments_removed_and_lines_kept():
    src = "import torch  # c1\n# c2\nx = torch.bfloat16\n"
    res = strip_comments("a.py", src)
    assert res.status == STRIPPED
    assert res.comments == 2
    assert res.text == "import torch\n\nx = torch.bfloat16\n"


def test_python_hash_inside_strings_is_code():
    # Line 3 starts with "#" but is inside the string that line 2 opens; it
    # closes the string and assigns. A line-level stripper would hide it.
    src = '_doc = """\n# """; KV_DTYPE = "fp4"\ny = "#x"  # real\n'
    res = strip_comments("a.py", src)
    assert res.status == STRIPPED
    assert 'KV_DTYPE = "fp4"' in res.text
    assert 'y = "#x"' in res.text
    assert "real" not in res.text
    assert ast.dump(ast.parse(res.text)) == ast.dump(ast.parse(src))


def test_python_continuation_and_fstrings():
    src = "x = (1 +  # c\n     2)\ny = f'{x}#{x!r}'  # c\nz = 1 + \\\n    2\n"
    res = strip_comments("a.py", src)
    assert res.status == STRIPPED
    assert res.text.count("\n") == src.count("\n")
    assert ast.dump(ast.parse(res.text)) == ast.dump(ast.parse(src))


def test_python_without_comments_is_unchanged():
    res = strip_comments("a.py", "x = 1\n")
    assert (res.status, res.text) == (NO_COMMENTS, "x = 1\n")


def test_python_that_does_not_parse_is_kept_as_is():
    src = "def f(:\n    # c\n"
    res = strip_comments("a.py", src)
    assert (res.status, res.text) == (UNPARSEABLE, src)


def test_python_crlf_and_no_trailing_newline():
    src = "x = 1  # c\r\ny = 2 # d"
    res = strip_comments("a.py", src)
    assert res.status == STRIPPED
    assert "# " not in res.text
    assert res.text.count("\n") == src.count("\n")


def test_unsupported_files_are_kept():
    res = strip_comments("conf.yaml", "# x\na: 1\n")
    assert (res.status, res.text) == (UNSUPPORTED, "# x\na: 1\n")


# -- C, C++, CUDA ------------------------------------------------------------


def test_c_line_and_block_comments():
    src = 'int a = 1; // c\nconst char* s = "// no /* no */"; /* b\n c */ int b;\n'
    res = strip_comments("k.cu", src)
    assert res.status == STRIPPED
    assert res.comments == 2
    assert '"// no /* no */"' in res.text
    assert "int b;" in res.text
    assert res.text.count("\n") == src.count("\n")


def test_c_block_comment_inside_directive_keeps_the_directive_whole():
    src = "#define X 1 /* a\n b */ + 2\nint y = X;\n"
    res = strip_comments("k.cuh", src)
    assert res.status == STRIPPED
    toks, _ = lex_c_family(res.text)
    assert toks[:6] == ["#", "define", "X", "1", "+", "2"]
    assert res.text.count("\n") == src.count("\n")


def test_c_raw_string_header_name_and_char_literals():
    src = (
        "#include <a//b.h>\n"
        'auto r = R"x(/* not */ // no)x"; // c\n'
        "char q = '\"'; int n = 1'000; // c\n"
        'auto u = u8"/*"; // c\n'
    )
    res = strip_comments("k.cpp", src)
    assert res.status == STRIPPED
    assert res.comments == 3
    assert "#include <a//b.h>" in res.text
    assert 'R"x(/* not */ // no)x"' in res.text
    assert "1'000" in res.text
    assert 'u8"/*"' in res.text


def test_c_spliced_line_comment_swallows_the_next_line_like_the_compiler():
    src = "// c \\\nint hidden;\nint shown;\n"
    res = strip_comments("k.cu", src)
    assert res.status == STRIPPED
    assert lex_c_family(res.text)[0] == ["int", "shown", ";"]
    assert res.text.count("\n") == src.count("\n")


def test_c_unterminated_block_comment_is_kept_as_is():
    src = "int a; /* never closed\n"
    res = strip_comments("k.cu", src)
    assert (res.status, res.text) == (UNPARSEABLE, src)


def test_c_statuses_are_known():
    for src in ("int a;\n", "int a; // c\n", "/* x"):
        assert strip_comments("k.c", src).status in (
            STRIPPED,
            NO_COMMENTS,
            UNPARSEABLE,
            CHECK_FAILED,
        )


# A spliced // comment is left out: -fpreprocessed assumes splices are already
# joined, while a real compile joins them first (translation phase 2).
_GCC_CASES = [
    "int a = 1; // c\nint b; /* c */ int c;\n",
    '#define S "a//b" /* c */\nconst char* s = S;\n',
    "#define F(x) \\\n  ((x) /* c */ + 1)\nint y = F(2);\n",
    'auto r = R"d(/* x */)d"; // c\n',
    'const char* k = R"""(\n// kernel source, not a comment\n)"""; // c\n',
    "char q = '/'; int n = 1'000'000; // c\n",
    "a/**/b; c/ /d; e//f\n;\n",
    "#if 0 // c\n#endif /* c */\n",
]


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
@pytest.mark.parametrize("src", _GCC_CASES)
def test_c_matches_gcc_comment_removal(src):
    """g++ -fpreprocessed -E removes comments without expanding macros."""
    ref = subprocess.run(
        ["g++", "-fpreprocessed", "-dD", "-E", "-P", "-x", "c++", "-w", "-"],
        input=src,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    res = strip_comments("k.cu", src)
    assert res.status in (STRIPPED, NO_COMMENTS)
    assert lex_c_family(res.text)[0] == lex_c_family(ref)[0]
