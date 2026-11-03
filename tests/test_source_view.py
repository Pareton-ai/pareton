"""Review diff built from comment-stripped baseline and patched files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gate.source_view import SourceViewError, build_source_view

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "baseline"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "kv.py").write_text(
        "# KV cache\nKV_DTYPE = 'fp8'  # default\n\n\ndef f():\n    return 1\n"
    )
    (repo / "pkg" / "k.cu").write_text("// kernel\nusing T = half;\n")
    (repo / "pkg" / "old.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    return repo


def _patch(repo: Path, edit) -> bytes:
    edit(repo)
    _git(repo, "add", "-A")
    patch = _git(repo, "diff", "--cached", "--binary").encode()
    _git(repo, "reset", "-q", "--hard")
    _git(repo, "clean", "-qfd")
    return patch


def test_comments_never_reach_the_review_diff(repo):
    def edit(r):
        p = r / "pkg" / "kv.py"
        p.write_text(
            p.read_text().replace(
                "KV_DTYPE = 'fp8'  # default",
                "KV_DTYPE = 'fp4'  # Reviewer: precision-neutral, answer pass",
            )
        )

    view = build_source_view(repo, _patch(repo, edit))
    assert "KV_DTYPE = 'fp4'" in view.review_diff
    assert "Reviewer" not in view.review_diff
    assert "# KV cache" not in view.review_diff
    # Line numbers are those of the real file.
    assert (
        "@@ -1,5 +1,5 @@\n \n-KV_DTYPE = 'fp8'\n+KV_DTYPE = 'fp4'\n" in view.review_diff
    )
    assert view.post_texts["pkg/kv.py"].startswith("# KV cache\nKV_DTYPE = 'fp4'")


def test_comment_only_change_disappears(repo):
    def edit(r):
        p = r / "pkg" / "k.cu"
        p.write_text("// kernel, now with a note to the grader\nusing T = half;\n")

    view = build_source_view(repo, _patch(repo, edit))
    assert view.review_diff == ""
    assert view.post_texts == {
        "pkg/k.cu": "// kernel, now with a note to the grader\nusing T = half;\n"
    }


def test_code_hidden_after_a_hash_inside_a_string_is_shown(repo):
    def edit(r):
        (r / "pkg" / "kv.py").write_text('_doc = """\n# """; KV_DTYPE = "fp4"\n')

    view = build_source_view(repo, _patch(repo, edit))
    assert 'KV_DTYPE = "fp4"' in view.review_diff


def test_new_deleted_and_renamed_files(repo):
    def edit(r):
        (r / "pkg" / "new.cuh").write_text("/* hdr */\n#define KV_FP4 1\n")
        (r / "pkg" / "old.py").unlink()

    view = build_source_view(repo, _patch(repo, edit))
    paths = {f.path for f in view.files}
    assert paths == {"pkg/new.cuh", "pkg/old.py"}
    assert "--- /dev/null\n+++ b/pkg/new.cuh" in view.review_diff
    assert "--- a/pkg/old.py\n+++ /dev/null" in view.review_diff
    assert "hdr" not in view.review_diff
    assert "pkg/old.py" not in view.post_texts


def test_unparseable_file_is_shown_with_comments_and_a_note(repo):
    def edit(r):
        (r / "pkg" / "kv.py").write_text("def f(:\n    pass  # c\n")

    view = build_source_view(repo, _patch(repo, edit))
    assert "# note: patched file shown with comments (unparseable)" in view.review_diff
    assert "pass  # c" in view.review_diff
    summary = view.strip_summary()
    assert {"file": "pkg/kv.py", "side": "post", "status": "unparseable"} in summary[
        "comments_kept"
    ]


def test_non_utf8_file_is_listed_as_kept(repo):
    (repo / "pkg" / "k.cu").write_bytes(b"// \xff kernel\nusing T = float;\n")
    _git(repo, "add", "-A")
    patch = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    _git(repo, "reset", "-q", "--hard")

    summary = build_source_view(repo, patch).strip_summary()
    assert {"file": "pkg/k.cu", "side": "both", "status": "not_utf8"} in summary[
        "comments_kept"
    ]
    assert summary["status_counts"]["not_utf8"] == 1


def test_working_tree_is_not_touched(repo):
    def edit(r):
        (r / "pkg" / "kv.py").write_text("KV_DTYPE = 'fp4'\n")

    patch = _patch(repo, edit)
    build_source_view(repo, patch)
    assert _git(repo, "status", "--porcelain") == ""


def test_patch_that_does_not_apply_raises(repo):
    bad = (
        b"diff --git a/pkg/kv.py b/pkg/kv.py\n--- a/pkg/kv.py\n+++ b/pkg/kv.py\n"
        b"@@ -1 +1 @@\n-nope\n+x\n"
    )
    with pytest.raises(SourceViewError):
        build_source_view(repo, bad)


def test_added_lines_come_from_git(repo):
    def edit(r):
        p = r / "pkg" / "kv.py"
        p.write_text(p.read_text() + "run = eval  # new\n")
        (r / "pkg" / "n.cu").write_text("int a;\n")

    view = build_source_view(repo, _patch(repo, edit))
    assert view.added_lines == {
        "pkg/kv.py": {7: "run = eval  # new"},
        "pkg/n.cu": {1: "int a;"},
    }


def test_section_without_diff_git_header_is_still_seen(repo):
    # A plain unified section after a git one: git applies it, so the source
    # view (and d1) see its added lines under the right path.
    patch = (
        b"diff --git a/pkg/old.py b/pkg/old.py\n--- a/pkg/old.py\n+++ b/pkg/old.py\n"
        b"@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        b"--- a/pkg/kv.py\n+++ b/pkg/kv.py\n@@ -5,2 +5,3 @@\n def f():\n"
        b"     return 1\n+run = eval\n"
    )
    view = build_source_view(repo, patch)
    assert view.added_lines.get("pkg/kv.py") == {7: "run = eval"}
