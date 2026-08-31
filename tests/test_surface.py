"""Adversarial surface-check tests for Stage 0 gate d."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


from gate.surface import check_surface, parse_diff_paths

ALLOW = ["vllm/**"]
DENY = [
    "tests/**",
    "benchmarks/**",
    ".github/**",
    "docker/**",
    "**/Dockerfile*",
    "**/pyproject.toml",
    "**/setup.py",
    "**/setup.cfg",
    "**/requirements*.txt",
    "**/CMakeLists.txt",
]


def _ok_diff(path: str = "vllm/foo.py") -> bytes:
    return f"""diff --git a/{path} b/{path}
index 1111111..2222222 100644
--- a/{path}
+++ b/{path}
@@ -1 +1 @@
-old
+new
""".encode()


def test_allows_vllm_path():
    res = check_surface(patch_bytes=_ok_diff(), allowed_paths=ALLOW, denied_paths=DENY)
    assert res.ok
    assert res.evidence["files"] == ["vllm/foo.py"]


def test_rejects_denied_tests_path():
    res = check_surface(
        patch_bytes=_ok_diff("tests/test_x.py"),
        allowed_paths=ALLOW,
        denied_paths=DENY,
    )
    assert not res.ok
    assert res.reason in ("path_denied", "path_not_allowed")


def test_rejects_path_traversal():
    bad = b"""diff --git a/vllm/../../etc/passwd b/vllm/../../etc/passwd
index 1111111..2222222 100644
--- a/vllm/../../etc/passwd
+++ b/vllm/../../etc/passwd
@@ -1 +1 @@
-x
+y
"""
    res = check_surface(patch_bytes=bad, allowed_paths=ALLOW, denied_paths=DENY)
    assert not res.ok
    assert res.reason == "path_traversal"


def test_rejects_absolute_path():
    # git may encode odd paths; also cover explicit absolute in header parse
    changes = parse_diff_paths(
        "diff --git a/vllm/x.py b/etc/passwd\n--- a/vllm/x.py\n+++ b/etc/passwd\n"
    )
    # Prefer a true absolute path in the b/ side
    abs_diff = (
        b"diff --git a/vllm/x.py b/../../../../../../etc/passwd\n"
        b"--- a/vllm/x.py\n"
        b"+++ b/../../../../../../etc/passwd\n"
        b"@@ -1 +1 @@\n"
        b"-x\n"
        b"+y\n"
    )
    res = check_surface(patch_bytes=abs_diff, allowed_paths=ALLOW, denied_paths=DENY)
    assert not res.ok
    assert res.reason == "path_traversal"
    assert changes  # smoke


def test_rejects_symlink_new_file():
    bad = b"""diff --git a/vllm/link b/vllm/link
new file mode 120000
index 0000000..1111111
--- /dev/null
+++ b/vllm/link
@@ -0,0 +1 @@
+target
"""
    res = check_surface(patch_bytes=bad, allowed_paths=ALLOW, denied_paths=DENY)
    assert not res.ok
    assert res.reason == "symlink_forbidden"


def test_rejects_rename_into_denied():
    bad = b"""diff --git a/vllm/foo.py b/tests/foo.py
similarity index 100%
rename from vllm/foo.py
rename to tests/foo.py
"""
    res = check_surface(patch_bytes=bad, allowed_paths=ALLOW, denied_paths=DENY)
    assert not res.ok
    assert res.reason in ("path_denied", "path_not_allowed")


def test_rejects_binary_patch():
    bad = b"""diff --git a/vllm/x.bin b/vllm/x.bin
index 1111111..2222222 100644
GIT binary patch
literal 3
xx
"""
    res = check_surface(patch_bytes=bad, allowed_paths=ALLOW, denied_paths=DENY)
    assert not res.ok
    assert res.reason == "binary_patch_forbidden"


def test_rejects_empty_patch():
    res = check_surface(patch_bytes=b"\n", allowed_paths=ALLOW, denied_paths=DENY)
    assert not res.ok
    assert "empty" in res.reason


def test_rejects_gitmodules():
    bad = b"""diff --git a/vllm/.gitmodules b/vllm/.gitmodules
index 1111111..2222222 100644
--- a/vllm/.gitmodules
+++ b/vllm/.gitmodules
@@ -1 +1 @@
-a
+b
"""
    # path may be denied as not matching allow, or submodule rule
    res = check_surface(patch_bytes=bad, allowed_paths=ALLOW, denied_paths=DENY)
    assert not res.ok


def test_rejects_setup_py():
    res = check_surface(
        patch_bytes=_ok_diff("setup.py"),
        allowed_paths=ALLOW,
        denied_paths=DENY,
    )
    assert not res.ok


def test_rejects_when_diff_git_path_disagrees_with_applied_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    nested = repo / "vllm"
    nested.mkdir(parents=True)
    (repo / "setup.py").write_text("safe\n")
    (nested / "setup.py").write_text("safe\n")
    subprocess.run(["git", "init", "-q", repo], check=True)
    monkeypatch.chdir(nested)
    bad = b"""diff --git a/vllm/x.py b/vllm/x.py
index 1111111..2222222 100644
--- a/setup.py
+++ b/setup.py
@@ -1 +1 @@
-safe
+PWNED
"""
    res = check_surface(patch_bytes=bad, allowed_paths=ALLOW, denied_paths=DENY)
    assert not res.ok
    assert res.reason == "path_denied"
    assert res.evidence["path"] == "setup.py"


def test_parse_diff_paths_basic():
    changes = parse_diff_paths(_ok_diff().decode())
    assert len(changes) == 1
    assert changes[0].path == "vllm/foo.py"
