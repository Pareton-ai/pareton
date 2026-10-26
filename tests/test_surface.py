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


_RENAME = b"""diff --git a/vllm/a.py b/vllm/b.py
similarity index 100%
rename from vllm/a.py
rename to vllm/b.py
"""


def test_patch_limits_unset_keep_current_behavior():
    long_line = _ok_diff().replace(b"+new", b"+" + b"x" * 5000)
    for patch in (long_line, _RENAME):
        res = check_surface(patch_bytes=patch, allowed_paths=ALLOW, denied_paths=DENY)
        assert res.ok, res.reason


def test_rejects_patch_over_max_bytes():
    patch = _ok_diff()
    kwargs = {"allowed_paths": ALLOW, "denied_paths": DENY}
    assert check_surface(patch_bytes=patch, max_patch_bytes=len(patch), **kwargs).ok
    res = check_surface(patch_bytes=patch, max_patch_bytes=len(patch) - 1, **kwargs)
    assert res.reason == "patch_too_large"
    assert res.evidence["size"] == len(patch)


def test_rejects_added_line_over_max_chars():
    patch = _ok_diff().replace(b"+new", b"+" + b"x" * 401)
    kwargs = {"allowed_paths": ALLOW, "denied_paths": DENY}
    res = check_surface(patch_bytes=patch, max_added_line_chars=400, **kwargs)
    assert res.reason == "added_line_too_long"
    assert res.evidence["length"] == 401
    ok = _ok_diff().replace(b"+new", b"+" + b"x" * 400)
    assert check_surface(patch_bytes=ok, max_added_line_chars=400, **kwargs).ok


def test_line_limit_ignores_headers_and_removed_lines():
    path = "vllm/" + "d" * 60 + ".py"
    patch = f"""diff --git a/{path} b/{path}
index 1111111..2222222 100644
--- a/{path}
+++ b/{path}
@@ -1 +1 @@
-{"y" * 100}
+new
""".encode()
    res = check_surface(
        patch_bytes=patch,
        allowed_paths=ALLOW,
        denied_paths=DENY,
        max_added_line_chars=20,
    )
    assert res.ok, res.reason


def test_line_limit_counts_unicode_separators_as_one_line():
    patch = _ok_diff().replace(b"+new", ("+" + "x " * 30).encode())
    res = check_surface(
        patch_bytes=patch,
        allowed_paths=ALLOW,
        denied_paths=DENY,
        max_added_line_chars=40,
    )
    assert res.reason == "added_line_too_long"


def test_reject_renames_blocks_in_bounds_rename():
    res = check_surface(
        patch_bytes=_RENAME,
        allowed_paths=ALLOW,
        denied_paths=DENY,
        reject_renames=True,
    )
    assert res.reason == "rename_forbidden"
    assert res.evidence["path"] == "vllm/a.py"


_MODE_CHANGE = b"""diff --git a/vllm/a.py b/vllm/a.py
old mode 100644
new mode 100755
"""

_MODE_CHANGE_WITH_EDIT = (
    _MODE_CHANGE
    + b"""index 1111111..2222222
--- a/vllm/a.py
+++ b/vllm/a.py
@@ -1 +1 @@
-old
+new
"""
)

_NEW_EXECUTABLE = b"""diff --git a/vllm/run.py b/vllm/run.py
new file mode 100755
index 0000000..2222222
--- /dev/null
+++ b/vllm/run.py
@@ -0,0 +1 @@
+new
"""

_NEW_REGULAR = _NEW_EXECUTABLE.replace(b"100755", b"100644")


def test_mode_changes_pass_when_unset():
    for patch in (_MODE_CHANGE_WITH_EDIT, _NEW_EXECUTABLE):
        res = check_surface(patch_bytes=patch, allowed_paths=ALLOW, denied_paths=DENY)
        assert res.ok, res.reason


@pytest.mark.parametrize(
    ("patch", "path", "mode"),
    [
        (_MODE_CHANGE, "vllm/a.py", "100755"),
        (_MODE_CHANGE_WITH_EDIT, "vllm/a.py", "100755"),
        (_NEW_EXECUTABLE, "vllm/run.py", "100755"),
    ],
)
def test_reject_mode_changes_blocks_mode_edits(patch, path, mode):
    res = check_surface(
        patch_bytes=patch,
        allowed_paths=ALLOW,
        denied_paths=DENY,
        reject_mode_changes=True,
    )
    assert res.reason == "mode_change_forbidden"
    assert res.evidence == {"path": path, "mode": mode}


def test_reject_mode_changes_blocks_copies():
    # A copy takes its source's mode (an executable copies as executable), and
    # without the baseline the gate can't tell which, so copies are refused.
    copy = b"""diff --git a/vllm/run.sh b/vllm/run2.py
similarity index 100%
copy from vllm/run.sh
copy to vllm/run2.py
"""
    kwargs = {"allowed_paths": ALLOW, "denied_paths": DENY}
    assert check_surface(patch_bytes=copy, **kwargs).ok
    res = check_surface(patch_bytes=copy, reject_mode_changes=True, **kwargs)
    assert res.reason == "mode_change_forbidden"
    assert res.evidence == {"path": "vllm/run2.py", "mode": "copy"}


def test_reject_mode_changes_allows_plain_edits_and_new_files():
    for patch in (_ok_diff(), _NEW_REGULAR):
        res = check_surface(
            patch_bytes=patch,
            allowed_paths=ALLOW,
            denied_paths=DENY,
            reject_mode_changes=True,
        )
        assert res.ok, res.reason
