"""Cleanup of temporary roots created by the base-apply gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gate.base_apply import check_base_apply
from gate.types import SubmissionState

pytestmark = pytest.mark.unit

_REPO = "https://example.invalid/baseline.git"
_COMMIT = "abc123"
_PATCH = b"diff --git a/vllm/foo.py b/vllm/foo.py\n"


def _completed(cmd, code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(list(cmd), code, stdout=stdout, stderr=stderr)


class _CreatedRoots:
    def __init__(self, monkeypatch):
        self.paths: list[Path] = []
        import tempfile

        real = tempfile.mkdtemp

        def spy(*args, **kwargs):
            path = real(*args, **kwargs)
            self.paths.append(Path(path))
            return path

        monkeypatch.setattr("gate.base_apply.tempfile.mkdtemp", spy)

    def assert_removed(self):
        assert self.paths, "expected check_base_apply to create a temporary root"
        assert all(not path.exists() for path in self.paths)


def _apply(**overrides):
    kwargs = {
        "baseline_repo": _REPO,
        "baseline_commit": _COMMIT,
        "patch_bytes": _PATCH,
    }
    kwargs.update(overrides)
    return check_base_apply(**kwargs)


def test_internal_root_removed_after_success(monkeypatch):
    created = _CreatedRoots(monkeypatch)
    monkeypatch.setattr(
        "gate.base_apply.subprocess.run",
        lambda cmd, **_kwargs: _completed(cmd),
    )

    result = _apply()

    assert result.ok is True
    assert result.state == SubmissionState.APPLIED
    assert result.reason == ""
    assert result.evidence["baseline_commit"] == _COMMIT
    assert result.evidence["repo_dir"].endswith("/baseline")
    created.assert_removed()


def test_internal_root_removed_after_rejected_apply(monkeypatch):
    created = _CreatedRoots(monkeypatch)

    def run(cmd, **_kwargs):
        cmd = list(cmd)
        if cmd[1] == "apply":
            return _completed(cmd, code=1, stdout="out-tail", stderr="bad patch")
        return _completed(cmd)

    monkeypatch.setattr("gate.base_apply.subprocess.run", run)

    result = _apply()

    assert result.ok is False
    assert result.state == SubmissionState.REJECTED
    assert result.reason == "git_apply_check_failed"
    assert result.evidence == {"stderr": "bad patch", "stdout": "out-tail"}
    created.assert_removed()


def test_internal_root_removed_after_checkout_failure(monkeypatch):
    created = _CreatedRoots(monkeypatch)

    def run(cmd, **_kwargs):
        cmd = list(cmd)
        if cmd[1] == "checkout":
            return _completed(cmd, code=1, stderr="checkout failed")
        return _completed(cmd)

    monkeypatch.setattr("gate.base_apply.subprocess.run", run)

    result = _apply()

    assert result.ok is False
    assert result.state == SubmissionState.REJECTED
    assert result.reason == "baseline_checkout_failed"
    assert result.evidence == {
        "stderr": "checkout failed",
        "baseline_commit": _COMMIT,
    }
    created.assert_removed()


@pytest.mark.parametrize(
    ("exc", "reason", "evidence"),
    [
        (
            subprocess.TimeoutExpired(cmd=["git", "clone"], timeout=600),
            "base_apply_timeout",
            {
                "error": str(
                    subprocess.TimeoutExpired(cmd=["git", "clone"], timeout=600)
                )
            },
        ),
        (
            subprocess.CalledProcessError(1, ["git", "clone"], stderr="clone failed"),
            "base_apply_error",
            None,
        ),
        (
            RuntimeError("boom"),
            "base_apply_error",
            {"error": "boom"},
        ),
    ],
)
def test_internal_root_removed_after_subprocess_failure(
    monkeypatch, exc, reason, evidence
):
    created = _CreatedRoots(monkeypatch)

    def run(cmd, **_kwargs):
        raise exc

    monkeypatch.setattr("gate.base_apply.subprocess.run", run)

    result = _apply()

    assert result.ok is False
    assert result.state == SubmissionState.REJECTED
    assert result.reason == reason
    if isinstance(exc, subprocess.CalledProcessError):
        assert result.evidence["stderr"] == "clone failed"
        assert result.evidence["error"] == str(exc)
    else:
        assert result.evidence == evidence
    created.assert_removed()


def test_caller_work_root_remains_after_success(monkeypatch, tmp_path):
    created = _CreatedRoots(monkeypatch)
    monkeypatch.setattr(
        "gate.base_apply.subprocess.run",
        lambda cmd, **_kwargs: _completed(cmd),
    )
    work_root = tmp_path / "caller"

    result = _apply(work_root=work_root)

    assert result.ok is True
    assert result.state == SubmissionState.APPLIED
    assert work_root.is_dir()
    assert (work_root / "submission.diff").read_bytes() == _PATCH
    assert result.evidence["repo_dir"] == str(work_root / "baseline")
    assert created.paths == []


def test_caller_work_root_remains_after_failure(monkeypatch, tmp_path):
    created = _CreatedRoots(monkeypatch)

    def run(cmd, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("gate.base_apply.subprocess.run", run)
    work_root = tmp_path / "caller"

    result = _apply(work_root=work_root)

    assert result.ok is False
    assert result.reason == "base_apply_error"
    assert work_root.is_dir()
    assert (work_root / "submission.diff").read_bytes() == _PATCH
    assert created.paths == []
