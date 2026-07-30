"""Unit tests for hermetic Dockerfile generation and registry digest refs."""

from __future__ import annotations

import subprocess

import pytest

from builder.hermetic import build_engine_image, dockerfile_for_patch
from builder.registry import (
    baseline_build_image_ref,
    baseline_engine_image_ref,
    pullable_digest_ref,
)

COMMIT = "ee0da84ab9e04ac7610e28580af62c365e898389"


@pytest.mark.unit
def test_dockerfile_nonempty_has_apply_and_entrypoint():
    text = dockerfile_for_patch(
        allow_empty_patch=False,
        patch_bytes=b"diff --git a/x b/x\n",
        baseline_commit=COMMIT,
    )
    assert "git apply --whitespace=nowarn /tmp/submission.diff" in text
    assert "pip install --no-deps --no-build-isolation -e ." in text
    assert "|| true" not in text
    assert "rm -rf /src/.git /src/rust/target /tmp/pip-* /root/.cache/pip" in text
    assert 'ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]' in text
    assert "ARG MAX_JOBS=1" in text
    assert "ARG CMAKE_BUILD_PARALLEL_LEVEL=1" in text
    assert "ENV NVCC_THREADS=1" in text
    assert "TORCH_CUDA_ARCH_LIST" not in text
    assert "# syntax=" not in text
    assert f"id=pareton-ccache-{COMMIT}" in text
    assert "CCACHE_DIR=/root/.ccache" in text
    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in text


@pytest.mark.unit
def test_dockerfile_empty_allowed_skips_apply():
    text = dockerfile_for_patch(
        allow_empty_patch=True, patch_bytes=b"", baseline_commit=COMMIT
    )
    assert "git apply" not in text
    assert "pip install --no-deps --no-build-isolation -e ." in text
    assert "|| true" not in text
    assert 'ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]' in text
    assert "ARG MAX_JOBS=1" in text
    assert "# syntax=" not in text


@pytest.mark.unit
def test_dockerfile_empty_not_allowed_still_has_apply():
    text = dockerfile_for_patch(
        allow_empty_patch=False, patch_bytes=b"   \n", baseline_commit=COMMIT
    )
    assert "git apply --whitespace=nowarn /tmp/submission.diff" in text


@pytest.mark.unit
def test_dockerfile_whitespace_empty_with_allow_skips_apply():
    text = dockerfile_for_patch(
        allow_empty_patch=True, patch_bytes=b"  \n\t", baseline_commit=COMMIT
    )
    assert "git apply" not in text


@pytest.mark.unit
def test_dockerfile_jobs_and_arch_overrides():
    text = dockerfile_for_patch(
        allow_empty_patch=True,
        patch_bytes=b"",
        baseline_commit=COMMIT,
        max_jobs=2,
        torch_cuda_arch_list="9.0",
    )
    assert "ARG MAX_JOBS=2" in text
    assert "ARG CMAKE_BUILD_PARALLEL_LEVEL=2" in text
    assert "ENV NVCC_THREADS=1" in text
    assert 'ENV TORCH_CUDA_ARCH_LIST="9.0"' in text
    assert "# syntax=" not in text


@pytest.mark.unit
def test_dockerfile_rejects_zero_jobs():
    with pytest.raises(ValueError, match="BUILD_MAX_JOBS"):
        dockerfile_for_patch(
            allow_empty_patch=True,
            patch_bytes=b"",
            baseline_commit=COMMIT,
            max_jobs=0,
        )


@pytest.mark.unit
def test_dockerfile_omits_blank_arch():
    text = dockerfile_for_patch(
        allow_empty_patch=True,
        patch_bytes=b"",
        baseline_commit=COMMIT,
        torch_cuda_arch_list="  ",
    )
    assert "TORCH_CUDA_ARCH_LIST" not in text


@pytest.mark.unit
def test_run_logged_tees_and_returns_code(tmp_path, capsys):
    from builder.hermetic import _run_logged

    log_path = tmp_path / "build.log"
    rc = _run_logged(
        ["python3", "-c", "print('hello-build')"],
        log_path=log_path,
        timeout=30,
    )
    assert rc == 0
    assert "hello-build" in log_path.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "hello-build" in err


@pytest.mark.unit
def test_build_rejects_empty_patch_without_allow(tmp_path):
    result = build_engine_image(
        baseline_repo="https://example.com/vllm.git",
        baseline_commit="a" * 40,
        base_image="ghcr.io/pareton-ai/pareton-baseline@sha256:" + ("b" * 64),
        patch_bytes=b"",
        patch_hash="sha256:" + ("c" * 64),
        work_root=tmp_path,
        push=False,
        allow_empty_patch=False,
    )
    assert not result.ok
    assert result.reason == "empty_patch_not_allowed"


@pytest.mark.unit
def test_build_timeout_appends_fail_line(tmp_path, monkeypatch):
    import builder.hermetic as hermetic

    def ok_run(*_a, **_k):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermetic.subprocess, "run", ok_run)

    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd=["docker", "build"], timeout=12)

    monkeypatch.setattr(hermetic, "_run_logged", boom)
    monkeypatch.setattr(hermetic.config, "BUILD_TIMEOUT_S", 12)

    result = build_engine_image(
        baseline_repo="https://example.com/vllm.git",
        baseline_commit=COMMIT,
        base_image="ghcr.io/pareton-ai/pareton-baseline@sha256:" + ("b" * 64),
        patch_bytes=b"diff --git a/x b/x\n",
        patch_hash="sha256:" + ("c" * 64),
        work_root=tmp_path,
        push=False,
        allow_empty_patch=False,
    )
    assert not result.ok
    assert result.reason == "build_timeout"
    log = (tmp_path / "build.log").read_text(encoding="utf-8")
    assert "FAIL build_timeout after 12s" in log


@pytest.mark.unit
def test_pullable_digest_ref_bare_and_pinned():
    digest = "sha256:" + ("a" * 64)
    assert (
        pullable_digest_ref(digest, image="pareton-engine")
        == f"ghcr.io/pareton-ai/pareton-engine@{digest}"
    )
    pinned = f"ghcr.io/other/engine@{digest.upper()}"
    assert pullable_digest_ref(pinned, image="pareton-engine") == (
        f"ghcr.io/other/engine@{digest}"
    )
    assert baseline_build_image_ref(digest) == (
        f"ghcr.io/pareton-ai/pareton-baseline@{digest}"
    )
    assert baseline_engine_image_ref(digest) == (
        f"ghcr.io/pareton-ai/pareton-engine@{digest}"
    )


@pytest.mark.unit
def test_pullable_digest_ref_rejects_bad():
    with pytest.raises(ValueError):
        pullable_digest_ref("not-a-digest", image="pareton-engine")
