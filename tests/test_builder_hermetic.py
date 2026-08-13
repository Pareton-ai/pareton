"""Unit tests for hermetic Dockerfile generation and registry digest refs."""

from __future__ import annotations

import subprocess

import pytest

from builder.hermetic import build_engine_image, dockerfile_for_patch
from campaign.engine import preset
from gate.types import GateResult, SubmissionState
from builder.registry import (
    baseline_build_image_ref,
    baseline_engine_image_ref,
    pullable_digest_ref,
)

COMMIT = "ee0da84ab9e04ac7610e28580af62c365e898389"


@pytest.mark.unit
def test_dockerfile_nonempty_has_apply_and_entrypoint(monkeypatch):
    # .env (loaded by conftest) may set build vars; pin defaults for this test.
    monkeypatch.setattr("config.BUILD_MAX_JOBS", 1)
    monkeypatch.setattr("config.TORCH_CUDA_ARCH_LIST", "")
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
    # Cross-build hits require dropping the random PEP 660 build-temp cwd
    # from the cache key; miner builds mount the shared cache read-only.
    assert "ENV CCACHE_NOHASHDIR=1" in text
    assert ",readonly" in text
    assert "sharing=locked" not in text
    assert "export CCACHE_READONLY=1" in text
    # Read-only cache dir requires a writable temp dir (ccache 4.5.1 manual).
    assert "CCACHE_TEMPDIR=/tmp/ccache-tmp" in text


@pytest.mark.unit
def test_dockerfile_empty_allowed_skips_apply(monkeypatch):
    monkeypatch.setattr("config.BUILD_MAX_JOBS", 1)
    monkeypatch.setattr("config.TORCH_CUDA_ARCH_LIST", "")
    text = dockerfile_for_patch(
        allow_empty_patch=True, patch_bytes=b"", baseline_commit=COMMIT
    )
    assert "git apply" not in text
    assert "pip install --no-deps --no-build-isolation -e ." in text
    assert "|| true" not in text
    assert 'ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]' in text
    assert "ARG MAX_JOBS=1" in text
    assert "# syntax=" not in text
    # Trusted baseline build keeps the writable mount to warm the cache.
    assert "ENV CCACHE_NOHASHDIR=1" in text
    assert "sharing=locked" in text
    assert ",readonly" not in text
    assert "CCACHE_READONLY" not in text
    assert "CCACHE_TEMPDIR" not in text


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
    assert "hello-build" not in err


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

    log_dir = tmp_path / "logs"
    result = build_engine_image(
        baseline_repo="https://example.com/vllm.git",
        baseline_commit=COMMIT,
        base_image="ghcr.io/pareton-ai/pareton-baseline@sha256:" + ("b" * 64),
        patch_bytes=b"diff --git a/x b/x\n",
        patch_hash="sha256:" + ("c" * 64),
        work_root=tmp_path / "work",
        log_dir=log_dir,
        push=False,
        allow_empty_patch=False,
    )
    assert not result.ok
    assert result.reason == "build_timeout"
    log = (log_dir / "build.log").read_text(encoding="utf-8")
    assert "FAIL build_timeout after 12s" in log


@pytest.mark.unit
def test_build_deletes_work_root_keeps_log(tmp_path, monkeypatch):
    """The GB-sized docker context must not survive the call (PAR-37)."""
    import builder.hermetic as hermetic

    monkeypatch.setattr(
        hermetic.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="boom"
        ),
    )

    log_dir = tmp_path / "logs"
    work_root = tmp_path / "work"
    result = build_engine_image(
        baseline_repo="https://example.com/vllm.git",
        baseline_commit=COMMIT,
        base_image="ghcr.io/pareton-ai/pareton-baseline@sha256:" + ("b" * 64),
        patch_bytes=b"diff --git a/x b/x\n",
        patch_hash="sha256:" + ("c" * 64),
        work_root=work_root,
        log_dir=log_dir,
        push=False,
        allow_empty_patch=False,
    )
    assert not result.ok
    assert not (work_root / "docker-context").exists()
    assert (log_dir / "build.log").exists()


@pytest.mark.unit
def test_sanitized_tail_strips_control_chars(tmp_path):
    import hashlib

    from builder.hermetic import _sanitized_tail

    log_path = tmp_path / "build.log"
    raw = b"line1\n\x1b[31mred\x1b[0m\x07 bell\nline3\n"
    log_path.write_bytes(raw)

    out = _sanitized_tail(log_path, lines=20)
    assert out["build_log_sha256"] == hashlib.sha256(raw).hexdigest()
    assert out["build_log"] == str(log_path)
    assert "\x1b" not in out["build_log_tail"]
    assert "\x07" not in out["build_log_tail"]
    assert "red" in out["build_log_tail"]
    assert "line1" in out["build_log_tail"]

    assert _sanitized_tail(tmp_path / "missing.log") == {}


@pytest.mark.unit
def test_sanitized_tail_caps_giant_single_line(tmp_path):
    from builder.hermetic import _sanitized_tail

    log_path = tmp_path / "build.log"
    # Miner-influenced output: one newline-free blob must not bypass the cap.
    log_path.write_bytes(b"A" * 2_000_000)

    out = _sanitized_tail(log_path)
    assert len(out["build_log_tail"]) <= 4096
    assert out["build_log_tail"] == "A" * 4096


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


@pytest.mark.unit
def test_baseline_build_image_ref_passes_full_engine_ref():
    digest = "sha256:" + ("6" * 64)
    ref = f"ghcr.io/pareton-ai/pareton-engine@{digest}"
    assert baseline_build_image_ref(ref) == ref


@pytest.mark.unit
def test_build_base_image_prefers_engine_pin():
    from types import SimpleNamespace

    from worker.pipeline import _build_base_image

    engine = "sha256:" + ("e" * 64)
    campaign = SimpleNamespace(
        campaign_id="c1",
        base_image_digest="sha256:" + ("b" * 64),
        bench={"baseline_engine_image_digest": engine},
    )
    assert _build_base_image(campaign) == baseline_engine_image_ref(engine)


@pytest.mark.unit
def test_build_base_image_falls_back_to_base():
    from types import SimpleNamespace

    from worker.pipeline import _build_base_image

    base = "sha256:" + ("b" * 64)
    campaign = SimpleNamespace(campaign_id="c1", base_image_digest=base, bench=None)
    assert _build_base_image(campaign) == baseline_build_image_ref(base)


# --- PAR-57: engine profile drives install_cmd + entrypoint -------------------


@pytest.mark.unit
def test_dockerfile_sglang_profile_swaps_install_and_entrypoint(monkeypatch):
    """SGLang differs from vLLM in exactly two lines (PAR-54, verified on B300)."""
    monkeypatch.setattr("config.BUILD_MAX_JOBS", 1)
    monkeypatch.setattr("config.TORCH_CUDA_ARCH_LIST", "")
    kw = dict(
        allow_empty_patch=False,
        patch_bytes=b"diff --git a/x b/x\n",
        baseline_commit=COMMIT,
    )
    vllm_text = dockerfile_for_patch(**kw)
    sglang_text = dockerfile_for_patch(**kw, engine=preset("sglang"))

    assert "pip install --no-deps --no-build-isolation -e python/" in sglang_text
    assert 'ENTRYPOINT ["python3", "-m", "sglang.launch_server"]' in sglang_text
    assert "vllm" not in sglang_text

    differing = [
        (a, b)
        for a, b in zip(vllm_text.splitlines(), sglang_text.splitlines())
        if a != b
    ]
    assert len(differing) == 2, differing


@pytest.mark.unit
def test_dockerfile_explicit_vllm_matches_no_engine(monkeypatch):
    """A campaign pinned before engine profiles must emit identical bytes."""
    monkeypatch.setattr("config.BUILD_MAX_JOBS", 1)
    monkeypatch.setattr("config.TORCH_CUDA_ARCH_LIST", "")
    kw = dict(
        allow_empty_patch=False,
        patch_bytes=b"diff --git a/x b/x\n",
        baseline_commit=COMMIT,
    )
    assert dockerfile_for_patch(**kw) == dockerfile_for_patch(**kw, engine=None)
    assert dockerfile_for_patch(**kw) == dockerfile_for_patch(
        **kw, engine=preset("vllm")
    )


@pytest.mark.unit
def test_dockerfile_sglang_keeps_miner_build_security(monkeypatch):
    """A non-default engine must not relax the read-only ccache mount."""
    monkeypatch.setattr("config.BUILD_MAX_JOBS", 1)
    monkeypatch.setattr("config.TORCH_CUDA_ARCH_LIST", "")
    text = dockerfile_for_patch(
        allow_empty_patch=False,
        patch_bytes=b"diff --git a/x b/x\n",
        baseline_commit=COMMIT,
        engine=preset("sglang"),
    )
    assert ",readonly" in text
    assert "export CCACHE_READONLY=1" in text
    assert "git apply --whitespace=nowarn /tmp/submission.diff" in text
    # Cleanup is engine-neutral: SGLang has a rust/ workspace too (PAR-54).
    assert "rm -rf /src/.git /src/rust/target /tmp/pip-* /root/.cache/pip" in text


@pytest.mark.unit
def test_dockerfile_sglang_empty_patch_skips_apply(monkeypatch):
    monkeypatch.setattr("config.BUILD_MAX_JOBS", 1)
    monkeypatch.setattr("config.TORCH_CUDA_ARCH_LIST", "")
    text = dockerfile_for_patch(
        allow_empty_patch=True,
        patch_bytes=b"",
        baseline_commit=COMMIT,
        engine=preset("sglang"),
    )
    assert "git apply" not in text
    assert "pip install --no-deps --no-build-isolation -e python/" in text
    assert ",readonly" not in text


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    [
        {"name": "trtllm", "install_cmd": "pip install -e .", "entrypoint": ["x"]},
        {"name": "vllm", "install_cmd": "ok\nRUN echo pwned", "entrypoint": ["x"]},
        {"name": "vllm", "install_cmd": "pip install -e .", "entrypoint": []},
        {"name": "vllm", "install_cmd": "pip install -e ."},
    ],
)
def test_dockerfile_rejects_bad_engine(bad):
    """Unknown names and injected newlines must fail loudly, not fall back."""
    with pytest.raises(ValueError):
        dockerfile_for_patch(
            allow_empty_patch=False,
            patch_bytes=b"diff --git a/x b/x\n",
            baseline_commit=COMMIT,
            engine=bad,
        )


@pytest.mark.unit
def test_build_engine_image_rejects_bad_engine(tmp_path):
    """An invalid profile surfaces as build_config_invalid, not a crash."""
    res = build_engine_image(
        baseline_repo="https://example.invalid/repo.git",
        baseline_commit=COMMIT,
        base_image="ghcr.io/pareton-ai/pareton-baseline@sha256:" + ("b" * 64),
        patch_bytes=b"diff --git a/x b/x\n",
        patch_hash="sha256:" + ("a" * 64),
        work_root=tmp_path / "work",
        log_dir=tmp_path / "logs",
        push=False,
        engine={"name": "nope", "install_cmd": "x", "entrypoint": ["y"]},
    )
    assert not res.ok
    assert res.reason == "build_config_invalid"


@pytest.mark.unit
def test_cli_engine_flag_maps_to_preset(monkeypatch, tmp_path):
    """`--engine sglang` must reach build_engine_image as the SGLang profile."""
    import builder.__main__ as cli

    seen: dict = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return GateResult.success(SubmissionState.BUILT, image_ref="img")

    monkeypatch.setattr(cli, "build_engine_image", fake_build)
    rc = cli.main(
        [
            "--baseline-repo",
            "https://example.invalid/repo.git",
            "--baseline-commit",
            COMMIT,
            "--base-image",
            "ghcr.io/pareton-ai/pareton-baseline@sha256:" + ("b" * 64),
            "--image-ref",
            "ghcr.io/pareton-ai/pareton-engine:baseline-sglang",
            "--empty-patch",
            "--engine",
            "sglang",
        ]
    )
    assert rc == 0
    assert seen["engine"] == preset("sglang")

    seen.clear()
    cli.main(
        [
            "--baseline-repo",
            "https://example.invalid/repo.git",
            "--baseline-commit",
            COMMIT,
            "--base-image",
            "ghcr.io/pareton-ai/pareton-baseline@sha256:" + ("b" * 64),
            "--image-ref",
            "ghcr.io/pareton-ai/pareton-engine:baseline",
            "--empty-patch",
        ]
    )
    assert seen["engine"] is None
