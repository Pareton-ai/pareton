"""Hermetic container build: apply patch inside pinned base image, network disabled."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import config
from builder.digest import (
    digest_pinned_ref,
    mock_digest_from_patch_hash,
    resolve_image_repo_digest,
)
from builder.registry import engine_image_ref
from gate.types import GateResult, SubmissionState

logger = logging.getLogger(__name__)

_VLLM_API_SERVER_ENTRYPOINT = (
    'ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]'
)
_CACHE_ID_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _progress(msg: str) -> None:
    """Human-visible step line for ops / tmux (stderr; never secrets)."""
    print(f"pareton-builder: {msg}", file=sys.stderr, flush=True)


def _patch_is_empty(patch_bytes: bytes) -> bool:
    return not patch_bytes.strip()


def _ccache_mount_id(baseline_commit: str) -> str:
    cleaned = _CACHE_ID_SAFE.sub("-", (baseline_commit or "unknown").strip())[:64]
    return f"pareton-ccache-{cleaned or 'unknown'}"


def dockerfile_for_patch(
    *,
    allow_empty_patch: bool = False,
    patch_bytes: bytes | None = None,
    baseline_commit: str = "unknown",
    max_jobs: int | None = None,
    torch_cuda_arch_list: str | None = None,
) -> str:
    """Generate the hermetic engine Dockerfile text (unit-testable, no Docker).

    When ``allow_empty_patch`` is True and ``patch_bytes`` is empty/whitespace,
    skip ``git apply``. Miner builds must leave ``allow_empty_patch`` False.

    Parallelism uses ARG (build-time) so job count is not a permanent image ENV
    fingerprint; ARG values are visible to RUN. NVCC_THREADS stays 1 (ENV).
    """
    jobs = int(config.BUILD_MAX_JOBS if max_jobs is None else max_jobs)
    if jobs < 1:
        raise ValueError(f"BUILD_MAX_JOBS must be >= 1, got {jobs}")
    arch = (
        config.TORCH_CUDA_ARCH_LIST
        if torch_cuda_arch_list is None
        else str(torch_cuda_arch_list).strip()
    )

    empty = patch_bytes is not None and _patch_is_empty(patch_bytes)
    skip_apply = bool(allow_empty_patch and empty)

    # Cap CUDA/nvcc: cicc alone can use 6–12GB; NVCC_THREADS stays 1.
    lines = [
        "ARG BASE_IMAGE",
        "FROM ${BASE_IMAGE}",
        f"ARG MAX_JOBS={jobs}",
        f"ARG CMAKE_BUILD_PARALLEL_LEVEL={jobs}",
        "ENV NVCC_THREADS=1",
        "ENV CCACHE_DIR=/root/.ccache",
    ]
    if arch:
        lines.append(f"ENV TORCH_CUDA_ARCH_LIST={json.dumps(arch)}")
    lines.extend(
        [
            "WORKDIR /src",
            "COPY baseline/ /src/",
            "COPY submission.diff /tmp/submission.diff",
        ]
    )

    setup_ccache = (
        "if command -v ccache >/dev/null 2>&1; then "
        "export CMAKE_C_COMPILER_LAUNCHER=ccache "
        "CMAKE_CXX_COMPILER_LAUNCHER=ccache "
        "CMAKE_CUDA_COMPILER_LAUNCHER=ccache; "
        'else echo "pareton-builder: ccache missing; continuing without"; fi'
    )
    if skip_apply:
        body_steps = [
            "pip install --no-deps --no-build-isolation -e .",
            # Drop compile debris so export does not unpack multi-GB rust/CUDA
            # intermediates (VPS ENOSPC on unpack after a successful wheel build).
            "rm -rf /src/.git /src/rust/target /tmp/pip-* /root/.cache/pip",
        ]
    else:
        body_steps = [
            "git apply --whitespace=nowarn /tmp/submission.diff",
            "pip install --no-deps --no-build-isolation -e .",
            "rm -rf /src/.git /src/rust/target /tmp/pip-* /root/.cache/pip",
        ]
    run_parts = [setup_ccache, *body_steps]
    cache_id = _ccache_mount_id(baseline_commit)
    # No # syntax= line — BuildKit builtin frontend supports RUN --mount.
    lines.append(
        f"RUN --mount=type=cache,id={cache_id},target=/root/.ccache,sharing=locked "
        + " \\\n    && ".join(run_parts)
    )
    lines.append(_VLLM_API_SERVER_ENTRYPOINT)
    lines.append("")
    return "\n".join(lines)


def _docker_login_ghcr() -> None:
    if not config.GHCR_TOKEN or not config.GHCR_USERNAME:
        logger.warning("GHCR credentials not set; skip docker login")
        return
    subprocess.run(
        [
            "docker",
            "login",
            "ghcr.io",
            "-u",
            config.GHCR_USERNAME,
            "--password-stdin",
        ],
        input=config.GHCR_TOKEN,
        text=True,
        check=False,
        capture_output=True,
    )


def _run_logged(
    cmd: list[str],
    *,
    log_path: Path,
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run cmd, tee stdout/stderr to log_path and stderr, enforce timeout."""
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"+ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None

        def _tee() -> None:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                sys.stderr.write(line)
                sys.stderr.flush()

        reader = threading.Thread(target=_tee, name="pareton-build-tee", daemon=True)
        reader.start()
        try:
            return int(proc.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            reader.join(timeout=5)
            raise
        finally:
            reader.join(timeout=30)


def build_engine_image(
    *,
    baseline_repo: str,
    baseline_commit: str,
    base_image: str,
    patch_bytes: bytes,
    patch_hash: str,
    work_root: Path | None = None,
    push: bool = True,
    allow_empty_patch: bool = False,
    image_ref_override: str | None = None,
) -> GateResult:
    """Build and optionally push an engine image tagged by patch_hash.

    ``allow_empty_patch`` is ops-only (A2b baseline engine). Miner gate surface
    still rejects empty patches before this runs.
    """
    if _patch_is_empty(patch_bytes) and not allow_empty_patch:
        return GateResult.reject("empty_patch_not_allowed")

    root = work_root or Path(tempfile.mkdtemp(prefix="pareton-build-"))
    root.mkdir(parents=True, exist_ok=True)
    ctx = root / "docker-context"
    if ctx.exists():
        shutil.rmtree(ctx)
    ctx.mkdir(parents=True)
    repo_dir = ctx / "baseline"
    (ctx / "submission.diff").write_bytes(patch_bytes)
    try:
        dockerfile = dockerfile_for_patch(
            allow_empty_patch=allow_empty_patch,
            patch_bytes=patch_bytes,
            baseline_commit=baseline_commit,
        )
    except ValueError as exc:
        return GateResult.reject("build_config_invalid", error=str(exc))
    (ctx / "Dockerfile").write_text(dockerfile)

    image_ref = image_ref_override or engine_image_ref(patch_hash)
    log_path = root / "build.log"
    log_path.write_text("", encoding="utf-8")
    _progress(f"work_root={root}")
    _progress(f"build_log={log_path}")
    _progress(f"image_ref={image_ref} base_image={base_image}")
    _progress(
        f"build_max_jobs={config.BUILD_MAX_JOBS} "
        f"torch_cuda_arch_list={config.TORCH_CUDA_ARCH_LIST or '(default)'}"
    )

    try:
        _progress("step 1/5 clone baseline repo")
        clone = subprocess.run(
            ["git", "clone", baseline_repo, str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if clone.returncode != 0:
            return GateResult.reject(
                "baseline_clone_failed",
                stderr=clone.stderr[-2000:],
            )
        _progress(f"step 2/5 checkout {baseline_commit}")
        checkout = subprocess.run(
            ["git", "checkout", "--force", baseline_commit],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if checkout.returncode != 0:
            subprocess.run(
                ["git", "fetch", "origin", baseline_commit],
                cwd=repo_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            checkout = subprocess.run(
                ["git", "checkout", "--force", baseline_commit],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
        if checkout.returncode != 0:
            return GateResult.reject(
                "baseline_checkout_failed",
                stderr=checkout.stderr[-2000:],
            )

        # Miner builds stay --network=none. A2b empty-patch needs network so
        # cmake FetchContent can populate /src/.deps (vLLM pin ee0da84).
        build_cmd = [
            "docker",
            "build",
            "--progress=plain",
            *([] if allow_empty_patch else ["--network=none"]),
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--build-arg",
            f"MAX_JOBS={config.BUILD_MAX_JOBS}",
            "--build-arg",
            f"CMAKE_BUILD_PARALLEL_LEVEL={config.BUILD_MAX_JOBS}",
            "-t",
            image_ref,
            str(ctx),
        ]
        _progress(
            f"step 3/5 docker build (timeout={config.BUILD_TIMEOUT_S}s); "
            f"tail -f {log_path}"
        )
        build_env = {**os.environ, "DOCKER_BUILDKIT": "1"}
        rc = _run_logged(
            build_cmd,
            log_path=log_path,
            timeout=config.BUILD_TIMEOUT_S,
            env=build_env,
        )
        if rc != 0:
            tail = log_path.read_text(encoding="utf-8")[-4000:]
            return GateResult.reject(
                "hermetic_build_failed",
                build_log_tail=tail,
                image_ref=image_ref,
            )

        image_tag = image_ref
        if push:
            _progress("step 4/5 docker login + push")
            _docker_login_ghcr()
            push_rc = _run_logged(
                ["docker", "push", image_ref],
                log_path=log_path,
                timeout=600,
            )
            if push_rc != 0:
                tail = log_path.read_text(encoding="utf-8")[-2000:]
                return GateResult.reject(
                    "registry_push_failed",
                    stderr=tail,
                    image_ref=image_ref,
                )
            _progress("step 5/5 resolve RepoDigest")
            digest = resolve_image_repo_digest(image_ref)
            if digest is None:
                return GateResult.reject(
                    "image_digest_unresolved",
                    image_ref=image_ref,
                )
            image_ref = digest_pinned_ref(image_tag, digest)
            _progress(f"done image_ref={image_ref}")
        else:
            _progress(f"done (no push) image_tag={image_tag}")

        return GateResult.success(
            SubmissionState.BUILT,
            image_ref=image_ref,
            image_tag=image_tag,
            build_log=str(log_path),
        )
    except subprocess.TimeoutExpired as exc:
        fail_line = f"FAIL build_timeout after {config.BUILD_TIMEOUT_S}s\n"
        try:
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(fail_line)
        except OSError:
            pass
        _progress(f"FAIL build_timeout after {config.BUILD_TIMEOUT_S}s; see {log_path}")
        return GateResult.reject("build_timeout", error=str(exc))
    except Exception as exc:
        _progress(f"FAIL build_error: {type(exc).__name__}")
        return GateResult.reject("build_error", error=str(exc))


def build_engine_image_local_mock(
    *,
    patch_hash: str,
    patch_bytes: bytes,
    work_root: Path | None = None,
) -> GateResult:
    """Offline mock build for e2e without Docker/registry.

    Writes a content-addressed artifact marker file and returns a fake image ref.
    """
    root = work_root or config.WORK_DIR / "mock-builds"
    root.mkdir(parents=True, exist_ok=True)
    image_tag = engine_image_ref(patch_hash)
    digest = mock_digest_from_patch_hash(patch_hash)
    image_ref = digest_pinned_ref(image_tag, digest)
    artifact = root / f"{patch_hash.replace(':', '_')}.built"
    artifact.write_bytes(patch_bytes)
    marker = root / f"{patch_hash.replace(':', '_')}.ref"
    marker.write_text(image_ref + "\n", encoding="utf-8")
    return GateResult.success(
        SubmissionState.BUILT,
        image_ref=image_ref,
        image_tag=image_tag,
        artifact=str(artifact),
        mock=True,
    )
