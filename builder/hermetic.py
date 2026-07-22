"""Hermetic container build: apply patch inside pinned base image, network disabled."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
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


def _patch_is_empty(patch_bytes: bytes) -> bool:
    return not patch_bytes.strip()


def dockerfile_for_patch(
    *,
    allow_empty_patch: bool = False,
    patch_bytes: bytes | None = None,
) -> str:
    """Generate the hermetic engine Dockerfile text (unit-testable, no Docker).

    When ``allow_empty_patch`` is True and ``patch_bytes`` is empty/whitespace,
    skip ``git apply``. Miner builds must leave ``allow_empty_patch`` False.
    """
    empty = patch_bytes is not None and _patch_is_empty(patch_bytes)
    skip_apply = bool(allow_empty_patch and empty)

    # Cap CUDA/nvcc parallelism — cicc alone can use 6–12GB; -j=N OOMs 32GB boxes.
    lines = [
        "ARG BASE_IMAGE",
        "FROM ${BASE_IMAGE}",
        "ENV MAX_JOBS=1",
        "ENV CMAKE_BUILD_PARALLEL_LEVEL=1",
        "ENV NVCC_THREADS=1",
        "WORKDIR /src",
        "COPY baseline/ /src/",
        "COPY submission.diff /tmp/submission.diff",
    ]
    if skip_apply:
        run_steps = [
            "pip install --no-deps --no-build-isolation -e .",
            "rm -rf /src/.git",
        ]
    else:
        run_steps = [
            "git apply --whitespace=nowarn /tmp/submission.diff",
            "pip install --no-deps --no-build-isolation -e .",
            "rm -rf /src/.git",
        ]
    # Single RUN so apply failure / install failure fail the build.
    lines.append("RUN " + " \\\n    && ".join(run_steps))
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
    (ctx / "Dockerfile").write_text(
        dockerfile_for_patch(
            allow_empty_patch=allow_empty_patch,
            patch_bytes=patch_bytes,
        )
    )

    image_ref = image_ref_override or engine_image_ref(patch_hash)
    log_path = root / "build.log"

    try:
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
            *([] if allow_empty_patch else ["--network=none"]),
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "-t",
            image_ref,
            str(ctx),
        ]
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                build_cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=config.BUILD_TIMEOUT_S,
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8")[-4000:]
            return GateResult.reject(
                "hermetic_build_failed",
                build_log_tail=tail,
                image_ref=image_ref,
            )

        image_tag = image_ref
        if push:
            _docker_login_ghcr()
            push_proc = subprocess.run(
                ["docker", "push", image_ref],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if push_proc.returncode != 0:
                return GateResult.reject(
                    "registry_push_failed",
                    stderr=push_proc.stderr[-2000:],
                    image_ref=image_ref,
                )
            digest = resolve_image_repo_digest(image_ref)
            if digest is None:
                return GateResult.reject(
                    "image_digest_unresolved",
                    image_ref=image_ref,
                )
            image_ref = digest_pinned_ref(image_tag, digest)

        return GateResult.success(
            SubmissionState.BUILT,
            image_ref=image_ref,
            image_tag=image_tag,
            build_log=str(log_path),
        )
    except subprocess.TimeoutExpired as exc:
        return GateResult.reject("build_timeout", error=str(exc))
    except Exception as exc:
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
