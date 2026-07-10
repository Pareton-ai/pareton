"""Gate c: git apply --check against pinned baseline (no fuzz)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from gate.types import GateResult, SubmissionState


def check_base_apply(
    *,
    baseline_repo: str,
    baseline_commit: str,
    patch_bytes: bytes,
    work_root: Path | None = None,
) -> GateResult:
    """Clone/fetch baseline at commit and run `git apply --check --whitespace=nowarn`."""
    root = work_root or Path(tempfile.mkdtemp(prefix="pareton-apply-"))
    root.mkdir(parents=True, exist_ok=True)
    repo_dir = root / "baseline"
    patch_path = root / "submission.diff"
    patch_path.write_bytes(patch_bytes)

    try:
        if not (repo_dir / ".git").exists():
            subprocess.run(
                ["git", "clone", "--filter=blob:none", baseline_repo, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", baseline_commit],
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
            # full fetch fallback for shallow clones that miss the commit
            subprocess.run(
                ["git", "fetch", "origin", baseline_commit],
                cwd=repo_dir,
                check=True,
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
                baseline_commit=baseline_commit,
            )

        apply = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path)],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if apply.returncode != 0:
            return GateResult.reject(
                "git_apply_check_failed",
                stderr=apply.stderr[-2000:],
                stdout=apply.stdout[-1000:],
            )
        return GateResult.success(
            SubmissionState.APPLIED,
            repo_dir=str(repo_dir),
            baseline_commit=baseline_commit,
        )
    except subprocess.TimeoutExpired as exc:
        return GateResult.reject("base_apply_timeout", error=str(exc))
    except subprocess.CalledProcessError as exc:
        return GateResult.reject(
            "base_apply_error",
            error=str(exc),
            stderr=(exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        )
    except Exception as exc:
        return GateResult.reject("base_apply_error", error=str(exc))


def check_base_apply_local(
    *,
    repo_dir: Path,
    patch_bytes: bytes,
) -> GateResult:
    """Run apply --check against an already-checked-out repo (tests / offline)."""
    with tempfile.NamedTemporaryFile(suffix=".diff", delete=False) as tmp:
        tmp.write(patch_bytes)
        patch_path = Path(tmp.name)
    try:
        apply = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path)],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if apply.returncode != 0:
            return GateResult.reject(
                "git_apply_check_failed",
                stderr=apply.stderr[-2000:],
            )
        return GateResult.success(SubmissionState.APPLIED, repo_dir=str(repo_dir))
    finally:
        patch_path.unlink(missing_ok=True)
