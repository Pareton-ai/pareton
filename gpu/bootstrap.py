"""Idempotent remote bootstrap over SSH."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from gpu.ssh import REPO_RSYNC_EXCLUDES, exec as ssh_exec, push
from gpu.ssh import SshRunner
from gpu.types import Pod

logger = logging.getLogger(__name__)

REMOTE_REPO = "/opt/pareton"
REMOTE_VENV = f"{REMOTE_REPO}/.venv"
REMOTE_HF_CACHE = "/workspace/hf-cache"


def local_code_sha(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return (proc.stdout or "").strip() or "unknown"


def bootstrap_script(*, with_nvidia_toolkit_install: bool = True) -> str:
    """Generate a verify-first bootstrap shell script (no secrets)."""
    toolkit = ""
    if with_nvidia_toolkit_install:
        toolkit = r"""
if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "nvidia container runtime missing; installing nvidia-container-toolkit"
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  $SUDO apt-get update -y
  $SUDO apt-get install -y nvidia-container-toolkit
  $SUDO nvidia-ctk runtime configure --runtime=docker
  $SUDO systemctl restart docker || $SUDO service docker restart || true
fi
"""
    return f"""set -euo pipefail
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo -E"; fi

# Docker: verify first; install only if missing.
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | $SUDO sh
fi
command -v docker >/dev/null 2>&1 || {{ echo "docker still missing after install"; exit 1; }}

# GPU driver required (do not attempt install).
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; install NVIDIA drivers on the host before bench"
  exit 1
fi
nvidia-smi >/dev/null

# NVIDIA container runtime: install only when absent.
{toolkit}
docker info 2>/dev/null | grep -qi nvidia || echo "warning: nvidia runtime still not listed in docker info"

# Python venv tooling.
if ! python3 -c "import venv" 2>/dev/null; then
  $SUDO apt-get update -y && $SUDO apt-get install -y python3-venv python3-pip
fi

$SUDO mkdir -p {REMOTE_REPO} {REMOTE_HF_CACHE}
$SUDO chown -R "$(id -u):$(id -g)" {REMOTE_REPO} {REMOTE_HF_CACHE} || true
"""


def bootstrap_pod(
    pod: Pod,
    *,
    repo_root: Path,
    image_refs: list[str] | None = None,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
) -> str:
    """Bootstrap remote host and ship the repo. Returns local git SHA."""
    del image_refs  # pulled later after env file is written (orchestrate)
    script = bootstrap_script()
    ssh_exec(
        pod,
        f"bash -s <<'PARETON_BOOTSTRAP'\n{script}\nPARETON_BOOTSTRAP",
        timeout_s=1800.0,
        runner=runner,
        state_dir=state_dir,
    )
    code_sha = local_code_sha(repo_root)
    # Ensure remote dir exists then rsync.
    ssh_exec(
        pod,
        f"mkdir -p {REMOTE_REPO}",
        timeout_s=60.0,
        runner=runner,
        state_dir=state_dir,
    )
    push(
        pod,
        repo_root,
        f"{REMOTE_REPO}/",
        excludes=REPO_RSYNC_EXCLUDES,
        runner=runner,
        state_dir=state_dir,
    )
    ssh_exec(
        pod,
        (
            f"python3 -m venv {REMOTE_VENV} && "
            f"{REMOTE_VENV}/bin/pip install -q -r {REMOTE_REPO}/requirements.txt"
        ),
        timeout_s=1800.0,
        runner=runner,
        state_dir=state_dir,
    )
    return code_sha


def pull_engine_images(
    pod: Pod,
    image_refs: list[str],
    *,
    env_file: str,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
) -> None:
    """Source env_file, docker login (stdin), then pull. Token never on argv."""
    if not image_refs:
        return
    pulls = " && ".join(f"docker pull {shlex.quote(img)}" for img in image_refs)
    env_q = shlex.quote(env_file)
    # Single shell so login sees vars from the env file; password via stdin.
    remote = (
        f"set -a && . {env_q} && set +a && "
        'if [ -n "${PARETON_GHCR_TOKEN:-}" ]; then '
        'echo "$PARETON_GHCR_TOKEN" | docker login ghcr.io '
        '-u "${PARETON_GHCR_USER:-${PARETON_GHCR_USERNAME:-}}" --password-stdin; '
        "fi && "
        f"{pulls}"
    )
    ssh_exec(
        pod,
        remote,
        timeout_s=3600.0,
        runner=runner,
        state_dir=state_dir,
    )
