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
REMOTE_ENGINE_CACHE = "/workspace/engine-cache"


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
if ! $SUDO docker info 2>/dev/null | grep -qi nvidia; then
  echo "nvidia container runtime missing; installing nvidia-container-toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  tmp_list=$(mktemp)
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o "$tmp_list"
  grep -q '^deb ' "$tmp_list" || { echo "nvidia toolkit list was not a deb source"; cat "$tmp_list"; exit 1; }
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' "$tmp_list" | \
    $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  rm -f "$tmp_list"
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
$SUDO docker info 2>/dev/null | grep -qi nvidia || echo "warning: nvidia runtime still not listed in docker info"

# Non-root (e.g. Shadeform): allow bare docker for bench/lifecycle.
# Must run AFTER any docker restart (toolkit install) so chmod hits the
# final socket; usermod alone is best-effort and may not apply mid-session.
if [ "$(id -u)" -ne 0 ]; then
  $SUDO usermod -aG docker "$(id -un)" 2>/dev/null || true
  $SUDO chmod 666 /var/run/docker.sock 2>/dev/null || true
fi

# Python venv tooling: ``import venv`` can succeed without ensurepip on
# Debian/Ubuntu; probe ensurepip and install the matching pythonX.Y-venv.
if ! python3 -c "import ensurepip" 2>/dev/null; then
  $SUDO apt-get update -y
  PYVER="$(python3 -c 'import sys; print("%d.%d" % (sys.version_info.major, sys.version_info.minor))')"
  $SUDO apt-get install -y "python${{PYVER}}-venv" python3-pip \
    || $SUDO apt-get install -y python3-venv python3-pip
fi
python3 -c "import ensurepip" || {{
  echo "ensurepip still missing after apt install; cannot create venv"
  exit 1
}}

$SUDO mkdir -p {REMOTE_REPO} {REMOTE_HF_CACHE} {REMOTE_ENGINE_CACHE}
$SUDO chown -R "$(id -u):$(id -g)" {REMOTE_REPO} {REMOTE_HF_CACHE} {REMOTE_ENGINE_CACHE} || true
"""


def _extra_ssh_pubkeys() -> list[str]:
    try:
        import config as _cfg

        return list(getattr(_cfg, "GPU_EXTRA_SSH_PUBKEYS", []) or [])
    except Exception:  # noqa: BLE001
        return []


def authorize_extra_keys(
    pod: Pod,
    *,
    keys: list[str] | None = None,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
) -> None:
    """Append operator pubkeys to the pod's authorized_keys (idempotent)."""
    keys = _extra_ssh_pubkeys() if keys is None else keys
    if not keys:
        return
    auth = "$HOME/.ssh/authorized_keys"
    lines = [
        'mkdir -p "$HOME/.ssh"',
        'chmod 700 "$HOME/.ssh"',
        f'touch "{auth}"',
        f'chmod 600 "{auth}"',
        # A file not ending in a newline would concatenate onto the last key.
        f'if [ -s "{auth}" ] && [ -n "$(tail -c1 "{auth}")" ]; then'
        f' printf "\\n" >> "{auth}"; fi',
    ]
    for key in keys:
        q = shlex.quote(key)
        lines.append(f'grep -qxF -- {q} "{auth}" || printf "%s\\n" {q} >> "{auth}"')
    ssh_exec(
        pod,
        " && ".join(lines),
        timeout_s=60.0,
        runner=runner,
        state_dir=state_dir,
    )
    logger.info("authorized %d extra SSH key(s) on pod %s", len(keys), pod.name)


def remote_docker(pod: Pod) -> str:
    """Docker argv prefix; Shadeform (and other non-root) need sudo."""
    if (pod.ssh.user or "").strip() in ("", "root"):
        return "docker"
    return "sudo -E docker"


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
    authorize_extra_keys(pod, runner=runner, state_dir=state_dir)
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
            f"rm -rf {REMOTE_VENV} && "
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
    docker = remote_docker(pod)
    pulls = " && ".join(f"{docker} pull {shlex.quote(img)}" for img in image_refs)
    env_q = shlex.quote(env_file)
    # Single shell so login sees vars from the env file; password via stdin.
    remote = (
        f"set -a && . {env_q} && set +a && "
        'if [ -n "${PARETON_GHCR_TOKEN:-}" ]; then '
        f'echo "$PARETON_GHCR_TOKEN" | {docker} login ghcr.io '
        '-u "${PARETON_GHCR_USER:-${PARETON_GHCR_USERNAME:-}}" --password-stdin && '
        # On a non-root pod the login runs under sudo, so docker writes
        # $HOME/.docker/config.json owned by root. bench/lifecycle.py then runs
        # bare docker as the pod user, cannot read those credentials, and falls
        # back to an anonymous pull that GHCR refuses. Hand the file back, the
        # same way bootstrap_script does for the repo and cache dirs.
        'if [ "$(id -u)" -ne 0 ]; then '
        'sudo chown -R "$(id -u):$(id -g)" "$HOME/.docker" 2>/dev/null || true; '
        "fi; "
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
