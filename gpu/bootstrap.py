"""Idempotent remote bootstrap over SSH."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path

from gpu.ssh import REPO_RSYNC_EXCLUDES, exec as ssh_exec, push
from gpu.ssh import SshRunner
from gpu.types import Pod

logger = logging.getLogger(__name__)

REMOTE_REPO = "/opt/pareton"
REMOTE_HF_CACHE = "/workspace/hf-cache"
REMOTE_ENGINE_CACHE = "/workspace/engine-cache"


def local_code_sha(repo_root: Path) -> str:
    # Runtime images exclude .git; the image build records the source revision.
    baked = os.environ.get("PARETON_CODE_SHA", "")
    if baked and baked != "unknown":
        return baked
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

# The host only needs the SSH data plane; Python and its virtualenv are in
# the harness image. Docker commands use sudo for non-root provider accounts.
if ! command -v rsync >/dev/null; then
  $SUDO apt-get update -y
  $SUDO apt-get install -y rsync
fi

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


def harness_image(code_sha: str) -> str:
    """A safe local image tag, also for development revisions such as unknown."""
    import hashlib

    return f"pareton-harness:{hashlib.sha256(code_sha.encode()).hexdigest()[:20]}"


def harness_command(
    pod: Pod,
    *,
    code_sha: str,
    env_file: str,
    request: str,
    output: str,
    mock_engine: bool = False,
) -> str:
    """Run beside the engine containers using the GPU host's Docker daemon.

    Host networking preserves the loopback ports bench/lifecycle discovers.
    Identical host/container paths let sibling engines mount staged weights
    and caches. GPU utility access preserves the harness environment probes.
    """
    import uuid

    name = "pareton-harness-" + uuid.uuid4().hex[:12]
    args = [
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--network",
        "host",
        "--uts",
        "host",
        "--cgroupns",
        "host",
        "--gpus",
        "all",
        "--env",
        "NVIDIA_DRIVER_CAPABILITIES=utility",
        "--volume",
        "/var/run/docker.sock:/var/run/docker.sock",
        "--volume",
        f"{REMOTE_REPO}:{REMOTE_REPO}",
        "--volume",
        f"{REMOTE_HF_CACHE}:{REMOTE_HF_CACHE}",
        "--volume",
        f"{REMOTE_ENGINE_CACHE}:{REMOTE_ENGINE_CACHE}",
        "--env-file",
        env_file,
        "--env",
        f"PARETON_BENCH_CODE_SHA={code_sha}",
        "--env",
        f"DOCKER_CONFIG={REMOTE_REPO}/.docker",
        harness_image(code_sha),
        "python",
        "-m",
        "bench",
        "--request",
        request,
        "--output-dir",
        output,
    ]
    if mock_engine:
        args.append("--mock-engine")
    docker = remote_docker(pod)
    # Root in the harness writes evidence; restore ownership even on a failing
    # benchmark so rsync works for non-root SSH accounts.
    return (
        f"mkdir -p {shlex.quote(output)} && "
        f"{docker} {' '.join(shlex.quote(arg) for arg in args)}; rc=$?; "
        f"{docker} rm -f {shlex.quote(name)} >/dev/null 2>&1 || true; "
        f'if [ "$(id -u)" -ne 0 ]; then sudo chown -R "$(id -u):$(id -g)" '
        f"{shlex.quote(output)}; fi; exit $rc"
    )


def bootstrap_pod(
    pod: Pod,
    *,
    repo_root: Path,
    image_refs: list[str] | None = None,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
) -> str:
    """Bootstrap the GPU host and build its harness image. Returns source SHA."""
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
            f"{remote_docker(pod)} build --target runtime "
            f"--build-arg {shlex.quote('PARETON_CODE_SHA=' + code_sha)} "
            f"-t {shlex.quote(harness_image(code_sha))} {REMOTE_REPO}"
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
    """Source env_file, docker login (stdin), then pull. Token never on argv.

    The pulls are best-effort warming, not a gate. bench/lifecycle.py pulls
    each image again at its own start turn, and bench/main.py turns a failure
    there into infra_failed for that one entry, which is decision 28: the round
    continues with the rest. Chaining these with && let one bad challenger
    image abort a whole 9-start round before any engine started.

    The login and the env file stay a gate. A bad GHCR token is host auth, not
    a candidate fault, and decision 28 does not cover it: failing fast here
    beats staging hundreds of GB of weights first and then voiding the round.
    """
    if not image_refs:
        return
    docker = remote_docker(pod)
    # Braces keep the group bound to the login's &&; without them only the
    # first pull would be skipped when the login fails. The trailing `true`
    # keeps a failed pull out of the exit code, so ssh_exec stays checked and
    # only login or env-file failures raise.
    pulls = "; ".join(f"{docker} pull {shlex.quote(img)}" for img in image_refs)
    env_q = shlex.quote(env_file)
    # Single shell so login sees vars from the env file; password via stdin.
    remote = (
        f"export DOCKER_CONFIG={REMOTE_REPO}/.docker && "
        f"set -a && . {env_q} && set +a && "
        'if [ -n "${PARETON_GHCR_TOKEN:-}" ]; then '
        f'echo "$PARETON_GHCR_TOKEN" | {docker} login ghcr.io '
        '-u "${PARETON_GHCR_USER:-${PARETON_GHCR_USERNAME:-}}" --password-stdin && '
        # The harness mounts REMOTE_REPO and uses the same credentials.
        'if [ "$(id -u)" -ne 0 ]; then '
        'sudo chown -R "$(id -u):$(id -g)" "$DOCKER_CONFIG" 2>/dev/null || true; '
        "fi; "
        "fi && "
        f"{{ {pulls}; true; }}"
    )
    ssh_exec(
        pod,
        remote,
        timeout_s=3600.0,
        runner=runner,
        state_dir=state_dir,
    )
