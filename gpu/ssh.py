"""SSH / rsync data plane for GPU pods (subprocess, injectable runner)."""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from gpu.errors import GpuError
from gpu.registry import _state_dir
from gpu.types import ExecResult, Pod

logger = logging.getLogger(__name__)

REPO_RSYNC_EXCLUDES: tuple[str, ...] = (
    ".env",
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "out/",
    "*.pyc",
    "docs/",
)


@dataclass
class SshResult:
    returncode: int
    stdout: str
    stderr: str


SshRunner = Callable[..., SshResult]

# Cloud GPU providers recycle ip:port; accept-new rejects a *changed* key.
_HOST_KEY_CHANGED = "REMOTE HOST IDENTIFICATION HAS CHANGED"


def ssh_base_opts(pod: Pod, *, state_dir: Path | None = None) -> list[str]:
    known = _state_dir(state_dir) / "known_hosts"
    return [
        "-i",
        str(pod.key_path),
        "-p",
        str(pod.ssh.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known}",
    ]


def _host_key_spec(pod: Pod) -> str:
    """OpenSSH known_hosts / ssh-keygen -R host token for this pod."""
    if int(pod.ssh.port) == 22:
        return pod.ssh.host
    return f"[{pod.ssh.host}]:{pod.ssh.port}"


def _is_host_key_changed(stderr: str) -> bool:
    return _HOST_KEY_CHANGED in (stderr or "")


def forget_host_key(pod: Pod, *, state_dir: Path | None = None) -> None:
    """Drop this pod's host key from Pareton's known_hosts only (not ~/.ssh)."""
    known = _state_dir(state_dir) / "known_hosts"
    if not known.is_file():
        return
    spec = _host_key_spec(pod)
    try:
        subprocess.run(
            ["ssh-keygen", "-R", spec, "-f", str(known)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ssh-keygen -R %s failed: %s", spec, exc)


def _run_with_hostkey_retry(
    runner: SshRunner,
    argv: Sequence[str],
    *,
    timeout: float,
    pod: Pod,
    state_dir: Path | None,
) -> SshResult:
    result = runner(argv, timeout=timeout)
    if result.returncode == 0 or not _is_host_key_changed(result.stderr):
        return result
    logger.warning(
        "SSH host key changed for %s:%s (recycled cloud IP?); "
        "forgetting Pareton known_hosts entry and retrying once",
        pod.ssh.host,
        pod.ssh.port,
    )
    forget_host_key(pod, state_dir=state_dir)
    return runner(argv, timeout=timeout)


def default_ssh_runner(
    cmd: Sequence[str],
    *,
    timeout: float,
    input_text: str | None = None,
) -> SshResult:
    # Never log full argv: ssh remote commands may embed secrets.
    if cmd and cmd[0] == "ssh" and len(cmd) >= 2:
        logger.info("ssh/rsync: ssh ... %s", cmd[-2])
    elif cmd:
        logger.info("ssh/rsync: %s ...", cmd[0])
    try:
        proc = subprocess.run(
            list(cmd),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GpuError(f"ssh/rsync timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise GpuError(f"command not found: {cmd[0]!r}") from exc
    return SshResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def exec(
    pod: Pod,
    cmd: str | Sequence[str],
    *,
    timeout_s: float = 600.0,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
    check: bool = True,
) -> ExecResult:
    runner = runner or default_ssh_runner
    remote = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
    argv = [
        "ssh",
        *ssh_base_opts(pod, state_dir=state_dir),
        f"{pod.ssh.user}@{pod.ssh.host}",
        remote,
    ]
    result = _run_with_hostkey_retry(
        runner, argv, timeout=timeout_s, pod=pod, state_dir=state_dir
    )
    out = ExecResult(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if check and result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-800:]
        raise GpuError(f"ssh exec failed (exit {result.returncode}): {tail}")
    return out


def _rsync_ssh_shell(pod: Pod, *, state_dir: Path | None) -> str:
    opts = ssh_base_opts(pod, state_dir=state_dir)
    return "ssh " + " ".join(shlex.quote(o) for o in opts)


def push(
    pod: Pod,
    local_path: Path,
    remote_path: str,
    *,
    excludes: Sequence[str] | None = None,
    timeout_s: float = 1800.0,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
) -> None:
    runner = runner or default_ssh_runner
    local = Path(local_path)
    if not local.exists():
        raise GpuError(f"push source missing: {local}")
    excl = list(excludes if excludes is not None else REPO_RSYNC_EXCLUDES)
    argv: list[str] = ["rsync", "-az"]
    for e in excl:
        argv.extend(["--exclude", e])
    argv.extend(
        [
            "-e",
            _rsync_ssh_shell(pod, state_dir=state_dir),
            str(local) if local.is_file() else f"{local}/",
            f"{pod.ssh.user}@{pod.ssh.host}:{remote_path}",
        ]
    )
    result = _run_with_hostkey_retry(
        runner, argv, timeout=timeout_s, pod=pod, state_dir=state_dir
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-800:]
        raise GpuError(f"rsync push failed: {tail}")


def pull(
    pod: Pod,
    remote_path: str,
    local_path: Path,
    *,
    timeout_s: float = 1800.0,
    runner: SshRunner | None = None,
    state_dir: Path | None = None,
) -> None:
    runner = runner or default_ssh_runner
    local = Path(local_path)
    local.mkdir(parents=True, exist_ok=True)
    argv = [
        "rsync",
        "-az",
        "-e",
        _rsync_ssh_shell(pod, state_dir=state_dir),
        f"{pod.ssh.user}@{pod.ssh.host}:{remote_path}",
        str(local) + "/",
    ]
    result = _run_with_hostkey_retry(
        runner, argv, timeout=timeout_s, pod=pod, state_dir=state_dir
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-800:]
        raise GpuError(f"rsync pull failed: {tail}")
