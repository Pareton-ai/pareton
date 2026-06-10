"""SSH into a pre-provisioned GPU pod (setup-gpu.sh already run).

Used when ``CACHEON_GPU_SSH`` is set. Skips marketplace search, rent,
setup, and teardown; only connects and runs the eval compose command.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator

import paramiko

from .. import config as validator_config
from . import PodHandle
from .ssh_keys import discover_ssh_private_key

logger = logging.getLogger(__name__)


class StaticSshProvider:
    """Connect to a fixed SSH target configured via ``CACHEON_GPU_SSH``."""

    name = "static"
    READY_TIMEOUT_S = 120

    def __init__(self) -> None:
        self._ssh_key_path = None
        self._ssh_pkey: paramiko.PKey | None = None
        try:
            self._ssh_key_path, self._ssh_pkey = discover_ssh_private_key()
        except RuntimeError:
            pass

    def connect(self) -> PodHandle:
        target = validator_config.GPU_SSH
        if target is None:
            raise RuntimeError("CACHEON_GPU_SSH is not configured")

        profile = validator_config.GPU_POD_PROFILE
        pod_id = f"{target.user}@{target.host}:{target.port}"
        return PodHandle(
            provider=profile,
            pod_id=pod_id,
            gpu_count=validator_config.GPU_COUNT,
            hourly_price_cents=0,
            raw={
                "ssh": {
                    "host": target.host,
                    "port": target.port,
                    "user": target.user,
                }
            },
        )

    def wait_ready(
        self, handle: PodHandle, timeout_s: int = READY_TIMEOUT_S
    ) -> PodHandle:
        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._ssh_connect(handle).close()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(5)
        else:
            raise TimeoutError(
                f"SSH to {handle.pod_id} not ready after {timeout_s}s: {last_err}"
            ) from last_err

        docker = "sudo -E docker" if handle.provider == "shadeform" else "docker"
        preflight = (
            "nvidia-smi >/dev/null 2>&1 && "
            "test -f ~/cacheon/validator/gpu-compose.yml && "
            f"{docker} info >/dev/null 2>&1"
        )
        result = self.exec(handle, preflight)
        if result.get("exit_code") != 0:
            stderr = result.get("stderr", "")
            stdout = result.get("stdout", "")
            raise RuntimeError(
                f"GPU pod preflight failed on {handle.pod_id}: "
                f"{stderr or stdout or 'unknown error'}"
            )
        return handle

    def _load_private_key(self) -> paramiko.PKey:
        if self._ssh_pkey is not None:
            return self._ssh_pkey
        self._ssh_key_path, self._ssh_pkey = discover_ssh_private_key()
        return self._ssh_pkey

    def _ssh_connect(self, handle: PodHandle) -> paramiko.SSHClient:
        ssh_info = handle.raw.get("ssh", {})
        host = ssh_info.get("host", "")
        port = int(ssh_info.get("port") or 22)
        user = ssh_info.get("user", "root")

        if not host:
            raise RuntimeError(f"No SSH host for pod {handle.pod_id}")

        pkey = self._load_private_key()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=host, port=port, username=user, pkey=pkey, timeout=30)
        return client

    def exec(self, handle: PodHandle, command: str) -> dict[str, Any]:
        client = self._ssh_connect(handle)
        try:
            _stdin, stdout, stderr = client.exec_command(command, timeout=10800)
            exit_code = stdout.channel.recv_exit_status()
            return {
                "stdout": stdout.read().decode("utf-8", errors="replace"),
                "stderr": stderr.read().decode("utf-8", errors="replace"),
                "exit_code": exit_code,
                "success": exit_code == 0,
            }
        finally:
            client.close()

    def stream_exec(
        self, handle: PodHandle, command: str
    ) -> Generator[dict[str, str], None, None]:
        client = self._ssh_connect(handle)
        try:
            _stdin, stdout, _stderr = client.exec_command(command, timeout=10800)
            for line in iter(stdout.readline, ""):
                yield {"type": "stdout", "data": line}
        finally:
            client.close()

    def teardown(self, handle: PodHandle) -> None:
        """No-op: pre-provisioned pods are not destroyed by the orchestrator."""
