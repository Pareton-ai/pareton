"""Targon GPU cloud provider using the v2 REST API.

Registers the host's existing SSH key with Targon and uses it for
paramiko exec. Requires TARGON_VOLUME_UID (e.g. vol-xxx) to attach
a persistent volume for model weights.
"""

from __future__ import annotations


import logging
import time
from pathlib import Path
from typing import Any, Generator


import paramiko
import requests

from . import GpuInstance, GpuProvider, PodHandle, lookup_vram
from .ssh_keys import discover_ssh_private_key

logger = logging.getLogger(__name__)

API_BASE = "https://api.targon.com/tha/v2"
TARGON_SSH_HOST = "ssh.deployments.targon.com"
TARGON_DASHBOARD = "https://targon.com/rentals"


def _normalize_gpu_type(raw: str) -> str:
    for prefix in ("NVIDIA-GeForce-", "NVIDIA-"):
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw


class TargonProvider:
    """GpuProvider backed by the Targon v2 REST API."""

    name = "targon"
    READY_TIMEOUT_S = 360

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._ssh_key_uid: str | None = None
        self._ssh_key_path: Path | None = None
        self._ssh_pkey: paramiko.PKey | None = None
        try:
            self._ssh_key_path, self._ssh_pkey = discover_ssh_private_key()
        except RuntimeError:
            pass

    # -- HTTP helpers ------------------------------------------------------

    def _get(self, path: str, **kwargs: Any) -> Any:
        resp = requests.get(
            f"{API_BASE}{path}", headers=self._headers, timeout=30, **kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict | None = None) -> Any:
        resp = requests.post(
            f"{API_BASE}{path}", headers=self._headers, json=json, timeout=30
        )
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def _put(self, path: str, json: dict | None = None) -> Any:
        resp = requests.put(
            f"{API_BASE}{path}", headers=self._headers, json=json, timeout=30
        )
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def _delete(self, path: str) -> None:
        resp = requests.delete(f"{API_BASE}{path}", headers=self._headers, timeout=30)
        resp.raise_for_status()

    def _deploy_workload(self, workload_uid: str) -> None:
        """Deploy with retries; Targon sometimes returns 502 while deploy proceeds."""
        url = f"{API_BASE}/workloads/{workload_uid}/deploy"
        last_resp: requests.Response | None = None
        for attempt in range(1, 4):
            last_resp = requests.post(url, headers=self._headers, timeout=60)
            if last_resp.ok:
                return
            if last_resp.status_code in (502, 503, 504) and attempt < 3:
                logger.warning(
                    "Targon deploy %s returned %s (attempt %d/3), retrying in 5s",
                    workload_uid,
                    last_resp.status_code,
                    attempt,
                )
                time.sleep(5)
                continue
            break

        state = self._get(f"/workloads/{workload_uid}/state")
        status = state.get("status", "").upper()
        if status in ("DEPLOYING", "PROVISIONING", "RUNNING"):
            logger.warning(
                "Targon deploy HTTP failed for %s but state is %s; continuing",
                workload_uid,
                status,
            )
            return

        try:
            self._delete(f"/workloads/{workload_uid}")
        except Exception:
            logger.exception(
                "Failed to delete Targon workload %s after deploy failure",
                workload_uid,
            )
        assert last_resp is not None
        last_resp.raise_for_status()

    # -- SSH key management ------------------------------------------------

    @staticmethod
    def _key_fingerprint(pub_key_line: str) -> str:
        """Extract 'type base64' from a public key line, ignoring the comment."""
        parts = pub_key_line.strip().split()
        return " ".join(parts[:2]) if len(parts) >= 2 else pub_key_line.strip()

    def _ensure_ssh_key(self) -> str:
        """Find or register the host's public key with Targon. Returns the key UID."""
        if self._ssh_key_uid:
            return self._ssh_key_uid

        if not self._ssh_key_path:
            raise RuntimeError("No SSH private key found in ~/.ssh")

        pub_path = self._ssh_key_path.with_name(f"{self._ssh_key_path.name}.pub")
        if not pub_path.exists():
            raise RuntimeError(f"No public key at {pub_path}")
        pub_key = pub_path.read_text().strip()
        local_fp = self._key_fingerprint(pub_key)

        existing = self._get("/ssh-keys")
        for key in existing.get("items", []):
            remote_fp = self._key_fingerprint(key.get("ssh_key", ""))
            if remote_fp == local_fp:
                self._ssh_key_uid = key["uid"]
                logger.info(
                    "SSH key already registered with Targon (uid=%s)", self._ssh_key_uid
                )
                return self._ssh_key_uid

        resp = self._post(
            "/ssh-keys",
            json={"name": "cacheon-cpu-validator", "ssh_key": pub_key},
        )
        self._ssh_key_uid = resp["uid"]
        logger.info("Registered SSH key with Targon: %s", self._ssh_key_uid)
        return self._ssh_key_uid

    # -- GpuProvider interface ---------------------------------------------

    def search(self) -> list[GpuInstance]:
        resp = requests.get(
            f"{API_BASE}/inventory",
            headers={"Authorization": f"Bearer {self._api_key}"},
            params={"type": "rental", "gpu": "true"},
            timeout=30,
        )
        resp.raise_for_status()

        out: list[GpuInstance] = []
        for item in resp.json():
            spec = item.get("spec", {})
            gpu_type_raw = _normalize_gpu_type(spec.get("gpu_type", ""))
            gpu_count = spec.get("gpu_count", 0)
            available = item.get("available", 0)

            if available <= 0:
                continue

            canon, vram = lookup_vram(gpu_type_raw)
            if not vram:
                continue

            storage_mb = spec.get("storage", 0)
            storage_gb = storage_mb // 1024 if storage_mb else 0
            price_dollars = item.get("cost_per_hour", 0)
            price_cents = int(round(price_dollars * 100))
            memory_mb = spec.get("memory", 0)

            out.append(
                GpuInstance(
                    provider="targon",
                    instance_id=item.get("name", ""),
                    description=item.get("display_name", ""),
                    hourly_price_cents=price_cents,
                    num_gpus=gpu_count,
                    gpu_type=canon,
                    vram_per_gpu_gb=vram,
                    total_vram_gb=gpu_count * vram,
                    storage_gb=storage_gb,
                    memory_gb=memory_mb // 1024 if memory_mb else 0,
                    vcpus=spec.get("vcpu", 0),
                    docker_in_docker=True,
                    raw=item,
                )
            )
        return out

    def rent(self, instance: GpuInstance) -> PodHandle:
        from .. import config as validator_config

        resource_name = instance.instance_id
        ssh_key_uid = self._ensure_ssh_key()

        volume_uid = validator_config.TARGON_VOLUME_UID
        if not volume_uid or not volume_uid.startswith("vol-"):
            raise RuntimeError(
                "TARGON_VOLUME_UID is required and must start with 'vol-' "
                "(create one at https://targon.com/volumes)"
            )

        body: dict[str, Any] = {
            "name": "cacheon-eval",
            "image": "ghcr.io/manifold-inc/ubuntu-systemd-docker:v3",
            "resource_name": resource_name,
            "type": "RENTAL",
            "ports": [
                {"port": 22, "protocol": "TCP", "routing": "DIRECT"},
            ],
            "ssh_keys": [ssh_key_uid],
            "volumes": [{"uid": volume_uid, "mount_path": "/workspace"}],
        }

        workload = self._post("/workloads", json=body)
        workload_uid = workload["uid"]

        self._deploy_workload(workload_uid)
        logger.info(
            "Targon workload %s created (resource=%s, volume=%s)",
            workload_uid,
            resource_name,
            volume_uid,
        )
        logger.info("🔗 Dashboard: %s/%s", TARGON_DASHBOARD, workload_uid)

        return PodHandle(
            provider="targon",
            pod_id=workload_uid,
            gpu_count=instance.num_gpus,
            hourly_price_cents=instance.hourly_price_cents,
            raw={"resource_name": resource_name},
        )

    def wait_ready(
        self, handle: PodHandle, timeout_s: int = READY_TIMEOUT_S
    ) -> PodHandle:
        deadline = time.monotonic() + timeout_s
        poll_count = 0
        while time.monotonic() < deadline:
            state = self._get(f"/workloads/{handle.pod_id}/state")
            status = state.get("status", "")
            elapsed = int(time.monotonic() - (deadline - timeout_s))
            poll_count += 1
            if poll_count % 3 == 1:
                logger.info(
                    "⏳ Waiting for workload %s: status=%s (%ds elapsed)",
                    handle.pod_id,
                    status,
                    elapsed,
                )

            if status.upper() == "RUNNING":
                handle.raw["ssh"] = {
                    "host": TARGON_SSH_HOST,
                    "port": 22,
                    "user": handle.pod_id,
                }
                return handle

            if status.upper() in ("FAILED", "ERROR", "TERMINATED"):
                raise RuntimeError(
                    f"Targon workload {handle.pod_id} entered {status}: "
                    f"{state.get('message', '')}"
                )
            time.sleep(10)

        raise TimeoutError(
            f"Targon workload {handle.pod_id} not ready after {timeout_s}s"
        )

    def _load_private_key(self) -> paramiko.PKey:
        if self._ssh_pkey is not None:
            return self._ssh_pkey
        self._ssh_key_path, self._ssh_pkey = discover_ssh_private_key()
        return self._ssh_pkey

    def _ssh_connect(self, handle: PodHandle) -> paramiko.SSHClient:
        """Open an SSH connection, retrying up to 3 times for key propagation."""
        ssh_info = handle.raw.get("ssh", {})
        host = ssh_info.get("host", "")
        port = ssh_info.get("port", 22)
        user = ssh_info.get("user", "root")

        if not host:
            raise RuntimeError(f"No SSH host for workload {handle.pod_id}")

        pkey = self._load_private_key()

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=host, port=port, username=user, pkey=pkey, timeout=30
                )
                if attempt > 1:
                    logger.info("SSH connected on attempt %d", attempt)
                return client
            except paramiko.AuthenticationException:
                client.close()
                if attempt == max_retries:
                    raise
                logger.info(
                    "SSH auth failed (attempt %d/%d), retrying in 15s...",
                    attempt,
                    max_retries,
                )
                time.sleep(15)
            except Exception:
                client.close()
                raise
        raise RuntimeError("unreachable")

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
        try:
            self._delete(f"/workloads/{handle.pod_id}")
            logger.info("Targon workload %s deleted", handle.pod_id)
        except Exception as exc:
            logger.error("Targon teardown failed for %s: %s", handle.pod_id, exc)
