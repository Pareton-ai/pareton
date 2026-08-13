"""Environment fingerprinting — degrades gracefully on machines without GPU/Docker."""

from __future__ import annotations

import hashlib
import logging
import platform
import re
import shutil
import socket
import subprocess
from typing import Any

from bench import __version__
from bench.schemas import EnvironmentInfo, GpuInfo

logger = logging.getLogger(__name__)


def _run(cmd: list[str], *, timeout: float = 5.0) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("env probe %s failed: %s", cmd, exc)
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _hostname_hash() -> str:
    host = socket.gethostname() or "unknown"
    digest = hashlib.sha256(host.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _parse_nvidia_smi_query(text: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=index,name,vbios_version,memory.total --format=csv,noheader,nounits`."""
    gpus: list[GpuInfo] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            memory_mb = int(float(parts[3]))
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=parts[1],
                vbios=parts[2],
                memory_mb=memory_mb,
            )
        )
    return gpus


def _driver_version() -> str:
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if out:
        return out.strip().splitlines()[0].strip()
    return ""


def _cuda_version() -> str:
    out = _run(["nvcc", "--version"])
    if out:
        m = re.search(r"release\s+([0-9.]+)", out)
        if m:
            return m.group(1)
    # Fall back to nvidia-smi CUDA Version line
    out = _run(["nvidia-smi"])
    if out:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        if m:
            return m.group(1)
    return ""


def _docker_version() -> str:
    out = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    if out and out.strip():
        return out.strip()
    out = _run(["docker", "--version"])
    if out:
        m = re.search(r"version\s+([^\s,]+)", out)
        if m:
            return m.group(1)
    return ""


def collect_environment(*, harness_version: str | None = None) -> EnvironmentInfo:
    """Probe GPU/driver/CUDA/Docker. Empty strings / empty gpu list when unavailable."""
    gpus: list[GpuInfo] = []
    gpu_out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,vbios_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_out:
        gpus = _parse_nvidia_smi_query(gpu_out)

    return EnvironmentInfo(
        gpu=gpus,
        driver_version=_driver_version(),
        cuda_version=_cuda_version(),
        docker_version=_docker_version(),
        harness_version=harness_version or __version__,
        hostname_hash=_hostname_hash(),
    )


def collect_env_raw_dumps() -> dict[str, str]:
    """Raw probe text for evidence/env/ (may be empty on no-GPU hosts)."""
    dumps: dict[str, str] = {
        "platform": f"{platform.platform()}\n{platform.python_version()}\n",
    }
    nvsmi = _run(["nvidia-smi", "-q"], timeout=15.0)
    if nvsmi:
        dumps["nvidia-smi-q.txt"] = nvsmi
    docker_info = _run(["docker", "info"], timeout=15.0)
    if docker_info:
        dumps["docker-info.txt"] = docker_info
    return dumps


def _sku_compact(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")


def warn_gpu_sku_mismatch(env: EnvironmentInfo, expected: str) -> str | None:
    """Return a warning string if expected SKU not found among probed GPUs."""
    if not expected:
        return None
    if not env.gpu:
        return f"gpu_sku_expected={expected!r} but no GPUs detected"
    names = " ".join(g.name for g in env.gpu)
    # Loose match: ignore spaces/hyphens so RTX5090 matches "NVIDIA GeForce RTX 5090".
    names_c = _sku_compact(names)
    expected_c = _sku_compact(expected.replace("NVIDIA-", ""))
    if expected_c not in names_c and expected.lower() not in names.lower():
        return f"gpu_sku_expected={expected!r} not found in detected GPUs: {names!r}"
    return None


def environment_to_probe_dict(env: EnvironmentInfo) -> dict[str, Any]:
    return env.to_dict()
