"""Environment fingerprinting — degrades gracefully on machines without GPU/Docker."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
from typing import Any

from bench import __version__
from bench.schemas import CpuInfo, EnvironmentInfo, GpuInfo

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


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _cpu_model() -> str:
    """First "model name" in /proc/cpuinfo, else whatever platform reports."""
    info = _read_text("/proc/cpuinfo")
    if info:
        for line in info.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in ("model name", "Model name", "Processor"):
                name = value.strip()
                if name:
                    return name
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if out and out.strip():
        return out.strip()
    return platform.processor() or ""


def _available_cores(logical: int) -> int:
    """Cores this process may run on. Lower than `logical` under a cpuset."""
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is None:
        return logical
    try:
        return len(getaffinity(0)) or logical
    except OSError:
        return logical


def _parse_cgroup_membership(text: str) -> tuple[str | None, str | None]:
    """Relative paths of this process in the v2 and v1-cpu hierarchies.

    `/proc/self/cgroup` records membership, not the files under `/sys/fs/cgroup`
    itself. A nested slice such as `/user.slice/bench.scope` is a different
    cgroup from the mount root, and that is where a pod quota usually lives.
    """
    v2_rel: str | None = None
    v1_rel: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # v2: `0::/user.slice/bench.scope`
        if line.startswith("0::"):
            v2_rel = line[3:] or "/"
            continue
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = parts[1].split(",")
        if "cpu" in controllers:
            v1_rel = parts[2] or "/"
    return v2_rel, v1_rel


def _parse_cgroup_mounts(
    text: str,
) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """cgroup2 and v1-cpu mounts from `/proc/self/mountinfo`.

    Field 4 is the root within the filesystem; field 5 is the mount point.
    A subtree mount has a root other than `/`, so membership cannot be
    appended to the mount point until that prefix is stripped.
    """
    v2_mount: tuple[str, str] | None = None
    v1_mount: tuple[str, str] | None = None
    for line in text.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        # mountinfo fields 1-5 are stable; optional fields sit after them.
        if len(left_fields) < 5 or not right_fields:
            continue
        mount_root = left_fields[3]
        mount_point = left_fields[4]
        fstype = right_fields[0]
        super_opts = right_fields[2].split(",") if len(right_fields) > 2 else []
        if fstype == "cgroup2" and v2_mount is None:
            v2_mount = (mount_root, mount_point)
        elif fstype == "cgroup" and "cpu" in super_opts and v1_mount is None:
            v1_mount = (mount_root, mount_point)
    return v2_mount, v1_mount


def _normalize_cgroup_path(path: str) -> str:
    if not path or path == "/":
        return "/"
    return "/" + "/".join(p for p in path.split("/") if p)


def _relpath_in_mount(membership: str, mount_root: str) -> str | None:
    """Membership relative to the mount root, or None if it is outside."""
    mem = _normalize_cgroup_path(membership)
    root = _normalize_cgroup_path(mount_root)
    if root == "/":
        return mem
    if mem == root:
        return "/"
    prefix = root + "/"
    if mem.startswith(prefix):
        return mem[len(root) :]
    return None


def _cgroup_dirs(mount: str, relpath: str) -> list[str]:
    """Directories from the leaf cgroup up to the mount point, inclusive."""
    parts = [p for p in relpath.strip("/").split("/") if p]
    dirs = [os.path.join(mount, *parts[:i]) for i in range(len(parts), 0, -1)]
    dirs.append(mount)
    return dirs


def _parse_v2_cpu_max(text: str) -> float | None:
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    try:
        quota, period = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if quota > 0 and period > 0:
        return quota / period
    return None


def _parse_v1_cpu_quota(quota_text: str, period_text: str) -> float | None:
    try:
        quota, period = int(quota_text.strip()), int(period_text.strip())
    except ValueError:
        return None
    # -1 is the kernel's "no limit"; a zero period would be a divide by zero.
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _tightest_quota(quotas: list[float | None]) -> float | None:
    finite = [q for q in quotas if q is not None]
    return min(finite) if finite else None


def _v2_quota_along(mount: str, relpath: str) -> tuple[bool, float | None]:
    """Walk `cpu.max` from leaf to root. `found` is True if any file existed."""
    found = False
    quotas: list[float | None] = []
    for directory in _cgroup_dirs(mount, relpath):
        raw = _read_text(os.path.join(directory, "cpu.max"))
        if raw is None:
            continue
        found = True
        quotas.append(_parse_v2_cpu_max(raw))
    return found, _tightest_quota(quotas)


def _v1_quota_along(mount: str, relpath: str) -> float | None:
    quotas: list[float | None] = []
    for directory in _cgroup_dirs(mount, relpath):
        raw_quota = _read_text(os.path.join(directory, "cpu.cfs_quota_us"))
        raw_period = _read_text(os.path.join(directory, "cpu.cfs_period_us"))
        if not raw_quota or not raw_period:
            continue
        quotas.append(_parse_v1_cpu_quota(raw_quota, raw_period))
    return _tightest_quota(quotas)


def _cgroup_quota_cores() -> float | None:
    """The cgroup CPU ceiling in whole cores. None means uncapped.

    Checked because "how many cores does the box have" and "how many may the
    engine use" are different questions, and only the second one bounds a
    score. Membership and the mounted hierarchy come from `/proc`; a nested
    cgroup's quota is the tightest `cpu.max` (v2) or CFS quota (v1) on the
    walk from that cgroup to the mount point. Membership is resolved relative
    to the mount root (mountinfo field 4), not appended onto the mount point.
    cgroup v2 first, then v1.
    """
    v2_rel, v1_rel = _parse_cgroup_membership(_read_text("/proc/self/cgroup") or "")
    v2_mount, v1_mount = _parse_cgroup_mounts(_read_text("/proc/self/mountinfo") or "")
    v2_root, v2_point = v2_mount or ("/", "/sys/fs/cgroup")
    v2_inside = _relpath_in_mount(v2_rel or "/", v2_root)
    if v2_inside is not None:
        found_v2, v2_quota = _v2_quota_along(v2_point, v2_inside)
        if found_v2:
            return v2_quota
    v1_root, v1_point = v1_mount or ("/", "/sys/fs/cgroup/cpu")
    v1_inside = _relpath_in_mount(v1_rel or "/", v1_root)
    if v1_inside is not None:
        return _v1_quota_along(v1_point, v1_inside)
    return None


def _memory_total_mb() -> int:
    info = _read_text("/proc/meminfo")
    if info:
        for line in info.splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) // 1024
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return 0
    if pages < 0 or page_size < 0:
        return 0
    return (pages * page_size) // (1024 * 1024)


def collect_cpu() -> CpuInfo:
    """Probe the host CPU. Zeroes and empty strings where a probe is unavailable."""
    logical = os.cpu_count() or 0
    return CpuInfo(
        model=_cpu_model(),
        logical_cores=logical,
        available_cores=_available_cores(logical),
        memory_total_mb=_memory_total_mb(),
        quota_cores=_cgroup_quota_cores(),
    )


def collect_environment(*, harness_version: str | None = None) -> EnvironmentInfo:
    """Probe GPU/driver/CUDA/Docker/CPU. Empty values when a probe is unavailable."""
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
        cpu=collect_cpu(),
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
    lscpu = _run(["lscpu"], timeout=15.0)
    if lscpu:
        dumps["lscpu.txt"] = lscpu
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
