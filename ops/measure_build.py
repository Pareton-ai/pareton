"""Sample host pressure and where a Docker build is charged.

Start this on the validator before a build. It records memory, swap, disk,
load, and any cgroup that appears after startup, then stops once the build
has been quiet. It does not start a build.

    python3 ops/measure_build.py --out /tmp/pareton-build-measure.json

A path is classified as:

- ``docker.container`` when systemd names it ``system.slice:docker:<id>``
- ``docker.scope`` when it is a ``docker-*.scope`` sibling under ``system.slice``
- ``pareton-worker.service`` when the worker unit owns it
- ``docker.service`` when it sits inside that unit

A container cgroup wins over ``docker.service``. ``runc`` stays inside
``docker.service`` even while the RUN step it started does not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_UNSET = {"", "[not set]", "[no data]", "infinity", "n/a"}
_WORKLOAD_CLASSES = ("docker.container", "docker.scope")


def is_build_cmdline(cmdline: bytes) -> bool:
    """True for buildx or buildkitd. False for runc, whose path mentions buildkit."""
    text = cmdline.replace(b"\0", b" ")
    if text.startswith(b"runc") or b"/runc " in text or b" runc " in text:
        return False
    return b"buildkitd" in text or b"buildx" in text


def classify_cgroup(path: str) -> str:
    """Return the placement class for one cgroup path."""
    text = path.strip().removeprefix("0::")
    if "pareton-worker.service" in text:
        return "pareton-worker.service"
    if "docker.service" in text:
        return "docker.service"
    if ".scope" in text and "docker-" in text:
        return "docker.scope"
    if ":docker:" in text:
        return "docker.container"
    return "other"


def parse_meminfo(text: str) -> dict[str, int | None]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            amount = int(parts[0])
        except ValueError:
            continue
        if len(parts) > 1 and parts[1] == "kB":
            amount *= 1024
        values[key] = amount
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    return {
        "mem_used_bytes": None
        if total is None or available is None
        else total - available,
        "swap_used_bytes": None
        if swap_total is None or swap_free is None
        else swap_total - swap_free,
    }


def parse_bytes(value: str | None) -> int | None:
    if value is None or value.strip() in _UNSET:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def parse_df(text: str) -> float | None:
    """Parse ``df -P`` output and return the used percent of the last row."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if len(rows) < 2:
        return None
    token = rows[-1][-2] if len(rows[-1]) >= 2 else ""
    if not token.endswith("%"):
        return None
    try:
        return float(token[:-1])
    except ValueError:
        return None


def cgroup_names(cgroup_root: Path) -> set[str]:
    """Relative cgroup paths under system.slice, including docker.service children."""
    system = cgroup_root / "system.slice"
    if not system.is_dir():
        return set()
    names: set[str] = set()
    for child in system.iterdir():
        if not child.is_dir():
            continue
        names.add(child.relative_to(cgroup_root).as_posix())
        if child.name != "docker.service":
            continue
        for grand in child.iterdir():
            if grand.is_dir():
                names.add(grand.relative_to(cgroup_root).as_posix())
    return names


def placement_of(paths: list[str]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for path in paths:
        kind = classify_cgroup(path)
        if kind == "other":
            continue
        counts[kind] = counts.get(kind, 0) + 1
    workload = [kind for kind in _WORKLOAD_CLASSES if counts.get(kind)]
    if len(workload) == 1:
        return workload[0], counts
    if len(workload) > 1:
        return "mixed", counts
    kinds = [kind for kind, count in counts.items() if count]
    if not kinds:
        return "none", counts
    if len(kinds) == 1:
        return kinds[0], counts
    return "mixed", counts


def _peak(samples: list[dict], key: str) -> float | None:
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    return float(max(values))


def _mib(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1048576, 1)


def summarize(samples: list[dict]) -> dict:
    paths: list[str] = []
    for sample in samples:
        paths.extend(sample.get("new_cgroups") or [])
        paths.extend(sample.get("build_process_cgroups") or [])
    placement, counts = placement_of(paths)
    return {
        "sample_count": len(samples),
        "placement": placement,
        "cgroup_classes": counts,
        "peak_mem_used_mib": _mib(_peak(samples, "mem_used_bytes")),
        "peak_swap_used_mib": _mib(_peak(samples, "swap_used_bytes")),
        "peak_disk_used_percent": _peak(samples, "disk_used_percent"),
        "peak_load1": _peak(samples, "load1"),
        "peak_docker_memory_mib": _mib(_peak(samples, "docker_memory_bytes")),
        "peak_worker_memory_mib": _mib(_peak(samples, "worker_memory_bytes")),
        "peak_api_memory_mib": _mib(_peak(samples, "api_memory_bytes")),
    }


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _proc_cgroup(pid_dir: Path) -> str:
    text = _read(pid_dir / "cgroup")
    line = text.splitlines()[-1] if text else ""
    if "::" in line:
        return line.split("::", 1)[1]
    return line


def build_process_cgroups(proc_root: Path) -> list[str]:
    found: list[str] = []
    if not proc_root.is_dir():
        return found
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmdline = (pid_dir / "cmdline").read_bytes()
        except OSError:
            continue
        if not is_build_cmdline(cmdline):
            continue
        cgroup = _proc_cgroup(pid_dir)
        if cgroup:
            found.append(cgroup)
    return found


def _systemctl_show(unit: str) -> dict[str, str]:
    import subprocess

    proc = subprocess.run(
        ["systemctl", "show", unit, "-p", "MemoryCurrent", "-p", "ControlGroup"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def _load1() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _df_root() -> float | None:
    import subprocess

    proc = subprocess.run(
        ["df", "-P", "/"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return parse_df(proc.stdout)


def collect_sample(
    *,
    baseline: set[str],
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
) -> dict:
    memory = parse_meminfo(_read(Path("/proc/meminfo")))
    current = cgroup_names(cgroup_root)
    docker = _systemctl_show("docker.service")
    worker = _systemctl_show("pareton-worker.service")
    api = _systemctl_show("pareton-api.service")
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mem_used_bytes": memory["mem_used_bytes"],
        "swap_used_bytes": memory["swap_used_bytes"],
        "disk_used_percent": _df_root(),
        "load1": _load1(),
        "docker_memory_bytes": parse_bytes(docker.get("MemoryCurrent")),
        "worker_memory_bytes": parse_bytes(worker.get("MemoryCurrent")),
        "api_memory_bytes": parse_bytes(api.get("MemoryCurrent")),
        "new_cgroups": sorted(current - baseline),
        "build_process_cgroups": build_process_cgroups(proc_root),
    }


def sample_until(
    *,
    interval: float,
    max_seconds: float,
    quiet_seconds: float,
    wait_for_build: bool,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
    sleep=time.sleep,
    now=time.monotonic,
    collect=None,
) -> list[dict]:
    """Collect samples. Stop after the build stays quiet, or at max_seconds."""
    baseline = cgroup_names(cgroup_root)
    collect_fn = collect or (
        lambda: collect_sample(
            baseline=baseline, cgroup_root=cgroup_root, proc_root=proc_root
        )
    )
    started = now()
    seen_build = False
    quiet_since: float | None = None
    samples: list[dict] = []
    while now() - started <= max_seconds:
        sample = collect_fn()
        samples.append(sample)
        busy = bool(sample.get("new_cgroups") or sample.get("build_process_cgroups"))
        if busy:
            seen_build = True
            quiet_since = None
        elif seen_build or not wait_for_build:
            if quiet_since is None:
                quiet_since = now()
            elif now() - quiet_since >= quiet_seconds:
                break
        remaining = max_seconds - (now() - started)
        if remaining <= 0:
            break
        sleep(min(interval, remaining))
    return samples


def _print_summary(report: dict) -> None:
    print(f"placement: {report['placement']}")
    print(f"samples: {report['sample_count']}")
    print(f"cgroup_classes: {json.dumps(report['cgroup_classes'], sort_keys=True)}")
    for key in (
        "peak_mem_used_mib",
        "peak_swap_used_mib",
        "peak_disk_used_percent",
        "peak_load1",
        "peak_docker_memory_mib",
        "peak_worker_memory_mib",
        "peak_api_memory_mib",
    ):
        print(f"{key}: {report[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--quiet-seconds", type=float, default=30.0)
    parser.add_argument(
        "--wait-for-build",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.max_seconds <= 0 or args.quiet_seconds < 0:
        print("interval and max-seconds must be positive", file=sys.stderr)
        return 2
    samples = sample_until(
        interval=args.interval,
        max_seconds=args.max_seconds,
        quiet_seconds=args.quiet_seconds,
        wait_for_build=args.wait_for_build,
    )
    report = summarize(samples)
    report["samples"] = samples
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
