"""Unit tests for the build-pressure sampler. No host, Docker, or network."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS = REPO_ROOT / "ops"


def _load():
    spec = importlib.util.spec_from_file_location(
        "measure_build", OPS / "measure_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure = _load()


@pytest.mark.unit
def test_classify_cgroup_distinguishes_scope_from_docker_service():
    assert (
        measure.classify_cgroup("0::/system.slice/docker-abc.scope") == "docker.scope"
    )
    assert (
        measure.classify_cgroup("/system.slice/docker.service/buildkit")
        == "docker.service"
    )
    assert (
        measure.classify_cgroup("/system.slice/pareton-worker.service")
        == "pareton-worker.service"
    )
    assert measure.classify_cgroup("/system.slice/pareton-api.service") == "other"
    assert (
        measure.classify_cgroup(
            "0::/system.slice/system.slice:docker:qhd7gutph7rj1778rtbkomeeo"
        )
        == "docker.container"
    )
    assert (
        measure.is_build_cmdline(b"runc --log /var/lib/docker/buildkit/executor/x")
        is False
    )
    assert measure.is_build_cmdline(b"docker buildx build --builder default") is True


@pytest.mark.unit
def test_parse_meminfo_and_df():
    memory = measure.parse_meminfo(
        "MemTotal: 1000 kB\nMemAvailable: 400 kB\nSwapTotal: 200 kB\nSwapFree: 50 kB\n"
    )
    assert memory["mem_used_bytes"] == 600 * 1024
    assert memory["swap_used_bytes"] == 150 * 1024
    assert (
        measure.parse_df(
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/vda1 100 76 24 76% /\n"
        )
        == 76.0
    )
    assert measure.parse_bytes("[not set]") is None
    assert measure.parse_bytes("4096") == 4096


@pytest.mark.unit
def test_cgroup_names_include_docker_service_children(tmp_path):
    root = tmp_path
    (root / "system.slice" / "docker.service" / "child").mkdir(parents=True)
    (root / "system.slice" / "docker-abc.scope").mkdir()
    (root / "system.slice" / "pareton-api.service").mkdir()
    assert measure.cgroup_names(root) == {
        "system.slice/docker.service",
        "system.slice/docker.service/child",
        "system.slice/docker-abc.scope",
        "system.slice/pareton-api.service",
    }


@pytest.mark.unit
def test_summarize_reports_scope_placement_and_peaks():
    report = measure.summarize(
        [
            {
                "mem_used_bytes": 100 * 1048576,
                "swap_used_bytes": 0,
                "disk_used_percent": 68.0,
                "load1": 0.2,
                "docker_memory_bytes": 10 * 1048576,
                "worker_memory_bytes": 20 * 1048576,
                "api_memory_bytes": 30 * 1048576,
                "new_cgroups": [],
                "build_process_cgroups": [],
            },
            {
                "mem_used_bytes": 800 * 1048576,
                "swap_used_bytes": 50 * 1048576,
                "disk_used_percent": 70.5,
                "load1": 4.0,
                "docker_memory_bytes": 40 * 1048576,
                "worker_memory_bytes": 20 * 1048576,
                "api_memory_bytes": 30 * 1048576,
                "new_cgroups": ["system.slice/docker-abc.scope"],
                "build_process_cgroups": ["/system.slice/docker.service"],
            },
        ]
    )
    assert report["placement"] == "docker.scope"
    assert report["cgroup_classes"] == {"docker.scope": 1, "docker.service": 1}
    assert report["peak_mem_used_mib"] == 800.0
    assert report["peak_swap_used_mib"] == 50.0
    assert report["peak_disk_used_percent"] == 70.5
    assert report["peak_load1"] == 4.0
    assert report["peak_docker_memory_mib"] == 40.0
    assert report["peak_api_memory_mib"] == 30.0
    assert report["peak_workload_memory_mib"] is None


@pytest.mark.unit
def test_systemd_container_cgroup_wins_over_runc_in_docker_service():
    report = measure.summarize(
        [
            {
                "new_cgroups": [
                    "system.slice/system.slice:docker:qhd7gutph7rj1778rtbkomeeo"
                ],
                "build_process_cgroups": ["/system.slice/docker.service"],
            }
        ]
    )
    assert report["placement"] == "docker.container"
    assert report["cgroup_classes"]["docker.service"] == 1


@pytest.mark.unit
def test_sample_until_waits_for_a_build_then_stops_after_quiet():
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    plan = [
        {"new_cgroups": [], "build_process_cgroups": []},
        {
            "new_cgroups": ["system.slice/docker.service/run"],
            "build_process_cgroups": [],
        },
        {"new_cgroups": [], "build_process_cgroups": []},
        {"new_cgroups": [], "build_process_cgroups": []},
        {"new_cgroups": [], "build_process_cgroups": []},
    ]
    calls = {"n": 0}

    def collect():
        item = dict(plan[min(calls["n"], len(plan) - 1)])
        calls["n"] += 1
        return item

    samples = measure.sample_until(
        interval=1,
        max_seconds=100,
        quiet_seconds=2,
        wait_for_build=True,
        sleep=sleep,
        now=now,
        collect=collect,
    )
    assert len(samples) == 5
    assert measure.summarize(samples)["placement"] == "docker.service"


@pytest.mark.unit
def test_sample_until_ignores_unrelated_units_while_waiting(tmp_path):
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    plan = [
        {
            "new_cgroups": ["system.slice/pareton-deploy.service"],
            "build_process_cgroups": [],
        },
        {"new_cgroups": [], "build_process_cgroups": []},
        {
            "new_cgroups": ["system.slice/system.slice:docker:abc"],
            "build_process_cgroups": [],
            "workload_memory_bytes": 3 * 1048576,
        },
        {"new_cgroups": [], "build_process_cgroups": []},
    ]
    calls = {"n": 0}

    def collect():
        item = dict(plan[min(calls["n"], len(plan) - 1)])
        calls["n"] += 1
        return item

    samples = measure.sample_until(
        interval=1,
        max_seconds=3,
        quiet_seconds=5,
        wait_for_build=True,
        sleep=sleep,
        now=now,
        collect=collect,
    )
    assert [sample.get("new_cgroups") for sample in samples][-2:] == [
        ["system.slice/system.slice:docker:abc"],
        [],
    ]
    assert measure.summarize(samples)["placement"] == "docker.container"
    assert measure.summarize(samples)["peak_workload_memory_mib"] == 3.0

    root = tmp_path
    container = root / "system.slice" / "system.slice:docker:abc"
    container.mkdir(parents=True)
    (container / "memory.current").write_text("1048576\n")
    (root / "system.slice" / "docker.service").mkdir()
    assert (
        measure.workload_memory_bytes(
            root,
            [
                "system.slice/system.slice:docker:abc",
                "system.slice/docker.service",
            ],
        )
        == 1048576
    )


@pytest.mark.unit
def test_lasting_docker_scope_does_not_hold_the_sampler_or_inflate_counts():
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    scope = "system.slice/docker-deploy.scope"
    calls = {"n": 0}

    def collect():
        calls["n"] += 1
        return {
            "new_cgroups": [scope],
            "build_process_cgroups": ["/system.slice/docker.service"]
            if calls["n"] == 1
            else [],
        }

    samples = measure.sample_until(
        interval=1,
        max_seconds=30,
        quiet_seconds=2,
        wait_for_build=True,
        sleep=sleep,
        now=now,
        collect=collect,
    )
    assert len(samples) < 10
    report = measure.summarize(samples)
    assert report["cgroup_classes"] == {"docker.scope": 1, "docker.service": 1}
    assert report["placement"] == "docker.scope"


@pytest.mark.unit
def test_build_container_stays_busy_and_scope_memory_does_not_replace_it(tmp_path):
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    container = "system.slice/system.slice:docker:build"
    scope = "system.slice/docker-deploy.scope"

    def collect():
        return {"new_cgroups": [container, scope], "build_process_cgroups": []}

    samples = measure.sample_until(
        interval=1,
        max_seconds=3,
        quiet_seconds=1,
        wait_for_build=True,
        sleep=sleep,
        now=now,
        collect=collect,
    )
    assert len(samples) == 4
    report = measure.summarize(samples)
    assert report["placement"] == "docker.container"
    assert report["cgroup_classes"]["docker.container"] == 1
    assert report["cgroup_classes"]["docker.scope"] == 1

    root = tmp_path
    (root / container).mkdir(parents=True)
    (root / container / "memory.current").write_text("2097152\n")
    (root / scope).mkdir(parents=True)
    (root / scope / "memory.current").write_text(str(8 * 1048576) + "\n")
    assert measure.workload_memory_bytes(root, [container, scope]) == 2097152


@pytest.mark.unit
def test_command_failures_leave_the_sample_fields_empty(monkeypatch):
    def boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=10)

    monkeypatch.setattr(measure.subprocess, "run", boom)
    assert measure._systemctl_show("docker.service") == {}
    assert measure._df_root() is None
