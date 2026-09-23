"""Unit tests for the build-pressure sampler. No host, Docker, or network."""

from __future__ import annotations

import importlib.util
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
    assert report["placement"] == "mixed"
    assert report["cgroup_classes"] == {"docker.scope": 1, "docker.service": 1}
    assert report["peak_mem_used_mib"] == 800.0
    assert report["peak_swap_used_mib"] == 50.0
    assert report["peak_disk_used_percent"] == 70.5
    assert report["peak_load1"] == 4.0
    assert report["peak_docker_memory_mib"] == 40.0
    assert report["peak_api_memory_mib"] == 30.0


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
