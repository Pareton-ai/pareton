"""Unit tests for the environment fingerprint, CPU half."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from bench import env
from bench.schemas import CpuInfo, EnvironmentInfo

CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) Platinum 8480+
cpu MHz\t\t: 2000.000

processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8480+
"""


def _fake_reads(mapping: dict[str, str]):
    return lambda path: mapping.get(path)


def test_cpu_model_comes_from_proc_cpuinfo(monkeypatch):
    monkeypatch.setattr(env, "_read_text", _fake_reads({"/proc/cpuinfo": CPUINFO}))
    assert env._cpu_model() == "Intel(R) Xeon(R) Platinum 8480+"


def test_cpu_model_falls_back_when_proc_is_absent(monkeypatch):
    """Not every host is Linux, and a missing probe must not raise."""
    monkeypatch.setattr(env, "_read_text", _fake_reads({}))
    monkeypatch.setattr(env, "_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(env.platform, "processor", lambda: "")
    assert env._cpu_model() == ""


def test_memory_total_is_read_in_mb(monkeypatch):
    monkeypatch.setattr(
        env,
        "_read_text",
        _fake_reads({"/proc/meminfo": "MemTotal:       32899072 kB\nMemFree: 12 kB\n"}),
    )
    assert env._memory_total_mb() == 32128


@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [
        # 6 cores out of a 100ms period: the shape a small bench pod has.
        ("600000 100000\n", 6.0),
        ("50000 100000\n", 0.5),
        # "max" is the kernel's uncapped sentinel, not a quota of zero.
        ("max 100000\n", None),
        ("garbage 100000\n", None),
    ],
)
def test_cgroup_v2_quota(monkeypatch, cpu_max: str, expected: float | None):
    monkeypatch.setattr(
        env, "_read_text", _fake_reads({"/sys/fs/cgroup/cpu.max": cpu_max})
    )
    assert env._cgroup_quota_cores() == expected


@pytest.mark.parametrize(
    ("quota", "period", "expected"),
    [
        ("800000\n", "100000\n", 8.0),
        # -1 is how cgroup v1 spells "no limit".
        ("-1\n", "100000\n", None),
        ("800000\n", "0\n", None),
    ],
)
def test_cgroup_v1_quota(monkeypatch, quota: str, period: str, expected: float | None):
    monkeypatch.setattr(
        env,
        "_read_text",
        _fake_reads(
            {
                "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": quota,
                "/sys/fs/cgroup/cpu/cpu.cfs_period_us": period,
            }
        ),
    )
    assert env._cgroup_quota_cores() == expected


def test_no_cgroup_files_means_uncapped(monkeypatch):
    monkeypatch.setattr(env, "_read_text", _fake_reads({}))
    assert env._cgroup_quota_cores() is None


MOUNTINFO_V2 = (
    "36 1 0:31 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:10 "
    "- cgroup2 cgroup2 rw,nsdelegate\n"
)
CGROUP_NESTED = "0::/user.slice/bench.scope\n"


def test_nested_cgroup_v2_quota_follows_membership(monkeypatch):
    """A pod quota lives on the leaf cgroup, not the mount root."""
    monkeypatch.setattr(
        env,
        "_read_text",
        _fake_reads(
            {
                "/proc/self/cgroup": CGROUP_NESTED,
                "/proc/self/mountinfo": MOUNTINFO_V2,
                "/sys/fs/cgroup/cpu.max": "max 100000\n",
                "/sys/fs/cgroup/user.slice/cpu.max": "max 100000\n",
                "/sys/fs/cgroup/user.slice/bench.scope/cpu.max": "200000 100000\n",
            }
        ),
    )
    assert env._cgroup_quota_cores() == 2.0


def test_nested_cgroup_v2_quota_uses_the_tightest_ancestor(monkeypatch):
    monkeypatch.setattr(
        env,
        "_read_text",
        _fake_reads(
            {
                "/proc/self/cgroup": CGROUP_NESTED,
                "/proc/self/mountinfo": MOUNTINFO_V2,
                "/sys/fs/cgroup/cpu.max": "max 100000\n",
                "/sys/fs/cgroup/user.slice/cpu.max": "200000 100000\n",
                "/sys/fs/cgroup/user.slice/bench.scope/cpu.max": "max 100000\n",
            }
        ),
    )
    assert env._cgroup_quota_cores() == 2.0


def test_cgroup_v2_quota_uses_the_mounted_hierarchy(monkeypatch):
    monkeypatch.setattr(
        env,
        "_read_text",
        _fake_reads(
            {
                "/proc/self/cgroup": CGROUP_NESTED,
                "/proc/self/mountinfo": (
                    "36 1 0:31 / /sys/fs/cgroup/unified rw,relatime "
                    "- cgroup2 cgroup2 rw\n"
                ),
                "/sys/fs/cgroup/cpu.max": "max 100000\n",
                "/sys/fs/cgroup/unified/user.slice/bench.scope/cpu.max": (
                    "200000 100000\n"
                ),
            }
        ),
    )
    assert env._cgroup_quota_cores() == 2.0


def test_cgroup_v2_subtree_mount_strips_the_mount_root(monkeypatch):
    """Field 4 is the root inside the filesystem; membership is relative to it."""
    monkeypatch.setattr(
        env,
        "_read_text",
        _fake_reads(
            {
                "/proc/self/cgroup": "0::/tenant/bench\n",
                "/proc/self/mountinfo": (
                    "36 1 0:31 /tenant /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n"
                ),
                "/sys/fs/cgroup/cpu.max": "max 100000\n",
                "/sys/fs/cgroup/bench/cpu.max": "200000 100000\n",
                # Wrong join of mount point + full membership; must not win.
                "/sys/fs/cgroup/tenant/bench/cpu.max": "999000 100000\n",
            }
        ),
    )
    assert env._cgroup_quota_cores() == 2.0


def test_cgroup_v1_subtree_mount_strips_the_mount_root(monkeypatch):
    monkeypatch.setattr(
        env,
        "_read_text",
        _fake_reads(
            {
                "/proc/self/cgroup": "2:cpu,cpuacct:/tenant/bench\n",
                "/proc/self/mountinfo": (
                    "36 1 0:31 /tenant /sys/fs/cgroup/cpu rw "
                    "- cgroup cgroup rw,cpu,cpuacct\n"
                ),
                "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1\n",
                "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
                "/sys/fs/cgroup/cpu/bench/cpu.cfs_quota_us": "200000\n",
                "/sys/fs/cgroup/cpu/bench/cpu.cfs_period_us": "100000\n",
                "/sys/fs/cgroup/cpu/tenant/bench/cpu.cfs_quota_us": "999000\n",
                "/sys/fs/cgroup/cpu/tenant/bench/cpu.cfs_period_us": "100000\n",
            }
        ),
    )
    assert env._cgroup_quota_cores() == 2.0


def test_collect_cpu_reports_cores_and_quota(monkeypatch):
    monkeypatch.setattr(
        env,
        "_read_text",
        _fake_reads(
            {
                "/proc/cpuinfo": CPUINFO,
                "/proc/meminfo": "MemTotal:       32899072 kB\n",
                "/sys/fs/cgroup/cpu.max": "600000 100000\n",
            }
        ),
    )
    monkeypatch.setattr(env.os, "cpu_count", lambda: 224)
    monkeypatch.setattr(env, "_available_cores", lambda logical: logical)

    cpu = env.collect_cpu()
    assert cpu.model == "Intel(R) Xeon(R) Platinum 8480+"
    assert cpu.logical_cores == 224
    # The score-relevant number: the box is large, the ceiling is not.
    assert cpu.quota_cores == 6.0
    assert cpu.memory_total_mb == 32128


def test_collect_cpu_survives_a_host_with_no_probes(monkeypatch):
    monkeypatch.setattr(env, "_read_text", _fake_reads({}))
    monkeypatch.setattr(env, "_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(env.os, "cpu_count", lambda: None)
    monkeypatch.setattr(env.platform, "processor", lambda: "")

    cpu = env.collect_cpu()
    assert cpu.logical_cores == 0
    assert cpu.quota_cores is None


def test_environment_serializes_cpu_when_probed():
    envinfo = EnvironmentInfo(
        gpu=[],
        driver_version="550.54",
        cuda_version="12.4",
        docker_version="27.0.3",
        harness_version="0.1.0",
        hostname_hash="sha256:" + "a" * 64,
        cpu=CpuInfo(
            model="Intel(R) Xeon(R) Platinum 8480+",
            logical_cores=224,
            available_cores=8,
            memory_total_mb=32128,
            quota_cores=6.0,
        ),
    )
    out = envinfo.to_dict()
    assert out["cpu"]["available_cores"] == 8
    assert out["cpu"]["quota_cores"] == 6.0


def test_environment_omits_cpu_when_it_was_never_probed():
    """Reports written before CPU fingerprinting must stay readable."""
    envinfo = EnvironmentInfo(
        gpu=[],
        driver_version="",
        cuda_version="",
        docker_version="",
        harness_version="0.1.0",
        hostname_hash="sha256:" + "a" * 64,
    )
    assert "cpu" not in envinfo.to_dict()


def test_report_validation_accepts_an_environment_with_cpu():
    """The new key must not trip the structural check on bench_report.json."""
    from bench.validate import validate_report_dict

    report = {
        "schema_version": 1,
        "task_id": "t1",
        "verdict": "pass",
        "started_at": "2026-09-03T00:00:00+00:00",
        "finished_at": "2026-09-03T01:00:00+00:00",
        "environment": EnvironmentInfo(
            gpu=[],
            driver_version="550.54",
            cuda_version="12.4",
            docker_version="27.0.3",
            harness_version="0.1.0",
            hostname_hash="sha256:" + "a" * 64,
            cpu=CpuInfo(
                model="Intel(R) Xeon(R) Platinum 8480+",
                logical_cores=224,
                available_cores=8,
                memory_total_mb=32128,
            ),
        ).to_dict(),
        "inputs_fingerprint": {
            "baseline_image_digest": "sha256:" + "0" * 64,
            "candidate_image_digest": ["sha256:" + "1" * 64],
            "model_repo": "org/model",
            "model_revision": "abc",
            "model_weights_sha256": "sha256:" + "2" * 64,
            "trace_sha256": "sha256:" + "3" * 64,
            "request_sha256": "sha256:" + "4" * 64,
        },
    }
    validate_report_dict(report)


def test_raw_dumps_keep_lscpu_next_to_docker_info(monkeypatch):
    """Both were asked for by name when miners could not match our numbers."""
    seen: list[str] = []

    def _fake_run(cmd, **_kw):
        seen.append(cmd[0])
        return f"{cmd[0]} output\n"

    monkeypatch.setattr(env, "_run", _fake_run)
    dumps = env.collect_env_raw_dumps()
    assert "lscpu" in seen
    assert dumps["lscpu.txt"] == "lscpu output\n"
    assert "docker-info.txt" in dumps
