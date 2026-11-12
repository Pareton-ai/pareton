"""The scoring rule's GPU memory cap: sampling, comparison and the round path."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import bench.main as bm
from bench import gpu_memory
from bench.concurrency import WEIGHTED_RULE
from bench.gpu_memory import (
    FixedGpuMemory,
    GpuMemorySampler,
    gpu_memory_report,
    merge_peaks,
    read_gpu_memory_mib,
)
from bench.main import EXIT_ENGINE, EXIT_OK, main
from campaign.models import validate_scoring_rule
from tests.test_bench_cli import _write_request
from worker.round_job import entry_results_from_report

pytestmark = pytest.mark.unit

GIB2 = 2048


# -- Sampling ----------------------------------------------------------------


@pytest.mark.parametrize(
    "stdout,returncode,expected",
    [
        ("120500\n120611\n", 0, [120500.0, 120611.0]),
        ("  7 \n\n", 0, [7.0]),
        ("", 0, None),
        ("[N/A]\n", 0, None),
        ("120500\nnan\n", 0, None),
        ("inf\n", 0, None),
        ("-1\n", 0, None),
        ("120500\n", 9, None),
    ],
)
def test_read_parses_one_value_per_gpu(monkeypatch, stdout, returncode, expected):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(gpu_memory.subprocess, "run", run)
    assert read_gpu_memory_mib() == expected
    assert calls[0][0] == "nvidia-smi"
    assert "--query-gpu=memory.used" in calls[0]


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 10)],
)
def test_read_is_none_without_a_working_nvidia_smi(monkeypatch, error):
    def run(cmd, **kwargs):
        raise error

    monkeypatch.setattr(gpu_memory.subprocess, "run", run)
    assert read_gpu_memory_mib() is None


def _sampler(samples):
    """A sampler fed these readings, one per call, without a live thread."""
    feed = iter(samples)
    # A long interval: only the entry and exit samples and the explicit
    # _sample() calls below run, so the test is deterministic.
    return GpuMemorySampler(read=lambda: next(feed, None), interval_s=3600)


def test_sampler_keeps_each_gpus_peak():
    sampler = _sampler([[10, 50], [30, 40], None, [20, 60], [5, 5]])
    with sampler:
        for _ in range(3):
            sampler._sample()
    # Entry, three middle samples (one unusable) and the closing sample.
    assert sampler.peak_mib == [30, 60]
    assert sampler.samples == 4


def test_sampler_drops_a_sample_with_a_different_gpu_count():
    sampler = _sampler([[10, 50], [999], [20, 40]])
    with sampler:
        sampler._sample()
    assert sampler.peak_mib == [20, 50]


def _timed_sampler(samples):
    """Readings paired with the time each one is taken at."""
    feed = iter(samples)
    now = [0.0]

    def read():
        now[0], value = next(feed)
        return value

    return GpuMemorySampler(read=read, interval_s=3600, clock=lambda: now[0])


def test_sampler_tolerates_a_short_gap_between_usable_samples():
    limit = gpu_memory.MAX_BLIND_S
    sampler = _timed_sampler(
        [(0, [10]), (2, None), (4, [999, 1]), (limit, [30]), (limit + 2, [20])]
    )
    with sampler:
        for _ in range(3):
            sampler._sample()
    assert sampler.peak_mib == [30]


@pytest.mark.parametrize(
    "samples",
    [
        # Blind in the middle: the next usable sample comes too late.
        [(0, [10]), (2, None), (40, [30]), (42, [20])],
        # Blind to the end: only the entry sample was ever usable.
        [(0, [10]), (2, None), (20, None), (40, None)],
        # A changed GPU count is as blind as no answer.
        [(0, [10]), (2, [1, 2]), (20, [1, 2]), (40, [1, 2])],
    ],
)
def test_sampler_has_no_peak_after_a_long_blind_stretch(samples):
    sampler = _timed_sampler(samples)
    with sampler:
        for _ in range(len(samples) - 2):
            sampler._sample()
    assert sampler.peak_mib is None


def test_sampler_has_no_peak_when_nvidia_smi_never_answers():
    with _sampler([]) as sampler:
        pass
    assert sampler.peak_mib is None


def test_sampler_polls_in_the_background():
    import threading

    seen = threading.Event()
    readings = iter([[1.0], [9.0]])

    def read():
        value = next(readings, [2.0])
        if value == [9.0]:
            seen.set()
        return value

    with GpuMemorySampler(read=read, interval_s=0.01) as sampler:
        assert seen.wait(5)
    assert sampler.peak_mib == [9.0]


# -- Comparison --------------------------------------------------------------


def test_merge_keeps_the_larger_peak_per_gpu():
    assert merge_peaks(None, [1, 5]) == [1, 5]
    assert merge_peaks([3, 2], [1, 5]) == [3, 5]
    # A second baseline that could not be sampled, or saw other GPUs, leaves
    # no reference rather than half of one.
    assert merge_peaks([3, 2], None) is None
    assert merge_peaks([3, 2], [1]) is None


def test_report_takes_the_largest_excess_on_any_gpu():
    report = gpu_memory_report([100, 100], [90, 100 + GIB2], GIB2)
    assert report == {
        "baseline_peak_mib": [100, 100],
        "candidate_peak_mib": [90, 100 + GIB2],
        "extra_mib": GIB2,
        "max_extra_gpu_memory_mib": GIB2,
        "over_cap": False,
    }
    assert gpu_memory_report([100, 100], [90, 101 + GIB2], GIB2)["over_cap"] is True
    assert gpu_memory_report([100, 100], [50, 60], GIB2)["extra_mib"] == -40


@pytest.mark.parametrize(
    "baseline,candidate", [(None, [1]), ([1], None), ([1, 2], [1]), ([], [])]
)
def test_report_is_none_when_the_sides_cannot_be_compared(baseline, candidate):
    assert gpu_memory_report(baseline, candidate, GIB2) is None


# -- Scoring rule ------------------------------------------------------------


@pytest.mark.parametrize("name", [WEIGHTED_RULE, "median_e2e_speedup"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 10**309, True, "2"])
def test_memory_cap_validation(name, value):
    with pytest.raises(ValueError, match="max_extra_gpu_memory_mib"):
        validate_scoring_rule({"name": name, "max_extra_gpu_memory_mib": value})


def test_memory_cap_is_pinned_only_when_set():
    assert "max_extra_gpu_memory_mib" not in validate_scoring_rule(
        {"name": WEIGHTED_RULE}
    )
    rule = validate_scoring_rule(
        {"name": WEIGHTED_RULE, "max_extra_gpu_memory_mib": GIB2}
    )
    assert rule["max_extra_gpu_memory_mib"] == 2048.0
    assert isinstance(rule["max_extra_gpu_memory_mib"], float)


# -- The round ---------------------------------------------------------------

CAPPED_RULE = {"name": "median_e2e_speedup", "max_extra_gpu_memory_mib": GIB2}


def _run(tmp_path: Path, candidates: list[dict], **overrides) -> tuple[int, Path]:
    req = _write_request(tmp_path, candidates=len(candidates), **overrides)
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-baseline-token-latency-s",
            "0.004",
            "--mock-candidates",
            json.dumps(candidates),
        ]
    )
    return code, out


def test_round_disqualifies_a_candidate_over_the_memory_cap(tmp_path: Path):
    code, out = _run(
        tmp_path,
        [
            {"speed_factor": 1.5, "gpu_memory_mib": [GIB2]},
            {"speed_factor": 3.0, "gpu_memory_mib": [GIB2 + 1]},
            {"speed_factor": 1.2},
        ],
        scoring_rule=CAPPED_RULE,
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    from bench.validate import validate_report_dict

    validate_report_dict(report)
    at_cap, over, same = report["entries"]
    # Exactly at the cap is allowed; the report shows the measurement.
    assert at_cap["status"] == "scored"
    assert at_cap["gpu_memory"] == {
        "baseline_peak_mib": [0.0],
        "candidate_peak_mib": [float(GIB2)],
        "extra_mib": float(GIB2),
        "max_extra_gpu_memory_mib": float(GIB2),
        "over_cap": False,
    }
    # One MiB over: disqualified whatever its speed, with the replay as
    # evidence, never graded, and not mistaken for a startup crash.
    assert over["status"] == "disqualified"
    assert over["score"] is None
    assert over["reason"].startswith("gpu_memory_over_cap")
    assert over["gpu_memory"]["over_cap"] is True
    assert over["gpu_memory"]["extra_mib"] == GIB2 + 1
    assert "engine_crashed" not in over
    assert "correctness" not in over
    assert over["sla"]["timings"]
    assert same["status"] == "scored"
    assert same["gpu_memory"]["extra_mib"] == 0
    statuses = json.loads((out / "entry_status.json").read_text(encoding="utf-8"))
    entries = statuses.get("entries", statuses)
    assert entries["1"]["status"] == "disqualified"


def test_round_without_a_cap_never_samples(tmp_path: Path, monkeypatch):
    def no_sampling(*args, **kwargs):
        raise AssertionError("nvidia-smi must not run without a memory cap")

    monkeypatch.setattr(bm._EngineProvider, "gpu_memory", no_sampling)
    code, out = _run(tmp_path, [{"gpu_memory_mib": [10**6]}])
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    assert report["entries"][0]["status"] == "scored"
    assert "gpu_memory" not in report["entries"][0]


def test_a_capped_docker_round_uses_the_real_sampler():
    provider = bm._EngineProvider.__new__(bm._EngineProvider)
    provider._gpu_memory_sampler = None
    provider._mock = False
    start = SimpleNamespace(candidate_index=0)
    assert isinstance(provider.gpu_memory(start), GpuMemorySampler)
    fake = FixedGpuMemory([1.0])
    provider._gpu_memory_sampler = lambda: fake
    assert provider.gpu_memory(start) is fake


def test_round_fails_when_the_baseline_cannot_be_sampled(tmp_path: Path, monkeypatch):
    """No reference, no verdicts: the round fails as the harness's own."""
    monkeypatch.setattr(
        bm._EngineProvider, "gpu_memory", lambda self, start: FixedGpuMemory()
    )
    code, out = _run(tmp_path, [{"speed_factor": 2.0}], scoring_rule=CAPPED_RULE)
    assert code == EXIT_ENGINE
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] != "pass"
    assert "GPU memory of baseline could not be sampled" in json.dumps(report)


def test_candidate_that_cannot_be_sampled_is_an_infra_failure(
    tmp_path: Path, monkeypatch
):
    def sampler(self, start):
        return FixedGpuMemory(None if start.candidate_index == 0 else [100.0])

    monkeypatch.setattr(bm._EngineProvider, "gpu_memory", sampler)
    code, out = _run(
        tmp_path,
        [{"speed_factor": 2.0}, {"speed_factor": 2.0}],
        scoring_rule=CAPPED_RULE,
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    unsampled, sampled = report["entries"]
    assert unsampled["status"] == "infra_failed"
    assert "could not be sampled" in unsampled["reason"]
    assert sampled["status"] == "scored"


def test_a_leader_over_the_cap_stays_disqualified():
    """Only a startup crash maps an incumbent to infra; a rule breach does not."""
    entries = [
        {"id": 1, "role": "baseline"},
        {"id": 2, "role": "leader", "submission_id": "s"},
    ]
    report = {
        "baseline": {"role": "baseline"},
        "entries": [
            {
                "index": 0,
                "status": "disqualified",
                "score": None,
                "reason": "gpu_memory_over_cap: ...",
                "gpu_memory": {"over_cap": True},
            }
        ],
    }
    leader = entry_results_from_report(entries, report)[1]
    assert leader["status"] == "disqualified"
    assert leader["report"]["gpu_memory"] == {"over_cap": True}
