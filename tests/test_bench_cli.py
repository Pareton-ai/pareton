"""CLI tests: a whole round via --mock-engine, plus request/exit-code paths."""

from __future__ import annotations

import json
from pathlib import Path

from bench.main import EXIT_BAD_REQUEST, EXIT_ENV, EXIT_OK, main

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "fixtures" / "bench" / "sample_request.json"
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"


def _candidate(letter: str) -> dict:
    return {
        "image": f"ghcr.io/example/engine@sha256:{letter * 64}",
        "serve_args": ["--enable-prefix-caching"],
        "env": {},
        "cache_dir": "/root/.cache/vllm",
    }


def _write_request(
    tmp_path: Path, *, mode: str = "all", candidates: int = 1, **overrides
) -> Path:
    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["mode"] = mode
    # Resolve trace path absolutely so CWD does not matter.
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req["engines"]["candidates"] = [
        _candidate("0123456789abcdef"[i]) for i in range(candidates)
    ]
    req.update(overrides)
    path = tmp_path / f"request_{mode}_{candidates}.json"
    path.write_text(json.dumps(req, indent=2) + "\n", encoding="utf-8")
    return path


def test_cli_invalid_request_exit_1(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1}\n', encoding="utf-8")
    out = tmp_path / "out"
    code = main(["--request", str(bad), "--output-dir", str(out)])
    assert code == EXIT_BAD_REQUEST
    assert not (out / "bench_report.json").exists()


def test_cli_mock_round_scores_every_entry(tmp_path: Path):
    req = _write_request(tmp_path, candidates=1)
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
        ]
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    from bench.validate import validate_report_dict

    validate_report_dict(report)
    assert report["verdict"] == "pass"
    assert report["scoring_rule"] == {"name": "median_e2e_speedup"}
    assert len(report["entries"]) == 1
    entry = report["entries"][0]
    assert entry["status"] == "scored"
    assert entry["correctness"]["verdict"] == "pass"
    assert entry["score"] is not None
    # The scoring contract: request id -> timing, on both sides.
    assert set(report["baseline"]["timings"]) == {"r-000001", "r-000002"}
    assert set(entry["sla"]["timings"]) == {"r-000001", "r-000002"}
    for timing in report["baseline"]["timings"].values():
        assert set(timing) == {"ttft_s", "itl_s", "completion_tokens"}
    # Drift is measured, not assumed.
    assert report["baseline_drift"] is not None
    assert report["drift_baseline"]["role"] == "baseline-drift"


def test_cli_mock_round_scripts_the_whole_matrix(tmp_path: Path):
    """Baseline 1.0, two close challengers, a leader, a correctness fail, an
    infra fail, a startup crash: every outcome a round can produce, in one run."""
    req = _write_request(tmp_path, candidates=6)
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-baseline-token-latency-s",
            "0.006",
            "--mock-candidates",
            json.dumps(
                [
                    {"speed_factor": 1.02},
                    {"speed_factor": 1.05},
                    {"speed_factor": 3.0},
                    {"garbage": True},
                    {"infra_fail": True},
                    {"crash": True},
                ]
            ),
        ]
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    by_index = {e["index"]: e for e in report["entries"]}
    assert len(by_index) == 6

    assert by_index[0]["status"] == "scored"
    assert by_index[1]["status"] == "scored"
    assert by_index[2]["status"] == "scored"
    # A wrong image is disqualified: no score at all, so it cannot lead.
    assert by_index[3]["status"] == "disqualified"
    assert by_index[3]["score"] is None
    assert by_index[3]["correctness"]["verdict"] == "fail_correctness"
    # A candidate that will not start is that entry's problem, not the round's.
    assert by_index[4]["status"] == "infra_failed"
    assert by_index[4]["score"] is None
    # An engine whose own process dies at startup is disqualified, terminal:
    # no requeue, unlike an infra flake.
    assert by_index[5]["status"] == "disqualified"
    assert by_index[5]["score"] is None
    assert "correctness" not in by_index[5]
    assert by_index[5]["engine_crashed"] is True
    assert "died before becoming healthy" in by_index[5]["reason"]

    # The clear leader must out-score the two close challengers.
    assert by_index[2]["score"] > by_index[0]["score"]
    assert by_index[2]["score"] > by_index[1]["score"]


def test_cli_mock_candidates_requires_mock_engine(tmp_path: Path):
    req = _write_request(tmp_path)
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-candidates",
            "[]",
        ]
    )
    assert code == EXIT_BAD_REQUEST


def test_cli_mock_candidates_rejects_bad_json(tmp_path: Path):
    req = _write_request(tmp_path)
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-candidates",
            "{not json",
        ]
    )
    assert code == EXIT_BAD_REQUEST


def test_cli_bad_trace_sha_exits_before_engines(tmp_path: Path, monkeypatch):
    """Trace validation must fail before any engine starts: a bad trace must
    not cost a pod-hour."""
    called = {"n": 0}

    def should_not_run(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("engines must not start after bad trace")

    monkeypatch.setattr("bench.main._EngineProvider.start", should_not_run)
    req = _write_request(tmp_path)
    raw = json.loads(req.read_text(encoding="utf-8"))
    raw["workload_trace"]["sha256"] = "sha256:" + ("0" * 64)
    req.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    code = main(["--request", str(req), "--output-dir", str(out), "--mock-engine"])
    assert code == EXIT_BAD_REQUEST
    assert called["n"] == 0
    assert not (out / "bench_report.json").exists()


def test_cli_host_environment_error_exit_2(tmp_path: Path, monkeypatch):
    """Docker unavailable must be exit 2, not engine exit 3."""
    from bench.lifecycle import HostEnvironmentError
    from bench.weights import StagedWeights

    def boom(*_args, **_kwargs):
        raise HostEnvironmentError("docker CLI not found on PATH")

    def fake_stage(model, *, token_env="HF_TOKEN", cache_dir=None):
        root = tmp_path / "staged-weights"
        root.mkdir(exist_ok=True)
        return StagedWeights(
            path=root,
            weights_sha256="sha256:" + ("c" * 64),
            num_files=1,
            total_bytes=1,
            manifest={
                "repo": model.hf_repo,
                "revision": model.hf_revision,
                "files": [],
            },
        )

    monkeypatch.setattr("bench.main.stage_weights", fake_stage)
    monkeypatch.setattr("bench.main._EngineProvider.start", boom)
    req = _write_request(tmp_path)
    out = tmp_path / "out"
    code = main(["--request", str(req), "--output-dir", str(out)])
    assert code == EXIT_ENV
    assert not (out / "bench_report.json").exists()


def test_variance_gate_infra_fails_before_score():
    """p99 e2e relative range over the 0.335 bar lands infra_failed, not a score."""
    from bench.main import _CandidateRun, _build_entries
    from bench.sla_bench import REPRO_BAR_MAX_REL_RANGE

    class _Result:
        cross_rep_variance = {
            "p99_ttft_ms_rel_range": 0.01,
            "p99_itl_ms_rel_range": 0.01,
            "p99_e2e_ms_rel_range": REPRO_BAR_MAX_REL_RANGE + 0.01,
        }
        timings = {}

    run = _CandidateRun(
        index=0, status="ok", replay=type("Replay", (), {"result": _Result()})()
    )
    entries = _build_entries(
        req=None,  # unused: the gate fires before score_candidate
        baseline=None,
        runs=[run],
        correctness={},
        digests=["sha256:" + "b" * 64],
    )
    assert len(entries) == 1
    assert entries[0].status == "infra_failed"
    assert entries[0].score is None
    assert "p99_e2e_ms_rel_range" in (entries[0].reason or "")
    assert entries[0].sla is run.replay.result
    assert (
        entries[0].sla.cross_rep_variance["p99_e2e_ms_rel_range"]
        > REPRO_BAR_MAX_REL_RANGE
    )


def test_sku_mismatch_accepts_rtx5090_with_spaces():
    from bench.env import warn_gpu_sku_mismatch
    from bench.schemas import EnvironmentInfo, GpuInfo

    env = EnvironmentInfo(
        gpu=[
            GpuInfo(index=0, name="NVIDIA GeForce RTX 5090", vbios="", memory_mb=32768)
        ],
        driver_version="",
        cuda_version="",
        docker_version="",
        harness_version="0",
        hostname_hash="sha256:" + ("a" * 64),
    )
    assert warn_gpu_sku_mismatch(env, "RTX5090") is None
    assert warn_gpu_sku_mismatch(env, "H200") is not None
