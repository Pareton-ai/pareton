"""CLI tests: Module A via --mock-engine; stub path for non-correctness modes."""

from __future__ import annotations

import json
from pathlib import Path

from bench.correctness import PromptCase
from bench.main import EXIT_BAD_REQUEST, EXIT_ENV, EXIT_OK, main
from bench.validate import validate_report_dict

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "fixtures" / "bench" / "sample_request.json"
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"


def _write_request(tmp_path: Path, *, mode: str, **overrides) -> Path:
    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["mode"] = mode
    # Resolve trace path absolutely so CWD does not matter.
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req.update(overrides)
    path = tmp_path / f"request_{mode}.json"
    path.write_text(json.dumps(req, indent=2) + "\n", encoding="utf-8")
    return path


def test_cli_mock_engine_mode_perf_screen_pass(tmp_path: Path):
    """mode=perf_screen runs Module B against the mock engines."""
    req = _write_request(tmp_path, mode="perf_screen")
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-baseline-token-latency-s",
            "0.01",
            "--mock-candidate-token-latency-s",
            "0.005",
        ]
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["verdict"] == "pass"
    assert report["perf_screen"]["verdict"] == "pass"
    assert report["perf_screen"]["throughput_ratio"] >= 1.0
    assert "correctness" not in report
    assert (out / "evidence" / "perf_screen" / "perf_screen.jsonl").is_file()


def test_cli_perf_only_fail_has_no_skipped_note(tmp_path: Path):
    """mode=perf_screen fail must not claim sla_bench was skipped."""
    req = _write_request(tmp_path, mode="perf_screen")
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-baseline-token-latency-s",
            "0.005",
            "--mock-candidate-token-latency-s",
            "0.02",
        ]
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["verdict"] == "fail_perf_screen"
    assert report["perf_screen"]["verdict"] == "fail_perf_screen"
    assert "sla_bench" not in report
    assert "skipped_note" not in report


def test_cli_invalid_request_exit_1(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1}\n', encoding="utf-8")
    out = tmp_path / "out"
    code = main(["--request", str(bad), "--output-dir", str(out)])
    assert code == EXIT_BAD_REQUEST
    assert not (out / "bench_report.json").exists()


def test_cli_mock_engine_mode_correctness_pass(tmp_path: Path):
    req = _write_request(tmp_path, mode="correctness")
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
        ]
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["verdict"] == "pass"
    assert report["correctness"]["verdict"] == "pass"
    assert report["correctness"]["num_positions_compared"] > 0
    assert "stub_note" not in report
    evidence = out / "evidence" / "correctness" / "logprob_diffs.jsonl"
    assert evidence.is_file()


def test_cli_mock_engine_mode_all_full_pass(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("PARETON_BENCH_SKIP_SLA", raising=False)
    req = _write_request(tmp_path, mode="all")
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-baseline-token-latency-s",
            "0.03",
            "--mock-candidate-token-latency-s",
            "0.015",
        ]
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    # A and B are deterministic; both must pass and produce reports.
    assert report["correctness"]["verdict"] == "pass"
    assert report["perf_screen"]["verdict"] == "pass"
    # C ran and produced a complete report; its verdict depends on wall-clock
    # TTFT reproducibility, which is genuinely noisy on a shared host, so we
    # assert structure rather than an unconditional pass.
    sla = report["sla_bench"]
    assert sla["verdict"] in ("pass", "error")
    assert sla["candidate"]["ttft_ms"]["p99"] > 0
    assert sla["speedup"]["p99_ttft_ratio"] > 0
    assert report["verdict"] in ("pass", "error")
    assert "stub_note" not in report
    assert "skipped_note" not in report


def test_cli_mock_engine_skip_sla_omits_sla_bench(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PARETON_BENCH_SKIP_SLA", "1")
    req = _write_request(tmp_path, mode="all")
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-baseline-token-latency-s",
            "0.03",
            "--mock-candidate-token-latency-s",
            "0.015",
        ]
    )
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["correctness"]["verdict"] == "pass"
    assert report["perf_screen"]["verdict"] == "pass"
    assert "sla_bench" not in report
    assert report["verdict"] == "pass"


def test_cli_mock_tampered_candidate_fails(tmp_path: Path):
    req = _write_request(tmp_path, mode="correctness")
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-tampered-candidate",
        ]
    )
    assert code == EXIT_OK  # harness completed; fail is in the report
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["verdict"] == "fail_correctness"
    assert report["correctness"]["verdict"] == "fail_correctness"


def test_cli_tampered_flag_requires_mock_engine(tmp_path: Path):
    req = _write_request(tmp_path, mode="correctness")
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-tampered-candidate",
        ]
    )
    assert code == EXIT_BAD_REQUEST


def test_cli_bad_trace_sha_exits_before_engines(tmp_path: Path, monkeypatch):
    """Trace validation must fail before mock/Docker engines start."""
    called = {"n": 0}

    def should_not_run(self, *_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("engines must not start after bad trace")

    monkeypatch.setattr("bench.main._EngineProvider.run_correctness", should_not_run)
    req = _write_request(tmp_path, mode="correctness")
    raw = json.loads(req.read_text(encoding="utf-8"))
    raw["workload_trace"]["sha256"] = "sha256:" + ("0" * 64)
    req.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
        ]
    )
    assert code == EXIT_BAD_REQUEST
    assert called["n"] == 0
    assert not (out / "bench_report.json").exists()


def test_cli_host_environment_error_exit_2(tmp_path: Path, monkeypatch):
    """Docker unavailable must be exit 2, not engine exit 3."""
    from bench.lifecycle import HostEnvironmentError
    from bench.weights import StagedWeights

    def boom(self, *_args, **_kwargs):
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
    monkeypatch.setattr("bench.main._EngineProvider.run_correctness", boom)
    req = _write_request(tmp_path, mode="correctness")
    out = tmp_path / "out"
    code = main(["--request", str(req), "--output-dir", str(out)])
    assert code == EXIT_ENV
    assert not (out / "bench_report.json").exists()


def test_docker_engines_run_sequentially(tmp_path: Path, monkeypatch):
    """Baseline container must exit before candidate starts (avoid dual GPU load)."""
    from dataclasses import dataclass

    from bench.main import _EngineProvider
    from bench.validate import load_bench_request

    live = {"n": 0, "max": 0}

    @dataclass
    class _Handle:
        base_url: str = "http://127.0.0.1:9"
        image_digest: str = "sha256:" + ("a" * 64)

    class _FakeNet:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    class _FakeEngine:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            live["n"] += 1
            live["max"] = max(live["max"], live["n"])
            return _Handle()

        def __exit__(self, *exc):
            live["n"] -= 1
            return None

    monkeypatch.setattr("bench.main.BenchNetwork", _FakeNet)
    monkeypatch.setattr("bench.main.EngineContainer", _FakeEngine)

    class _Baseline:
        prompts = [PromptCase(id="p1", prompt="hi")]
        scored = []

    monkeypatch.setattr(
        "bench.main.collect_baseline_correctness",
        lambda *a, **k: _Baseline(),
    )
    monkeypatch.setattr(
        "bench.main.finish_correctness_with_candidate",
        lambda *a, **k: type(
            "R",
            (),
            {
                "verdict": "pass",
                "num_prompts": 1,
                "num_positions_compared": 1,
                "mean_abs_logprob_diff": 0.0,
                "max_abs_logprob_diff": 0.0,
                "argmax_mismatch_rate": 0.0,
                "evidence": "evidence/correctness/logprob_diffs.jsonl",
            },
        )(),
    )

    req_path = _write_request(tmp_path, mode="correctness")
    req, _ = load_bench_request(req_path)
    provider = _EngineProvider(req=req, mock=False, logs_dir=tmp_path / "logs")
    provider.run_correctness(
        [PromptCase(id="p1", prompt="hi")],
        req.correctness,
        req.task_id,
        tmp_path / "evidence" / "correctness",
    )
    assert live["max"] == 1
    assert live["n"] == 0


def _fake_docker_harness(monkeypatch, *, corr_verdict: str = "pass"):
    """Shared BenchNetwork/EngineContainer fakes for fused vs 4-start tests."""
    from dataclasses import dataclass

    live = {"n": 0, "max": 0, "starts": 0}
    order: list[str] = []

    @dataclass
    class _Handle:
        base_url: str = "http://127.0.0.1:9"
        image_digest: str = "sha256:" + ("a" * 64)

    class _FakeNet:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    class _FakeEngine:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            live["n"] += 1
            live["starts"] += 1
            live["max"] = max(live["max"], live["n"])
            return _Handle()

        def __exit__(self, *exc):
            live["n"] -= 1
            return None

    monkeypatch.setattr("bench.main.BenchNetwork", _FakeNet)
    monkeypatch.setattr("bench.main.EngineContainer", _FakeEngine)

    class _Baseline:
        prompts = [PromptCase(id="p1", prompt="hi")]
        scored = []

    def collect(*a, **k):
        order.append("baseline-correctness")
        return _Baseline()

    def finish_corr(*a, **k):
        order.append("candidate-correctness")
        return type(
            "R",
            (),
            {
                "verdict": corr_verdict,
                "num_prompts": 1,
                "num_positions_compared": 1,
                "mean_abs_logprob_diff": 0.0,
                "max_abs_logprob_diff": 0.0,
                "argmax_mismatch_rate": 0.0,
                "evidence": "evidence/correctness/logprob_diffs.jsonl",
            },
        )()

    def perf_engine(url, *, role, requests, cfg, evidence_dir, **kwargs):
        order.append(f"{role}-perf")
        evidence_dir.mkdir(parents=True, exist_ok=True)
        rows_path = evidence_dir / f"{role}_rows.jsonl"
        rows_path.write_text("{}\n", encoding="utf-8")
        return {
            "role": role,
            "wall_s": 1.0,
            "completion_tokens": 1,
            "output_tokens_per_s": 1.0,
            "num_requests": 1,
            "rows_file": rows_path.name,
        }

    def finish_perf(*a, **k):
        order.append("candidate-perf")
        return type(
            "P",
            (),
            {
                "verdict": "pass",
                "baseline_output_tokens_per_s": 1.0,
                "candidate_output_tokens_per_s": 1.0,
                "throughput_ratio": 1.0,
                "evidence": "evidence/perf_screen/perf_screen.jsonl",
            },
        )()

    monkeypatch.setattr("bench.main.collect_baseline_correctness", collect)
    monkeypatch.setattr("bench.main.finish_correctness_with_candidate", finish_corr)
    monkeypatch.setattr("bench.main.run_perf_screen_engine", perf_engine)
    monkeypatch.setattr("bench.main.finish_perf_screen", finish_perf)
    monkeypatch.setenv("PARETON_BENCH_SKIP_SLA", "1")
    return live, order


def _sglang_engines() -> dict:
    a = "ghcr.io/example/engine@sha256:" + ("a" * 64)
    b = "ghcr.io/example/engine@sha256:" + ("b" * 64)
    return {
        "baseline": {"image": a, "serve_args": ["--tp-size", "8"], "env": {}},
        "candidate": {"image": b, "serve_args": ["--tp-size", "8"], "env": {}},
    }


def test_mode_all_fuses_containers_per_role(tmp_path: Path, monkeypatch):
    from bench.main import _EngineProvider, run_all_modules
    from bench.output import OutputLayout
    from bench.validate import load_bench_request, load_workload_trace

    live, order = _fake_docker_harness(monkeypatch)
    req_path = _write_request(tmp_path, mode="all", engines=_sglang_engines())
    req, _ = load_bench_request(req_path)
    trace = load_workload_trace(
        Path(req.workload_trace.path), expected_sha256=req.workload_trace.sha256
    )
    layout = OutputLayout(tmp_path / "out")
    layout.prepare()
    provider = _EngineProvider(req=req, mock=False, logs_dir=tmp_path / "logs")
    corr, perf, sla, note = run_all_modules(
        req=req,
        provider=provider,
        prompts=[PromptCase(id="p1", prompt="hi")],
        trace=trace,
        layout=layout,
    )
    assert live["starts"] == 2
    assert live["max"] == 1
    assert live["n"] == 0
    assert order == [
        "baseline-correctness",
        "baseline-perf",
        "candidate-correctness",
        "candidate-perf",
    ]
    assert corr is not None and corr.verdict == "pass"
    assert perf is not None and perf.verdict == "pass"
    assert sla is None
    assert note is None
    assert provider.baseline_digest == "sha256:" + ("a" * 64)
    assert provider.candidate_digest == "sha256:" + ("a" * 64)


def test_fused_correctness_fail_skips_candidate_perf(tmp_path: Path, monkeypatch):
    from bench.main import _EngineProvider, _NOTE_CORRECTNESS_FAILED, run_all_modules
    from bench.output import OutputLayout
    from bench.validate import load_bench_request, load_workload_trace

    live, order = _fake_docker_harness(monkeypatch, corr_verdict="fail_correctness")
    req_path = _write_request(tmp_path, mode="all", engines=_sglang_engines())
    req, _ = load_bench_request(req_path)
    trace = load_workload_trace(
        Path(req.workload_trace.path), expected_sha256=req.workload_trace.sha256
    )
    layout = OutputLayout(tmp_path / "out")
    layout.prepare()
    provider = _EngineProvider(req=req, mock=False, logs_dir=tmp_path / "logs")
    corr, perf, sla, note = run_all_modules(
        req=req,
        provider=provider,
        prompts=[PromptCase(id="p1", prompt="hi")],
        trace=trace,
        layout=layout,
    )
    assert live["starts"] == 2
    assert live["max"] == 1
    assert "candidate-perf" not in order
    assert corr is not None and corr.verdict == "fail_correctness"
    assert perf is None
    assert sla is None
    assert note == _NOTE_CORRECTNESS_FAILED
    assert not (layout.perf_screen_dir / "baseline_rows.jsonl").exists()


def test_vllm_serve_args_do_not_fuse(tmp_path: Path, monkeypatch):
    from bench.main import _EngineProvider, run_all_modules
    from bench.output import OutputLayout
    from bench.validate import load_bench_request, load_workload_trace

    live, order = _fake_docker_harness(monkeypatch)
    req_path = _write_request(tmp_path, mode="all")
    req, _ = load_bench_request(req_path)
    trace = load_workload_trace(
        Path(req.workload_trace.path), expected_sha256=req.workload_trace.sha256
    )
    layout = OutputLayout(tmp_path / "out")
    layout.prepare()
    provider = _EngineProvider(req=req, mock=False, logs_dir=tmp_path / "logs")
    corr, perf, sla, note = run_all_modules(
        req=req,
        provider=provider,
        prompts=[PromptCase(id="p1", prompt="hi")],
        trace=trace,
        layout=layout,
    )
    assert live["starts"] == 4
    assert live["max"] == 1
    assert live["n"] == 0
    assert order == [
        "baseline-correctness",
        "candidate-correctness",
        "baseline-perf",
        "candidate-perf",
    ]
    assert corr is not None and corr.verdict == "pass"
    assert perf is not None and perf.verdict == "pass"
    assert sla is None
    assert note is None


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
