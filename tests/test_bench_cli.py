"""CLI tests: Module A via --mock-engine; stub path for non-correctness modes."""

from __future__ import annotations

import json
from pathlib import Path

from bench.correctness import PromptCase
from bench.main import EXIT_BAD_REQUEST, EXIT_OK, main
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


def test_cli_stub_non_correctness_mode(tmp_path: Path):
    """mode=perf_screen still stubs (Modules B/C not built)."""
    req = _write_request(tmp_path, mode="perf_screen")
    out = tmp_path / "out"
    code = main(["--request", str(req), "--output-dir", str(out)])
    assert code == EXIT_OK
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["verdict"] == "error"
    assert "stub_note" in report
    assert (out / "harness.log").is_file()


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


def test_cli_mock_engine_mode_all_pass_then_stub(tmp_path: Path):
    req = _write_request(tmp_path, mode="all")
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
    # Correctness passed but B/C not implemented → overall error + stub_note.
    assert report["verdict"] == "error"
    assert report["correctness"]["verdict"] == "pass"
    assert "stub_note" in report
    assert "Modules B/C" in report["stub_note"]


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

    def should_not_run(**_kwargs):
        called["n"] += 1
        raise AssertionError("engines must not start after bad trace")

    monkeypatch.setattr("bench.main.run_with_mock_engines", should_not_run)
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


def test_docker_engines_run_sequentially(tmp_path: Path, monkeypatch):
    """Baseline container must exit before candidate starts (avoid dual GPU load)."""
    from dataclasses import dataclass

    from bench.main import run_with_docker_engines
    from bench.output import OutputLayout
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
    layout = OutputLayout(tmp_path / "out")
    layout.prepare()
    run_with_docker_engines(
        req=req,
        layout=layout,
        prompts=[PromptCase(id="p1", prompt="hi")],
    )
    assert live["max"] == 1
    assert live["n"] == 0
