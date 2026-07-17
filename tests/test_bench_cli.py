"""CLI tests: Module A via --mock-engine; stub path for non-correctness modes."""

from __future__ import annotations

import json
from pathlib import Path

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
