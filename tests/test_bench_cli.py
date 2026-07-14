"""CLI stub tests: valid request → exit 0 + report; invalid → exit 1."""

from __future__ import annotations

import json
from pathlib import Path

from bench.main import EXIT_BAD_REQUEST, EXIT_OK, main
from bench.validate import validate_report_dict

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "fixtures" / "bench" / "sample_request.json"


def test_cli_stub_valid_request(tmp_path: Path):
    out = tmp_path / "out"
    code = main(["--request", str(SAMPLE_REQUEST), "--output-dir", str(out)])
    assert code == EXIT_OK
    report_path = out / "bench_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["verdict"] == "error"  # stub: modules not run
    assert "stub_note" in report
    assert (out / "harness.log").is_file()
    assert (out / "evidence" / "env").is_dir()
    assert (out / "evidence" / "correctness").is_dir()
    assert (out / "evidence" / "perf_screen").is_dir()
    assert (out / "evidence" / "sla_bench").is_dir()
    # Fingerprints populated from request
    assert report["inputs_fingerprint"]["model_repo"] == "Qwen/Qwen2.5-7B-Instruct"
    baseline_digest = report["inputs_fingerprint"]["baseline_image_digest"]
    assert baseline_digest.startswith("sha256:")
    assert len(baseline_digest) == len("sha256:") + 64
    # Must be the digest alone, not the full image ref
    assert "/" not in baseline_digest
    assert report["environment"]["harness_version"]


def test_cli_invalid_request_exit_1(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1}\n', encoding="utf-8")
    out = tmp_path / "out"
    code = main(["--request", str(bad), "--output-dir", str(out)])
    assert code == EXIT_BAD_REQUEST
    assert not (out / "bench_report.json").exists()


def test_cli_accepts_mock_engine_flag(tmp_path: Path):
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(SAMPLE_REQUEST),
            "--output-dir",
            str(out),
            "--mock-engine",
        ]
    )
    assert code == EXIT_OK
