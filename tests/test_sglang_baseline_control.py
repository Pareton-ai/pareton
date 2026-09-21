"""The identical-image diagnostic uses production request pins, without a GPU."""

import json
import runpy
from pathlib import Path

import pytest

import bench.main as harness
from bench.correctness import BASELINE_INDEX
from bench.schemas import CorrectnessReport, TraceRequest, TraceSampling

ROOT = Path(__file__).resolve().parents[1]
CONTROL = runpy.run_path(str(ROOT / "ops/sglang-baseline-control.py"))


def test_control_preserves_campaign_settings_and_uses_identical_images(tmp_path):
    fields = json.loads(CONTROL["FIELDS"].read_text())
    trace = tmp_path / "trace.json"
    trace.write_bytes((ROOT / "fixtures/bench/sample_trace.json").read_bytes())
    request = CONTROL["prepare_request"](fields, trace)
    assert request["engines"]["candidates"] == [request["engines"]["baseline"]]
    assert (
        request["engines"]["baseline"]["image"]
        == fields["bench"]["baseline_engine_image_digest"]
    )
    assert request["model"] == fields["bench"]["model"]
    assert request["hardware"] == {"gpu_count": 4, "gpu_sku_expected": "RTX5090"}
    assert request["sla_bench"]["repetitions"] == 3
    assert request["correctness"] == fields["bench"]["correctness"]
    assert request["scoring_rule"] == fields["scoring_rule"]


def report(mean=-1.0, **overrides):
    return CorrectnessReport(
        **{
            "verdict": "pass",
            "num_prompts": 32,
            "num_positions_scored": 1000,
            "mean_logprob": mean,
            "min_logprob": -20.0,
            "quantile_logprob": -11.0,
            "coverage_ratio": 1.0,
            "evidence": "evidence.jsonl",
            **overrides,
        }
    )


def thresholds():
    fields = json.loads(CONTROL["FIELDS"].read_text())
    return fields["bench"]["correctness"]["thresholds"]


def test_endpoint_stress_changes_only_role_temperatures_and_restores_harness(
    tmp_path, monkeypatch
):
    source = [
        TraceRequest(
            "hf-000",
            2,
            5120,
            TraceSampling(0.7, 1.0),
            prompt="write",
            input_length_group="2k",
        )
    ]
    calls = []
    globals_ = CONTROL["endpoint_stress"].__wrapped__.__globals__
    original_replay, original_grade = harness.run_sla_engine, harness.grade_all

    def replay(url, **kwargs):
        calls.append((url, kwargs))
        return "replayed"

    results = {BASELINE_INDEX: report(), 0: report(-3.0)}
    monkeypatch.setitem(globals_, "run_sla_engine", replay)
    monkeypatch.setitem(globals_, "grade_all", lambda *args, **kwargs: results.copy())
    with CONTROL["endpoint_stress"](tmp_path, [0.1, 1.01], thresholds()):
        for role in ("baseline", "baseline-drift", "candidate-0"):
            assert (
                harness.run_sla_engine(
                    "http://local", role=role, requests=source, cfg="three-reps"
                )
                == "replayed"
            )
        actual = harness.grade_all("http://scorer", [])
        actual.pop(BASELINE_INDEX)  # Production consumes the baseline entry.
    assert harness.run_sla_engine is original_replay
    assert harness.grade_all is original_grade
    assert [call[1]["requests"][0].sampling.temperature for call in calls] == [
        0.1,
        0.1,
        1.01,
    ]
    for _, call in calls:
        request = call["requests"][0]
        assert (
            request.id,
            request.prompt,
            request.max_tokens,
            request.arrival_offset_ms,
            request.input_length_group,
        ) == ("hf-000", "write", 5120, 2, "2k")
        assert request.sampling.top_p == 1.0
        assert request.sampling.ignore_eos is False
        assert call["cfg"] == "three-reps"
    assert source[0].sampling.temperature == 0.7
    saved = json.loads((tmp_path / "likelihood_summary.json").read_text())
    assert saved["mean_logprob_drop"] == 2.0
    assert saved["relative_drop_checks"] == {"1.5": False, "2.5": True}
    assert all(saved["engines"]["candidate"]["absolute_checks"].values())
    assert set(json.loads((tmp_path / "endpoint_correctness.json").read_text())) == {
        "baseline",
        "candidate-0",
    }
    assert (
        json.loads((tmp_path / "temperature_overrides.json").read_text())[
            "performance_comparison_valid"
        ]
        is False
    )


def test_endpoint_failure_writes_incomplete_likelihood_without_inventing_passes(
    tmp_path,
):
    original = harness.run_sla_engine
    with (
        pytest.raises(RuntimeError, match="baseline length"),
        CONTROL["endpoint_stress"](tmp_path, [0.1, 1.01], thresholds()),
    ):
        raise RuntimeError("baseline length failure")
    assert harness.run_sla_engine is original
    saved = json.loads((tmp_path / "likelihood_summary.json").read_text())
    assert saved["engines"] == {
        "baseline": {"scored": False},
        "candidate": {"scored": False},
    }
    assert saved["mean_logprob_drop"] is None
    assert saved["relative_drop_checks"] == {"1.5": None, "2.5": None}


def test_endpoint_summary_reports_absolute_failures_and_relative_boundary():
    summary = CONTROL["likelihood_summary"](
        {
            BASELINE_INDEX: report(-2.0),
            0: report(
                -4.5,
                quantile_logprob=-13,
                coverage_ratio=0.4,
                verdict="fail_correctness",
            ),
        },
        thresholds(),
    )
    assert summary["relative_drop_checks"] == {"1.5": False, "2.5": True}
    assert summary["engines"]["candidate"]["absolute_checks"] == {
        "mean_logprob_pass": False,
        "token_quantile_pass": False,
        "coverage_pass": False,
    }


def test_endpoint_override_flows_through_mock_harness_and_saved_replays(tmp_path):
    """Local mock HTTP engines exercise the adapter, not SGLang/GPU inference."""
    request = json.loads((ROOT / "fixtures/bench/sample_request.json").read_text())
    request["workload_trace"]["path"] = str(ROOT / "fixtures/bench/sample_trace.json")
    request["correctness"]["thresholds"] = thresholds()
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    with CONTROL["endpoint_stress"](tmp_path, [0.1, 1.01], thresholds()):
        code = harness.main(
            [
                "--request",
                str(request_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--mock-engine",
            ]
        )
    assert code == 0
    for role, temperature in (
        ("baseline", 0.1),
        ("baseline-drift", 0.1),
        ("candidate-0", 1.01),
    ):
        paths = sorted(
            (tmp_path / "output/evidence/sla_bench" / role).glob("*/requests.jsonl")
        )
        assert len(paths) == 4  # Mock vLLM has one warmup and three measured reps.
        for path in paths:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows = [row for row in rows if not row.get("_rep_meta")]
            assert len(rows) == 2
            assert all(
                row["sampling"]["temperature"] == temperature
                and row["sampling"]["seed"] == 0
                for row in rows
            )
    saved = json.loads((tmp_path / "likelihood_summary.json").read_text())
    assert saved["baseline_reference_passed"] is True
    assert all(engine["scored"] for engine in saved["engines"].values())
