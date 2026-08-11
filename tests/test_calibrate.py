"""Offline B7 calibration prepare + analyze tests (no GPU/network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from bench.calibrate import (
    A3A_TRACE_SHA256,
    APPLY_DEFAULT_MIN_POSITIONS,
    CalibrationError,
    analyze_reports,
    analyze_runs_dir,
    compare_shape_to_fixture,
    correctness_dict_from_summary,
    correctness_would_pass,
    discover_run_dirs,
    prepare_calibration_request,
    suggest_threshold,
)
from bench.validate import validate_bench_request_dict
from builder.registry import baseline_engine_image_ref, normalize_digest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TRACE = ROOT / "fixtures" / "campaigns" / "synthetic_v0" / "workload_trace.json"
SHAPE_FIXTURE = ROOT / "fixtures" / "bench" / "vllm_completion_response_shape.json"
DIGEST = "sha256:" + ("a" * 64)


def _trace_url() -> str:
    return f"file://{FIXTURE_TRACE.resolve()}"


def _env() -> dict:
    return {
        "gpu": [{"index": 0, "name": "Fake", "vbios": "", "memory_mb": 1}],
        "driver_version": "0",
        "cuda_version": "0",
        "docker_version": "0",
        "harness_version": "0",
        "hostname_hash": "h",
    }


def _sla_engine(*, p99_ttft: float = 100.0) -> dict:
    lat = {"p50": 50.0, "p95": 80.0, "p99": p99_ttft}
    return {
        "ttft_ms": lat,
        "itl_ms": lat,
        "e2e_ms": {"p50": 100.0, "p95": 120.0, "p99": 150.0},
        "output_tokens_per_s": 10.0,
        "requests_per_s": 1.0,
        "sla_goodput_ratio": 1.0,
    }


def _report(
    *,
    mean: float = 0.0,
    max_d: float = 0.0,
    argmax: float = 0.0,
    p99_ttft: float = 100.0,
    inner_rel: float = 0.02,
    digest: str = DIGEST,
) -> dict:
    return {
        "schema_version": 1,
        "task_id": "550e8400-e29b-41d4-a716-446655440000",
        "verdict": "pass",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "environment": _env(),
        "inputs_fingerprint": {
            "baseline_image_digest": digest,
            "candidate_image_digest": digest,
            "model_repo": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "model_weights_sha256": "sha256:" + ("0" * 64),
            "trace_sha256": A3A_TRACE_SHA256,
            "request_sha256": "sha256:" + ("1" * 64),
        },
        "correctness": {
            "verdict": "pass",
            "num_prompts": 1,
            "num_positions_compared": 1,
            "mean_abs_logprob_diff": mean,
            "max_abs_logprob_diff": max_d,
            "argmax_mismatch_rate": argmax,
            "evidence": "e",
        },
        "perf_screen": {
            "verdict": "pass",
            "baseline_output_tokens_per_s": 1.0,
            "candidate_output_tokens_per_s": 1.0,
            "throughput_ratio": 1.0,
            "evidence": "e",
        },
        "sla_bench": {
            "verdict": "pass",
            "repetitions": 3,
            "candidate": _sla_engine(p99_ttft=p99_ttft),
            "baseline": _sla_engine(p99_ttft=p99_ttft),
            "speedup": {
                "output_tokens_per_s_ratio": 1.0,
                "requests_per_s_ratio": 1.0,
                "p99_ttft_ratio": 1.0,
                "p99_itl_ratio": 1.0,
                "p99_e2e_ratio": 1.0,
            },
            "cross_rep_variance": {
                "p99_ttft_ms_rel_range": inner_rel,
                "p99_itl_ms_rel_range": 0.01,
                "p99_e2e_ms_rel_range": 0.01,
            },
            "evidence": "e",
        },
    }


def test_prepare_pins_model_and_equal_engines(tmp_path: Path):
    out = tmp_path / "b7"
    req = prepare_calibration_request(
        engine_digest=DIGEST,
        gpu_sku="RTX-4090",
        output_dir=out,
        trace_url=_trace_url(),
        trace_sha256=A3A_TRACE_SHA256,
        task_id="550e8400-e29b-41d4-a716-446655440000",
    )
    validate_bench_request_dict(req)
    assert req["mode"] == "all"
    assert req["model"]["hf_repo"] == "Qwen/Qwen2.5-7B-Instruct"
    assert req["model"]["hf_revision"] == "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
    assert req["hardware"]["gpu_sku_expected"] == "RTX-4090"
    expect = baseline_engine_image_ref(normalize_digest(DIGEST))
    assert req["engines"]["baseline"]["image"] == expect
    assert req["engines"]["candidate"]["image"] == expect
    assert req["correctness"]["thresholds"]["mean_abs_logprob_diff"] == 1e9
    assert req["perf_screen"]["min_throughput_ratio"] == 0.0
    # A3a / synthetic_v0 has 2 requests; prepare must clamp config defaults (8).
    assert req["correctness"]["num_prompts"] == 2
    assert req["perf_screen"]["num_requests"] == 2
    assert (out / "bench_request.json").is_file()
    assert (out / "workload_trace.json").is_file()


def test_prepare_rejects_bad_digest(tmp_path: Path):
    with pytest.raises((CalibrationError, ValueError)):
        prepare_calibration_request(
            engine_digest="not-a-digest",
            gpu_sku="H200",
            output_dir=tmp_path,
            trace_url=_trace_url(),
            trace_sha256=A3A_TRACE_SHA256,
        )


def test_prepare_rejects_trace_mismatch(tmp_path: Path):
    with pytest.raises(CalibrationError, match="trace_sha256_mismatch"):
        prepare_calibration_request(
            engine_digest=DIGEST,
            gpu_sku="H200",
            output_dir=tmp_path,
            trace_url=_trace_url(),
            trace_sha256="sha256:" + ("f" * 64),
        )


def test_suggest_threshold_zero_floors_to_config():
    s = suggest_threshold(
        0.0,
        safety_factor=2.0,
        floor=float(config.BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF),
    )
    assert s["all_observed_zero"] is True
    assert s["suggested"] == float(config.BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF)


def test_suggest_threshold_scales_above_floor():
    s = suggest_threshold(0.01, safety_factor=2.0, floor=0.005)
    assert s["suggested"] == pytest.approx(0.02)
    assert s["all_observed_zero"] is False


def test_analyze_clean_zeros_and_tampered_holdout(tmp_path: Path):
    clean_paths = []
    for i, p99 in enumerate((100.0, 102.0, 101.0)):
        p = tmp_path / f"clean-{i}.json"
        p.write_text(
            json.dumps(_report(p99_ttft=p99, inner_rel=0.03)), encoding="utf-8"
        )
        clean_paths.append(p)

    summary = analyze_reports(clean_paths, safety_factor=2.0)
    thr = {
        "mean_abs_logprob_diff": summary["correctness"]["mean_abs_logprob_diff"][
            "suggested"
        ],
        "max_abs_logprob_diff": summary["correctness"]["max_abs_logprob_diff"][
            "suggested"
        ],
        "argmax_mismatch_rate": summary["correctness"]["argmax_mismatch_rate"][
            "suggested"
        ],
    }
    assert summary["correctness"]["mean_abs_logprob_diff"]["all_observed_zero"] is True
    assert thr["mean_abs_logprob_diff"] == float(
        config.BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF
    )
    assert correctness_would_pass(
        {
            "mean_abs_logprob_diff": 0.0,
            "max_abs_logprob_diff": 0.0,
            "argmax_mismatch_rate": 0.0,
        },
        thr,
    )

    tampered = {
        "mean_abs_logprob_diff": 1.0,
        "max_abs_logprob_diff": 1.0,
        "argmax_mismatch_rate": 1.0,
    }
    assert not correctness_would_pass(tampered, thr)
    assert summary["sla_repro"]["outer_p99_ttft_ms_rel_range"] > 0


def test_analyze_rejects_mismatched_digests(tmp_path: Path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_report(digest=DIGEST)), encoding="utf-8")
    other = "sha256:" + ("b" * 64)
    b.write_text(json.dumps(_report(digest=other)), encoding="utf-8")
    with pytest.raises(CalibrationError, match="disagree"):
        analyze_reports([a, b])


def test_analyze_rejects_unequal_baseline_candidate(tmp_path: Path):
    r = _report()
    r["inputs_fingerprint"]["candidate_image_digest"] = "sha256:" + ("c" * 64)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(CalibrationError, match="self-check"):
        analyze_reports([p])


def test_shape_compare_match_and_drift():
    fixture = json.loads(SHAPE_FIXTURE.read_text(encoding="utf-8"))
    ok = compare_shape_to_fixture({"fingerprint": fixture["fingerprint"]})
    assert ok["match"] is True
    assert ok["diffs"] == []

    drifted = json.loads(json.dumps(fixture["fingerprint"]))
    drifted["extra_field"] = "str"
    bad = compare_shape_to_fixture({"fingerprint": drifted})
    assert bad["match"] is False
    assert any("extra_field" in d for d in bad["diffs"])


def test_analyze_runs_dir_with_shape(tmp_path: Path):
    runs = tmp_path / "runs"
    for i in range(1, 3):
        d = runs / f"run-{i:03d}"
        corr = d / "evidence" / "correctness"
        corr.mkdir(parents=True)
        (d / "bench_report.json").write_text(
            json.dumps(_report(mean=0.001 * i, max_d=0.002 * i)),
            encoding="utf-8",
        )
        fixture = json.loads(SHAPE_FIXTURE.read_text(encoding="utf-8"))
        (corr / "completion_response_shape.json").write_text(
            json.dumps({"fingerprint": fixture["fingerprint"]}),
            encoding="utf-8",
        )
    summary = analyze_runs_dir(runs, safety_factor=2.0)
    assert summary["n_reports"] == 2
    assert summary["shape"]["match"] is True
    assert summary["correctness"]["mean_abs_logprob_diff"][
        "observed_max"
    ] == pytest.approx(0.002)


def test_discover_flat_and_numbered(tmp_path: Path):
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "bench_report.json").write_text("{}", encoding="utf-8")
    assert discover_run_dirs(flat) == [flat]

    nested = tmp_path / "nested"
    (nested / "run-002").mkdir(parents=True)
    (nested / "run-001").mkdir(parents=True)
    assert [p.name for p in discover_run_dirs(nested)] == ["run-001", "run-002"]


def test_analyze_correctness_only_skips_sla(tmp_path: Path):
    r = _report(mean=0.01, max_d=0.02, argmax=0.0)
    del r["sla_bench"]
    del r["perf_screen"]
    p = tmp_path / "corr-only.json"
    p.write_text(json.dumps(r), encoding="utf-8")
    summary = analyze_reports([p], safety_factor=2.0)
    assert summary["sla_repro"] is None
    assert summary["correctness"]["mean_abs_logprob_diff"][
        "observed_max"
    ] == pytest.approx(0.01)
    assert summary["correctness"]["mean_abs_logprob_diff"]["suggested"] >= 0.01


def test_correctness_dict_from_summary_sets_calibration_blob():
    summary = {
        "fingerprint": {
            "engine_digest": DIGEST,
            "model_repo": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "trace_sha256": A3A_TRACE_SHA256,
        },
        "correctness": {
            "mean_abs_logprob_diff": {"suggested": 0.05},
            "max_abs_logprob_diff": {"suggested": 0.1},
            "argmax_mismatch_rate": {"suggested": 0.01},
            "observed": {"mean_abs_logprob_diff": [0.02]},
        },
    }
    fp = {
        "model_repo": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
        "baseline_engine_image_digest": DIGEST,
        "trace_sha256": A3A_TRACE_SHA256,
        "serve_args": [],
    }
    out = correctness_dict_from_summary(summary, campaign_fingerprint=fp)
    assert out["thresholds"]["mean_abs_logprob_diff"] == 0.05
    planned = out["num_prompts"] * out["max_new_tokens"]
    assert out["min_positions_compared"] == min(APPLY_DEFAULT_MIN_POSITIONS, planned)
    assert out["calibration"]["thresholds"] == out["thresholds"]
    assert out["calibration"]["fingerprint"] == fp


def test_correctness_dict_from_summary_caps_min_positions_to_planned():
    summary = {
        "correctness": {
            "mean_abs_logprob_diff": {"suggested": 0.05},
            "max_abs_logprob_diff": {"suggested": 0.1},
            "argmax_mismatch_rate": {"suggested": 0.01},
        },
    }
    out = correctness_dict_from_summary(
        summary,
        existing_correctness={
            "num_prompts": 8,
            "max_new_tokens": 32,
            "min_positions_compared": 1024,
        },
        campaign_fingerprint={"model_repo": "x"},
    )
    assert out["min_positions_compared"] == 256


def test_assert_summary_matches_campaign_rejects_mismatch():
    from bench.calibrate import assert_summary_matches_campaign, CalibrationError

    bench = {
        "baseline_engine_image_digest": DIGEST,
        "model": {
            "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
            "hf_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
        },
    }
    ok = {
        "fingerprint": {
            "engine_digest": DIGEST,
            "model_repo": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "trace_sha256": A3A_TRACE_SHA256,
        }
    }
    assert_summary_matches_campaign(ok, bench=bench, trace_sha256=A3A_TRACE_SHA256)
    bad = {
        "fingerprint": {
            **ok["fingerprint"],
            "model_revision": "deadbeef",
        }
    }
    with pytest.raises(CalibrationError, match="model_revision"):
        assert_summary_matches_campaign(bad, bench=bench, trace_sha256=A3A_TRACE_SHA256)
