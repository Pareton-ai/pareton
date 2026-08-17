"""Unit tests for z-score calibration rejects (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from bench.calibrate import CalibrationError, analyze_z_calibration
from bench.promote import PROMOTION_METRICS

pytestmark = pytest.mark.unit

DIGEST = "sha256:" + ("a" * 64)


def _sla_engine(*, p99_ttft: float, p99_itl: float = 10.0) -> dict:
    lat_t = {"p50": 50.0, "p95": 80.0, "p99": p99_ttft}
    lat_i = {"p50": 5.0, "p95": 8.0, "p99": p99_itl}
    return {
        "ttft_ms": lat_t,
        "itl_ms": lat_i,
        "e2e_ms": {"p50": 100.0, "p95": 120.0, "p99": 150.0},
        "output_tokens_per_s": 10.0,
        "requests_per_s": 1.0,
        "sla_goodput_ratio": 1.0,
    }


def _report(
    path: Path,
    *,
    trace: str,
    mean: float,
    max_d: float,
    argmax: float,
    ttft: float,
    itl: float,
    throughput: float,
) -> Path:
    doc = {
        "schema_version": 1,
        "task_id": "550e8400-e29b-41d4-a716-446655440000",
        "verdict": "pass",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "environment": {
            "gpu": [{"index": 0, "name": "Fake", "vbios": "", "memory_mb": 1}],
            "driver_version": "0",
            "cuda_version": "0",
            "docker_version": "0",
            "harness_version": "0",
            "hostname_hash": "h",
        },
        "inputs_fingerprint": {
            "baseline_image_digest": DIGEST,
            "candidate_image_digest": DIGEST,
            "model_repo": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "model_weights_sha256": "sha256:" + ("0" * 64),
            "trace_sha256": trace,
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
            "candidate_output_tokens_per_s": throughput,
            "throughput_ratio": throughput,
            "evidence": "e",
        },
        "sla_bench": {
            "verdict": "pass",
            "repetitions": 3,
            "candidate": _sla_engine(p99_ttft=ttft, p99_itl=itl),
            "baseline": _sla_engine(p99_ttft=ttft, p99_itl=itl),
            "speedup": {
                "output_tokens_per_s_ratio": 1.0,
                "requests_per_s_ratio": 1.0,
                "p99_ttft_ratio": 1.0,
                "p99_itl_ratio": 1.0,
                "p99_e2e_ratio": 1.0,
            },
            "cross_rep_variance": {
                "p99_ttft_ms_rel_range": 0.02,
                "p99_itl_ms_rel_range": 0.01,
                "p99_e2e_ms_rel_range": 0.01,
            },
            "evidence": "e",
        },
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _trace(i: int) -> str:
    return "sha256:" + (f"{i:x}" * 64)[:64]


def test_analyze_z_rejects_repeats(tmp_path: Path):
    paths = [
        _report(
            tmp_path / "a.json",
            trace=_trace(1),
            mean=0.01,
            max_d=0.02,
            argmax=0.0,
            ttft=100,
            itl=10,
            throughput=1.0,
        ),
        _report(
            tmp_path / "b.json",
            trace=_trace(1),
            mean=0.02,
            max_d=0.03,
            argmax=0.001,
            ttft=110,
            itl=11,
            throughput=1.01,
        ),
    ]
    with pytest.raises(CalibrationError, match="repeated traces"):
        analyze_z_calibration(paths, min_samples=2)


def test_analyze_z_rejects_zero_variance(tmp_path: Path):
    paths = [
        _report(
            tmp_path / f"{i}.json",
            trace=_trace(i + 1),
            mean=0.01,
            max_d=0.02,
            argmax=0.0,
            ttft=100.0,
            itl=10.0,
            throughput=1.0,
        )
        for i in range(3)
    ]
    with pytest.raises(CalibrationError, match="zero variance"):
        analyze_z_calibration(paths, min_samples=3)


def test_analyze_z_accepts_varied_traces(tmp_path: Path):
    paths = []
    for i in range(3):
        paths.append(
            _report(
                tmp_path / f"{i}.json",
                trace=_trace(i + 1),
                mean=0.01 * (i + 1),
                max_d=0.02 * (i + 1),
                argmax=0.001 * i,
                ttft=100.0 + i,
                itl=10.0 + i,
                throughput=1.0 + 0.01 * i,
            )
        )
    summary = analyze_z_calibration(paths, min_samples=3)
    assert summary["n_reports"] == 3
    assert set(summary["metrics"]) == set(PROMOTION_METRICS)
    for _name, block in summary["metrics"].items():
        assert block["std"] > 0
        assert block["n"] == 3
    assert "p99_ttft_ms" not in summary["metrics"]


def test_analyze_z_default_floor_is_config_min_samples(tmp_path: Path):
    """No --min-samples falls back to PARETON_CALIB_MIN_SAMPLES."""
    paths = [
        _report(
            tmp_path / f"{i}.json",
            trace=_trace(i + 1),
            mean=0.01 * (i + 1),
            max_d=0.02 * (i + 1),
            argmax=0.001 * i,
            ttft=100.0 + i,
            itl=10.0 + i,
            throughput=1.0 + 0.01 * i,
        )
        for i in range(config.CALIB_MIN_SAMPLES - 1)
    ]
    with pytest.raises(
        CalibrationError, match=f"need at least {config.CALIB_MIN_SAMPLES}"
    ):
        analyze_z_calibration(paths)


def test_prepare_generates_distinct_traces(tmp_path: Path):
    from types import SimpleNamespace

    from bench.calibrate import prepare_pool_calibration_requests
    from campaign.models import SLA

    rows = [{"trajectory": [{"role": "user", "text": f"prompt-{i}"}]} for i in range(8)]
    rule = {
        "type": "hf_rows",
        "seed_block_offset": 1,
        "dataset": "nebius/SWE-agent-trajectories",
        "revision": "deadbeef" * 5,
        "config": "default",
        "split": "train",
        "n_rows": 8,
        "n_prompts": 3,
        "max_tokens": 128,
        "algo_version": 1,
    }
    manifest = SimpleNamespace(
        bench={
            "model": {
                "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
                "hf_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
                "dtype": "bfloat16",
                "quantization": None,
                "max_model_len": 8192,
            },
            "baseline_engine_image_digest": DIGEST,
            "gpu_count": 1,
            "serve_args": [],
            "correctness": {"num_prompts": 8, "max_new_tokens": 8},
        },
        gpu_skus=["H200"],
        sampling_rule=rule,
        workload_trace_sha256=DIGEST,
        workload_trace_url="file:///unused.json",
        sla=SLA(p99_ttft_ms=1e9, p99_itl_ms=1e9),
    )
    reqs = prepare_pool_calibration_requests(
        campaign_id="11111111-1111-1111-1111-111111111111",
        output_dir=tmp_path,
        max_samples=3,
        get_campaign_fn=lambda _cid: manifest,
        row_fetcher=lambda idx: rows[idx],
    )
    assert len(reqs) == 3
    shas = [r["workload_trace"]["sha256"] for r in reqs]
    assert len(set(shas)) == 3
    assert (tmp_path / "sample-000" / "workload_trace.json").is_file()
    assert (tmp_path / "sample-000" / "bench_request.json").is_file()
    assert reqs[0]["model"]["hf_repo"] == "Qwen/Qwen2.5-7B-Instruct"
    assert reqs[0]["hardware"]["gpu_count"] == 1
    assert reqs[0]["mode"] == "all"
    assert reqs[0]["correctness"]["num_prompts"] == 3
    assert reqs[0]["perf_screen"]["num_requests"] == 3
