"""Schema round-trip + invalid-request tests for bench/ (WS-B1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.schemas import BenchRequest
from bench.validate import (
    RequestValidationError,
    load_bench_request,
    sha256_file,
    validate_bench_request_dict,
    validate_report_dict,
    validate_workload_trace_dict,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "fixtures" / "bench" / "sample_request.json"
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"
SYNTHETIC_TRACE = (
    ROOT / "fixtures" / "campaigns" / "synthetic_v0" / "workload_trace.json"
)


def test_sample_request_round_trip():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req = validate_bench_request_dict(raw)
    assert isinstance(req, BenchRequest)
    assert req.schema_version == 1
    assert req.mode == "all"
    assert req.model.quantization is None
    assert "@sha256:" in req.engines.baseline.image
    assert len(req.engines.candidates) == 1
    back = req.to_dict()
    # Round-trip through validator again
    again = validate_bench_request_dict(back)
    assert again.task_id == req.task_id
    assert again.correctness.thresholds.min_mean_logprob == -4.0
    assert again.scoring_rule == {"name": "median_e2e_speedup"}


def test_candidates_must_be_a_non_empty_list():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["engines"]["candidates"] = []
    with pytest.raises(RequestValidationError, match="candidates"):
        validate_bench_request_dict(raw)


def test_every_candidate_image_must_be_digest_pinned():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["engines"]["candidates"].append({"image": "ghcr.io/example/engine:latest"})
    with pytest.raises(RequestValidationError, match=r"candidates\[1\]"):
        validate_bench_request_dict(raw)


def test_scoring_rule_is_required():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw.pop("scoring_rule")
    with pytest.raises(RequestValidationError, match="scoring_rule"):
        validate_bench_request_dict(raw)


def test_load_bench_request_file():
    req, raw = load_bench_request(SAMPLE_REQUEST)
    assert req.task_id == "550e8400-e29b-41d4-a716-446655440000"
    assert raw.startswith(b"{")


def test_sample_trace_valid_and_sha_matches_request():
    trace_obj = json.loads(SAMPLE_TRACE.read_text(encoding="utf-8"))
    trace = validate_workload_trace_dict(trace_obj)
    assert len(trace.requests) == 2
    digest = sha256_file(SAMPLE_TRACE)
    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    assert req["workload_trace"]["sha256"] == digest


def test_synthetic_v0_trace_validates():
    raw = json.loads(SYNTHETIC_TRACE.read_text(encoding="utf-8"))
    assert all("messages" not in r for r in raw["requests"])
    trace = validate_workload_trace_dict(raw)
    assert trace.schema_version == 1
    assert len(trace.requests) == 2
    assert [r.id for r in trace.requests] == ["syn-001", "syn-002"]
    assert [r.arrival_offset_ms for r in trace.requests] == [0, 100]
    for r in trace.requests:
        assert r.prompt and r.prompt.strip()
        assert r.max_tokens == 128
        assert r.sampling.temperature == 0.0
        assert r.sampling.top_p == 1.0


def test_invalid_request_missing_field():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    del raw["engines"]
    with pytest.raises(RequestValidationError, match="missing required fields"):
        validate_bench_request_dict(raw)


def test_invalid_request_bad_mode():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["mode"] = "nope"
    with pytest.raises(RequestValidationError, match="mode"):
        validate_bench_request_dict(raw)


def test_invalid_request_image_without_digest():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["engines"]["baseline"]["image"] = "ghcr.io/example/engine:latest"
    with pytest.raises(RequestValidationError, match="digest"):
        validate_bench_request_dict(raw)


@pytest.mark.parametrize(
    "bad_image",
    [
        # substring "sha256:" but not a digest pin
        "ghcr.io/example/sha256:notadigest/engine:latest",
        # tag that happens to include the substring
        "ghcr.io/example/engine:sha256-fake",
        # @sha256: with wrong length
        "ghcr.io/example/engine@sha256:abcd",
        # @sha256: with non-hex
        "ghcr.io/example/engine@sha256:" + ("g" * 64),
    ],
)
def test_invalid_request_image_loose_sha256_substring(bad_image: str):
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["engines"]["baseline"]["image"] = bad_image
    with pytest.raises(RequestValidationError, match="digest"):
        validate_bench_request_dict(raw)


def test_extract_image_digest_canonicalizes():
    from bench.validate import extract_image_digest

    digest = "sha256:" + ("a" * 64)
    assert extract_image_digest(digest) == digest
    assert extract_image_digest(f"ghcr.io/pareton-ai/pareton-engine@{digest}") == digest
    assert (
        extract_image_digest(f"ghcr.io/pareton-ai/pareton-engine@{digest.upper()}")
        == digest
    )


def test_invalid_request_bad_task_id():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["task_id"] = "not-a-uuid"
    with pytest.raises(RequestValidationError, match="UUID"):
        validate_bench_request_dict(raw)


def test_invalid_request_non_numeric_gpu_count():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["hardware"]["gpu_count"] = None
    with pytest.raises(RequestValidationError):
        validate_bench_request_dict(raw)


def test_invalid_request_coverage_ratio_out_of_range():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["correctness"]["thresholds"]["min_coverage_ratio"] = 1.5
    with pytest.raises(RequestValidationError, match="min_coverage_ratio"):
        validate_bench_request_dict(raw)


def test_invalid_request_token_quantile_out_of_range():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["correctness"]["thresholds"]["min_token_quantile"] = 1.0
    with pytest.raises(RequestValidationError, match="min_token_quantile"):
        validate_bench_request_dict(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_mean_logprob_drop", 0.0),
        ("max_mean_logprob_drop", float("nan")),
    ],
)
def test_invalid_request_relative_logprob_threshold(key: str, value: float):
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["correctness"]["thresholds"][key] = value
    with pytest.raises(RequestValidationError, match=key):
        validate_bench_request_dict(raw)


def test_invalid_request_zero_num_prompts():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["correctness"]["num_prompts"] = 0
    with pytest.raises(RequestValidationError, match="num_prompts"):
        validate_bench_request_dict(raw)


def test_invalid_trace_non_numeric_schema_version():
    with pytest.raises(RequestValidationError):
        validate_workload_trace_dict(
            {"schema_version": "x", "requests": [{"id": "r1"}]}
        )


def test_invalid_trace_duplicate_request_ids():
    with pytest.raises(RequestValidationError, match="duplicate id"):
        validate_workload_trace_dict(
            {
                "schema_version": 1,
                "requests": [
                    {
                        "id": "r1",
                        "arrival_offset_ms": 0,
                        "prompt": "a",
                        "max_tokens": 1,
                        "sampling": {"temperature": 0.0, "top_p": 1.0},
                    },
                    {
                        "id": "r1",
                        "arrival_offset_ms": 1,
                        "prompt": "b",
                        "max_tokens": 1,
                        "sampling": {"temperature": 0.0, "top_p": 1.0},
                    },
                ],
            }
        )


def test_report_dict_validation_accepts_stub_shape():
    report = {
        "schema_version": 1,
        "task_id": "550e8400-e29b-41d4-a716-446655440000",
        "verdict": "error",
        "started_at": "2026-07-14T18:00:00Z",
        "finished_at": "2026-07-14T18:00:00Z",
        "environment": {
            "gpu": [],
            "driver_version": "",
            "cuda_version": "",
            "docker_version": "",
            "harness_version": "0.1.0",
            "hostname_hash": "sha256:" + ("c" * 64),
        },
        "inputs_fingerprint": {
            "baseline_image_digest": "sha256:" + ("a" * 64),
            "candidate_image_digest": ["sha256:" + ("b" * 64)],
            "model_repo": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "model_weights_sha256": "sha256:" + ("0" * 64),
            "trace_sha256": "sha256:" + ("d" * 64),
            "request_sha256": "sha256:" + ("e" * 64),
        },
        "stub_note": "skeleton",
    }
    validate_report_dict(report)  # does not raise


def test_every_engine_must_pin_its_cache_dir():
    # EngineSpec defaults cache_dir to the vLLM path, so an SGLang request that
    # omitted it would mount the cache where SGLang never looks and say nothing.
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    del raw["engines"]["baseline"]["cache_dir"]
    with pytest.raises(RequestValidationError, match="cache_dir"):
        validate_bench_request_dict(raw)


def test_every_candidate_must_pin_its_cache_dir():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    del raw["engines"]["candidates"][0]["cache_dir"]
    with pytest.raises(RequestValidationError, match=r"candidates\[0\]"):
        validate_bench_request_dict(raw)
