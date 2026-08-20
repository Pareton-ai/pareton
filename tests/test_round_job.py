"""Unit tests for the round runner. No DB, no GPU, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from bench.validate import sha256_bytes
from campaign.engine import preset
from campaign.models import SLA
from gpu.orchestrate import EXIT_DESTROY_FAILED
from round.store import (
    VOID_LEADER_IMAGE_MISSING,
    VOID_POD_FAILED,
    VOID_POD_PROVISION_FAILED,
    VOID_ROUND_TIMEOUT,
    VOID_TRACE_UNAVAILABLE,
    infra_failed_follow_up_states,
)
from worker.round_job import (
    RoundInfraError,
    bind_report_to_round,
    build_round_request,
    classify_round_failure,
    entry_results_from_report,
    materialize_round_trace,
    remaining_round_budget_s,
)

BASELINE_DIGEST = "sha256:" + "a" * 64
CAND_A = "sha256:" + "b" * 64
CAND_B = "sha256:" + "c" * 64
BASELINE_REF = f"ghcr.io/pareton-ai/pareton-engine@{BASELINE_DIGEST}"
CAND_A_REF = f"ghcr.io/pareton-ai/pareton-engine@{CAND_A}"
CAND_B_REF = f"ghcr.io/pareton-ai/pareton-engine@{CAND_B}"
HF_REV = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
TRACE_SHA = "sha256:" + "e" * 64


def _campaign(*, engine: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        bench={
            "model": {
                "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
                "hf_revision": HF_REV,
                "dtype": "bfloat16",
                "quantization": None,
                "max_model_len": 8192,
            },
            "baseline_engine_image_digest": BASELINE_DIGEST,
            "gpu_count": 1,
            "serve_args": ["--enable-prefix-caching"],
            "correctness": {
                "num_prompts": 2,
                "thresholds": {
                    "min_mean_logprob": -4.0,
                    "min_token_logprob": -12.0,
                    "min_coverage_ratio": 0.5,
                },
            },
        },
        sla=SLA(p99_ttft_ms=2000.0, p99_itl_ms=50.0),
        engine=engine,
        workload_trace_url="file:///tmp/trace.json",
    )


def _round_row(**over) -> dict:
    row = {
        "id": str(uuid4()),
        "campaign_id": str(uuid4()),
        "ordinal": 1,
        "gpu_sku": "H200",
        "sampled_trace_sha256": TRACE_SHA,
        "scoring_rule": {"name": "median_e2e_speedup"},
        "sampling_receipt": {"type": "fixed_trace", "workload_trace_url": "file:///t"},
        "started_at": datetime.now(timezone.utc),
    }
    row.update(over)
    return row


def _entries() -> list[dict]:
    return [
        {
            "id": 1,
            "role": "baseline",
            "submission_id": None,
            "engine_image_ref": BASELINE_REF,
            "hotkey": None,
        },
        {
            "id": 2,
            "role": "leader",
            "submission_id": str(uuid4()),
            "engine_image_ref": CAND_A_REF,
            "hotkey": "5Leader",
        },
        {
            "id": 3,
            "role": "challenger",
            "submission_id": str(uuid4()),
            "engine_image_ref": CAND_B_REF,
            "hotkey": "5Challenger",
        },
    ]


def _write_trace(path, body: bytes | None = None) -> bytes:
    raw = body or (
        b'{"schema_version":1,"meta":{"name":"t"},"requests":['
        b'{"id":"r-1","arrival_offset_ms":0,"max_tokens":8,'
        b'"sampling":{"temperature":0.0,"top_p":1.0},"prompt":"hi"}]}'
    )
    path.write_bytes(raw)
    return raw


def test_build_round_request_maps_candidates_in_entry_order(tmp_path):
    trace = tmp_path / "trace.json"
    raw = _write_trace(trace)
    row = _round_row(sampled_trace_sha256=sha256_bytes(raw))
    req = build_round_request(
        row, _campaign(), _entries(), task_id=str(uuid4()), trace_path=str(trace)
    )
    images = [c["image"] for c in req["engines"]["candidates"]]
    assert images == [CAND_A_REF, CAND_B_REF]
    assert req["engines"]["baseline"]["image"] == BASELINE_REF
    assert req["scoring_rule"] == {"name": "median_e2e_speedup"}
    assert "--max-model-len" in req["engines"]["baseline"]["serve_args"]
    assert "perf_screen" not in req


def test_build_round_request_omits_max_model_len_for_sglang(tmp_path):
    trace = tmp_path / "trace.json"
    raw = _write_trace(trace)
    row = _round_row(sampled_trace_sha256=sha256_bytes(raw))
    req = build_round_request(
        row,
        _campaign(engine=preset("sglang")),
        _entries(),
        task_id=str(uuid4()),
        trace_path=str(trace),
    )
    args = req["engines"]["candidates"][0]["serve_args"]
    assert "--max-model-len" not in args
    assert "--dtype" in args


def test_bind_report_to_round_accepts_positional_candidate_digests():
    task_id = str(uuid4())
    request_bytes = b'{"task_id":"x"}'
    report = {
        "schema_version": 1,
        "task_id": task_id,
        "verdict": "pass",
        "started_at": "t0",
        "finished_at": "t1",
        "environment": {
            "gpu": [],
            "driver_version": "",
            "cuda_version": "",
            "docker_version": "",
            "harness_version": "",
            "hostname_hash": "",
        },
        "inputs_fingerprint": {
            "baseline_image_digest": BASELINE_DIGEST,
            "candidate_image_digest": [CAND_A, CAND_B],
            "model_repo": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": HF_REV,
            "model_weights_sha256": "sha256:" + "0" * 64,
            "trace_sha256": TRACE_SHA,
            "request_sha256": sha256_bytes(request_bytes),
        },
    }
    bind_report_to_round(
        report,
        request_task_id=task_id,
        executed_request_bytes=request_bytes,
        baseline_digest=BASELINE_REF,
        candidate_digests=[CAND_A_REF, CAND_B_REF],
        trace_sha256=TRACE_SHA,
    )


def test_bind_report_to_round_rejects_candidate_digest_mismatch():
    task_id = str(uuid4())
    request_bytes = b"{}"
    report = {
        "schema_version": 1,
        "task_id": task_id,
        "verdict": "pass",
        "started_at": "t0",
        "finished_at": "t1",
        "environment": {
            "gpu": [],
            "driver_version": "",
            "cuda_version": "",
            "docker_version": "",
            "harness_version": "",
            "hostname_hash": "",
        },
        "inputs_fingerprint": {
            "baseline_image_digest": BASELINE_DIGEST,
            "candidate_image_digest": [CAND_B],
            "model_repo": "x/y",
            "model_revision": HF_REV,
            "model_weights_sha256": "sha256:" + "0" * 64,
            "trace_sha256": TRACE_SHA,
            "request_sha256": sha256_bytes(request_bytes),
        },
    }
    with pytest.raises(RoundInfraError) as exc:
        bind_report_to_round(
            report,
            request_task_id=task_id,
            executed_request_bytes=request_bytes,
            baseline_digest=BASELINE_REF,
            candidate_digests=[CAND_A_REF],
            trace_sha256=TRACE_SHA,
        )
    assert exc.value.reason == VOID_POD_FAILED


def test_entry_results_from_report_copies_score_and_sla():
    entries = _entries()
    payload = {
        "index": 0,
        "image_digest": CAND_A,
        "status": "scored",
        "score": 0.35,
        "score_report": {"name": "median_e2e_speedup"},
        "sla": {
            "timings": {
                "r-1": {"ttft_s": 0.1, "itl_s": [0.01], "completion_tokens": 4}
            },
            "cross_rep_variance": {"p99_e2e_ms_rel_range": 0.01},
        },
        "reason": None,
    }
    report = {
        "baseline": {"role": "baseline", "timings": {}, "cross_rep_variance": {}},
        "baseline_drift": 0.001,
        "entries": [
            payload,
            {
                "index": 1,
                "image_digest": CAND_B,
                "status": "disqualified",
                "score": None,
                "reason": "fail_correctness",
                "sla": {"timings": {}, "cross_rep_variance": {}},
            },
        ],
    }
    results = entry_results_from_report(entries, report)
    assert results[0]["role"] == "baseline"
    assert results[0]["status"] == "scored"
    assert results[0]["score"] == 0.0
    assert results[0]["report"]["cross_rep_variance"] == {}
    assert results[1]["status"] == "scored"
    assert results[1]["score"] == 0.35
    assert results[1]["report"]["sla"]["timings"]["r-1"]["ttft_s"] == 0.1
    assert results[2]["status"] == "disqualified"
    assert results[2]["disqualify_reason"] == "fail_correctness"


def test_materialize_round_trace_sha_mismatch(tmp_path):
    trace = tmp_path / "t.json"
    raw = _write_trace(trace)
    campaign = _campaign()
    campaign.workload_trace_url = f"file://{trace}"
    row = _round_row(
        sampled_trace_sha256="sha256:" + "f" * 64,
        sampling_receipt={
            "type": "fixed_trace",
            "workload_trace_url": f"file://{trace}",
        },
    )
    with pytest.raises(RoundInfraError) as exc:
        materialize_round_trace(row, campaign, tmp_path / "out")
    assert exc.value.reason == VOID_TRACE_UNAVAILABLE
    assert sha256_bytes(raw) != row["sampled_trace_sha256"]


def test_infra_failed_requeue_once_accounting():
    assert infra_failed_follow_up_states(False) == ("infra_failed", "bench_queued")
    assert infra_failed_follow_up_states(True) == ("infra_failed",)


def test_failure_matrix_maps_trigger_to_void_reason():
    matrix = [
        ({"provision_error": True}, VOID_POD_PROVISION_FAILED),
        ({"timed_out": True}, VOID_ROUND_TIMEOUT),
        ({"exit_code": 1, "has_report": False, "bound": False}, VOID_POD_FAILED),
        ({"exit_code": 2, "has_report": False, "bound": False}, VOID_POD_FAILED),
        ({"exit_code": 3, "has_report": False, "bound": False}, VOID_POD_FAILED),
        ({"exit_code": 3, "has_report": True, "bound": False}, VOID_POD_FAILED),
        ({"exit_code": 0, "has_report": False, "bound": False}, VOID_POD_FAILED),
        ({"exit_code": 99, "has_report": True, "bound": True}, VOID_POD_FAILED),
        ({"exit_code": 0, "has_report": True, "bound": True}, None),
        (
            {"exit_code": EXIT_DESTROY_FAILED, "has_report": True, "bound": True},
            None,
        ),
    ]
    for kwargs, expected in matrix:
        assert classify_round_failure(**kwargs) == expected


def test_round_infra_error_carries_void_reason():
    err = RoundInfraError(VOID_LEADER_IMAGE_MISSING, "missing")
    assert err.reason == VOID_LEADER_IMAGE_MISSING


def test_process_round_reports_stored_leader_score_as_prev_score(tmp_path, monkeypatch):
    """A disqualified incumbent has no in-round score; the history row must
    still carry the score it won with, from leaders.last_score."""
    import json as _json
    from pathlib import Path

    import worker.round_job as rj
    from round.rank import EVENT_OVERTAKEN

    campaign = _campaign()
    round_row = _round_row(incumbent_submission_id=str(uuid4()))
    entries = _entries()
    captured: dict = {}

    monkeypatch.setattr(rj, "get_campaign", lambda _cid: campaign)
    monkeypatch.setattr(rj, "list_round_entries", lambda _rid: entries)
    monkeypatch.setattr(rj, "remaining_round_budget_s", lambda *a, **k: 100.0)
    monkeypatch.setattr(rj, "get_leader", lambda _cid: {"last_score": 0.20})
    monkeypatch.setattr(rj, "set_round_phase", lambda **k: True)
    monkeypatch.setattr(rj, "touch_round_heartbeat", lambda **k: True)

    def fake_materialize(_row, _campaign, dest_dir, **_k):
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        trace = dest / "trace.json"
        _write_trace(trace)
        return trace

    def fake_build(_row, _campaign, _entries, *, task_id, trace_path):
        captured["task_id"] = task_id
        return {
            "task_id": task_id,
            "engines": {
                "baseline": {"image": BASELINE_REF},
                "candidates": [{"image": CAND_A_REF}, {"image": CAND_B_REF}],
            },
        }

    def fake_bench(request_path, output_dir, **_k):
        request_bytes = Path(request_path).read_bytes()
        report = {
            "schema_version": 1,
            "task_id": captured["task_id"],
            "verdict": "pass",
            "started_at": "t0",
            "finished_at": "t1",
            "environment": {
                "gpu": [],
                "driver_version": "",
                "cuda_version": "",
                "docker_version": "",
                "harness_version": "",
                "hostname_hash": "",
            },
            "inputs_fingerprint": {
                "baseline_image_digest": BASELINE_DIGEST,
                "candidate_image_digest": [CAND_A, CAND_B],
                "model_repo": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": HF_REV,
                "model_weights_sha256": "sha256:" + "0" * 64,
                "trace_sha256": TRACE_SHA,
                "request_sha256": sha256_bytes(request_bytes),
            },
            "baseline": {"role": "baseline", "timings": {}, "cross_rep_variance": {}},
            "entries": [
                {"index": 0, "status": "disqualified", "reason": "wrong_outputs"},
                {"index": 1, "status": "scored", "score": 0.42},
            ],
            "baseline_drift": 0.0,
        }
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bench_report.json").write_text(_json.dumps(report))
        return 0

    def fake_complete(**kwargs):
        captured["decision"] = kwargs["decision"]
        return True

    monkeypatch.setattr(rj, "materialize_round_trace", fake_materialize)
    monkeypatch.setattr(rj, "build_round_request", fake_build)
    monkeypatch.setattr(rj, "complete_round", fake_complete)

    outcome = rj.process_round(
        round_row,
        mock_bench=True,
        work_root=tmp_path / "work",
        run_bench_fn=fake_bench,
    )

    assert outcome == "ok"
    decision = captured["decision"]
    assert decision.event == EVENT_OVERTAKEN
    assert decision.prev_score == 0.20


def test_remaining_budget_voids_when_the_clock_has_run_out():
    started = datetime.now(timezone.utc) - timedelta(seconds=10)
    with pytest.raises(RoundInfraError) as exc:
        remaining_round_budget_s(started, max_duration_s=5)
    assert exc.value.reason == VOID_ROUND_TIMEOUT
    leftover = remaining_round_budget_s(datetime.now(timezone.utc), max_duration_s=100)
    assert leftover > 0
