"""Unit tests for WS-D bench request builder, binding, and outcome table."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from bench.lifecycle import EngineError
from bench.validate import (
    sha256_bytes,
    validate_bench_request_dict,
    validate_report_dict,
)
from campaign.manifest import (
    build_manifest,
    compute_manifest_hash,
    freeze_manifest_fields,
)
from campaign.models import SLA
from worker.bench_job import (
    REASON_CANDIDATE_NOT_PINNED,
    BenchInfraError,
    bind_report_to_run,
    build_bench_request_dict,
    materialize_trace,
    process_bench_job,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"
TRACE_SHA = "sha256:43953c2732b7216909f2b661b28cdee67232def6521aeb58035a7cf9d92aca9d"
BASE_DIGEST = "sha256:" + ("a" * 64)
CAND_DIGEST = "sha256:" + ("b" * 64)
FIXED_TASK = "11111111-1111-4111-8111-111111111111"


def _bench_spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "model": {
            "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
            "hf_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "dtype": "bfloat16",
            "quantization": None,
            "max_model_len": 8192,
        },
        "baseline_engine_image_digest": BASE_DIGEST,
        "gpu_count": 1,
        "serve_args": ["--enable-prefix-caching"],
        "correctness": None,
        "perf_screen": None,
    }
    spec.update(overrides)
    return spec


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": str(uuid4()),
        "job_id": 42,
        "engine_image_ref": f"ghcr.io/pareton-ai/pareton-engine@{CAND_DIGEST}",
        "bench": _bench_spec(),
        "sla": {"p99_ttft_ms": 2000.0, "p99_itl_ms": 50.0},
        "workload_trace_url": f"file://{SAMPLE_TRACE.resolve()}",
        "workload_trace_sha256": TRACE_SHA,
        "gpu_skus": ["H200-SXM-141GB"],
        "patch_hash": "sha256:" + ("d" * 64),
    }
    row.update(overrides)
    return row


def _env() -> dict[str, Any]:
    return {
        "gpu": [],
        "driver_version": "x",
        "cuda_version": "x",
        "docker_version": "x",
        "harness_version": "0",
        "hostname_hash": "sha256:" + ("e" * 64),
    }


def _report(
    *,
    task_id: str,
    request_bytes: bytes,
    verdict: str = "pass",
    error_role: str | None = None,
    baseline: str = BASE_DIGEST,
    candidate: str = CAND_DIGEST,
    trace_sha: str = TRACE_SHA,
    modules: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema_version": 1,
        "task_id": task_id,
        "verdict": verdict,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "environment": _env(),
        "inputs_fingerprint": {
            "baseline_image_digest": baseline,
            "candidate_image_digest": candidate,
            "model_repo": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "model_weights_sha256": "sha256:" + ("0" * 64),
            "trace_sha256": trace_sha,
            "request_sha256": sha256_bytes(request_bytes),
        },
    }
    if error_role:
        out["error_role"] = error_role
    if modules and verdict == "pass":
        out["correctness"] = {
            "verdict": "pass",
            "num_prompts": 1,
            "num_positions_compared": 1,
            "mean_abs_logprob_diff": 0.0,
            "max_abs_logprob_diff": 0.0,
            "argmax_mismatch_rate": 0.0,
            "evidence": "e",
        }
        out["perf_screen"] = {
            "verdict": "pass",
            "baseline_output_tokens_per_s": 1.0,
            "candidate_output_tokens_per_s": 1.0,
            "throughput_ratio": 1.0,
            "evidence": "e",
        }
        out["sla_bench"] = {
            "verdict": "pass",
            "repetitions": 1,
            "candidate": {},
            "baseline": {},
            "speedup": {"output_tokens_per_s": 1.0},
            "cross_rep_variance": {},
            "evidence": "e",
        }
    elif modules and verdict == "fail_correctness":
        out["correctness"] = {
            "verdict": "fail_correctness",
            "num_prompts": 1,
            "num_positions_compared": 1,
            "mean_abs_logprob_diff": 1.0,
            "max_abs_logprob_diff": 1.0,
            "argmax_mismatch_rate": 1.0,
            "evidence": "e",
        }
    elif modules and verdict == "fail_perf_screen":
        out["correctness"] = {
            "verdict": "pass",
            "num_prompts": 1,
            "num_positions_compared": 1,
            "mean_abs_logprob_diff": 0.0,
            "max_abs_logprob_diff": 0.0,
            "argmax_mismatch_rate": 0.0,
            "evidence": "e",
        }
        out["perf_screen"] = {
            "verdict": "fail_perf_screen",
            "baseline_output_tokens_per_s": 2.0,
            "candidate_output_tokens_per_s": 1.0,
            "throughput_ratio": 0.5,
            "evidence": "e",
        }
    elif modules and verdict == "fail_sla":
        out["correctness"] = {
            "verdict": "pass",
            "num_prompts": 1,
            "num_positions_compared": 1,
            "mean_abs_logprob_diff": 0.0,
            "max_abs_logprob_diff": 0.0,
            "argmax_mismatch_rate": 0.0,
            "evidence": "e",
        }
        out["perf_screen"] = {
            "verdict": "pass",
            "baseline_output_tokens_per_s": 1.0,
            "candidate_output_tokens_per_s": 1.0,
            "throughput_ratio": 1.0,
            "evidence": "e",
        }
        out["sla_bench"] = {
            "verdict": "fail_sla",
            "repetitions": 1,
            "candidate": {},
            "baseline": {},
            "speedup": {},
            "cross_rep_variance": {},
            "evidence": "e",
        }
    return out


class _FixedUUID:
    hex = "deadbeef"

    def __str__(self) -> str:
        return FIXED_TASK


def _prepare_injected_report(
    tmp_path: Path,
    row: dict[str, Any],
    *,
    verdict: str,
    error_role: str | None = None,
    modules: bool = True,
    bad_task_id: str | None = None,
) -> dict[str, Any] | None:
    if verdict == "__none__":
        return None
    trace = materialize_trace(
        url=row["workload_trace_url"],
        expected_sha256=TRACE_SHA,
        dest_dir=tmp_path / "pre-trace",
    )
    req = build_bench_request_dict(
        row, task_id=FIXED_TASK, trace_path=str(trace.resolve())
    )
    req_bytes = (json.dumps(req, indent=2) + "\n").encode("utf-8")
    report = _report(
        task_id=bad_task_id or FIXED_TASK,
        request_bytes=req_bytes,
        verdict=verdict,
        error_role=error_role,
        modules=modules,
    )
    return report


def _run_injected(
    monkeypatch,
    *,
    exit_code: int,
    verdict: str,
    tmp_path: Path,
    error_role: str | None = None,
    modules: bool = True,
    bad_task_id: str | None = None,
) -> tuple[str, list, list]:
    events_log: list = []
    finalize_calls: list = []
    fails: list[str] = []

    def fake_finalize(**kwargs):
        finalize_calls.append(kwargs)
        events_log.extend(kwargs["events"])

    def fake_set_job_status(_submission_id, status, **kwargs):
        fails.append(kwargs.get("last_error") or status)

    monkeypatch.setattr("worker.bench_job.finalize_bench_job", fake_finalize)
    monkeypatch.setattr("worker.bench_job.set_job_status", fake_set_job_status)
    monkeypatch.setattr("worker.bench_job.uuid4", lambda: _FixedUUID())

    row = _row()
    report = _prepare_injected_report(
        tmp_path,
        row,
        verdict=verdict,
        error_role=error_role,
        modules=modules,
        bad_task_id=bad_task_id,
    )
    outcome = process_bench_job(
        row,
        mock_bench=True,
        work_root=tmp_path / "work",
        injected_exit=exit_code,
        injected_report=report,
    )
    return outcome, events_log, fails


def test_build_request_valid_and_sla_from_manifest(tmp_path: Path):
    row = _row()
    trace = materialize_trace(
        url=row["workload_trace_url"],
        expected_sha256=TRACE_SHA,
        dest_dir=tmp_path,
    )
    req = build_bench_request_dict(row, task_id=str(uuid4()), trace_path=str(trace))
    validate_bench_request_dict(req)
    assert req["sla_bench"]["thresholds"]["p99_ttft_ms"] == 2000.0
    assert req["sla_bench"]["thresholds"]["p99_itl_ms"] == 50.0
    assert (
        req["engines"]["baseline"]["serve_args"]
        == req["engines"]["candidate"]["serve_args"]
    )
    assert req["engines"]["baseline"]["serve_args"][:4] == [
        "--model",
        "/model",
        "--max-model-len",
        "8192",
    ]
    assert "--enable-prefix-caching" in req["engines"]["baseline"]["serve_args"]
    assert req["engines"]["baseline"]["env"] == {}
    assert req["engines"]["candidate"]["env"] == {}


def test_campaign_bench_overrides(tmp_path: Path):
    row = _row(
        bench=_bench_spec(
            correctness={
                "num_prompts": 3,
                "max_new_tokens": 16,
                "thresholds": {
                    "mean_abs_logprob_diff": 0.01,
                    "max_abs_logprob_diff": 0.1,
                    "argmax_mismatch_rate": 0.01,
                },
            },
            perf_screen={
                "num_requests": 4,
                "concurrency": 2,
                "min_throughput_ratio": 1.2,
            },
        )
    )
    trace = materialize_trace(
        url=row["workload_trace_url"],
        expected_sha256=TRACE_SHA,
        dest_dir=tmp_path,
    )
    req = build_bench_request_dict(row, task_id=str(uuid4()), trace_path=str(trace))
    assert req["correctness"]["num_prompts"] == 3
    assert req["perf_screen"]["min_throughput_ratio"] == 1.2


def test_non_digest_candidate_fails(tmp_path: Path):
    row = _row(engine_image_ref="ghcr.io/pareton-ai/pareton-engine:latest")
    trace = materialize_trace(
        url=row["workload_trace_url"],
        expected_sha256=TRACE_SHA,
        dest_dir=tmp_path,
    )
    with pytest.raises(BenchInfraError) as ei:
        build_bench_request_dict(row, task_id=str(uuid4()), trace_path=str(trace))
    assert ei.value.code == REASON_CANDIDATE_NOT_PINNED


def test_trace_sha_mismatch(tmp_path: Path):
    with pytest.raises(BenchInfraError) as ei:
        materialize_trace(
            url=f"file://{SAMPLE_TRACE.resolve()}",
            expected_sha256="sha256:" + ("f" * 64),
            dest_dir=tmp_path,
        )
    assert ei.value.code == "trace_sha256_mismatch"


def test_https_trace_via_fetcher(tmp_path: Path):
    raw = SAMPLE_TRACE.read_bytes()

    def fetch(url: str) -> bytes:
        assert url.startswith("https://")
        return raw

    path = materialize_trace(
        url="https://cdn.test/trace.json",
        expected_sha256=TRACE_SHA,
        dest_dir=tmp_path,
        fetcher=fetch,
    )
    assert path.is_file()


def test_bind_mismatches():
    task_id = str(uuid4())
    req_bytes = b'{"ok":true}\n'
    report = _report(task_id=task_id, request_bytes=req_bytes)
    bind_report_to_run(
        report,
        request_task_id=task_id,
        executed_request_bytes=req_bytes,
        baseline_digest=BASE_DIGEST,
        candidate_digest=CAND_DIGEST,
        trace_sha256=TRACE_SHA,
    )
    bad = dict(report)
    bad["task_id"] = str(uuid4())
    with pytest.raises(BenchInfraError) as ei:
        bind_report_to_run(
            bad,
            request_task_id=task_id,
            executed_request_bytes=req_bytes,
            baseline_digest=BASE_DIGEST,
            candidate_digest=CAND_DIGEST,
            trace_sha256=TRACE_SHA,
        )
    assert ei.value.code == "bench_report_task_id_mismatch"

    tampered = dict(report)
    tampered["inputs_fingerprint"] = dict(report["inputs_fingerprint"])
    tampered["inputs_fingerprint"]["request_sha256"] = "sha256:" + ("1" * 64)
    with pytest.raises(BenchInfraError) as ei:
        bind_report_to_run(
            tampered,
            request_task_id=task_id,
            executed_request_bytes=req_bytes,
            baseline_digest=BASE_DIGEST,
            candidate_digest=CAND_DIGEST,
            trace_sha256=TRACE_SHA,
        )
    assert ei.value.code == "bench_report_request_sha256_mismatch"

    dig = dict(report)
    dig["inputs_fingerprint"] = dict(report["inputs_fingerprint"])
    dig["inputs_fingerprint"]["baseline_image_digest"] = "sha256:" + ("9" * 64)
    with pytest.raises(BenchInfraError) as ei:
        bind_report_to_run(
            dig,
            request_task_id=task_id,
            executed_request_bytes=req_bytes,
            baseline_digest=BASE_DIGEST,
            candidate_digest=CAND_DIGEST,
            trace_sha256=TRACE_SHA,
        )
    assert ei.value.code == "bench_report_baseline_digest_mismatch"

    tr = dict(report)
    tr["inputs_fingerprint"] = dict(report["inputs_fingerprint"])
    tr["inputs_fingerprint"]["trace_sha256"] = "sha256:" + ("8" * 64)
    with pytest.raises(BenchInfraError) as ei:
        bind_report_to_run(
            tr,
            request_task_id=task_id,
            executed_request_bytes=req_bytes,
            baseline_digest=BASE_DIGEST,
            candidate_digest=CAND_DIGEST,
            trace_sha256=TRACE_SHA,
        )
    assert ei.value.code == "bench_report_trace_sha256_mismatch"


@pytest.mark.parametrize(
    "verdict,exit_code,expect_states,expect_job",
    [
        ("pass", 0, ["correct", "screened", "benched"], "ok"),
        ("fail_correctness", 0, ["rejected"], "ok"),
        ("fail_perf_screen", 0, ["correct", "rejected"], "ok"),
        ("fail_sla", 0, ["correct", "screened", "rejected"], "ok"),
    ],
)
def test_outcome_verdicts(
    monkeypatch, tmp_path, verdict, exit_code, expect_states, expect_job
):
    outcome, events, fails = _run_injected(
        monkeypatch, exit_code=exit_code, verdict=verdict, tmp_path=tmp_path
    )
    assert outcome == expect_job
    assert [s for s, _ in events] == expect_states
    assert fails == []


def test_outcome_error_candidate_rejected(monkeypatch, tmp_path):
    outcome, events, fails = _run_injected(
        monkeypatch,
        exit_code=3,
        verdict="error",
        error_role="candidate",
        modules=False,
        tmp_path=tmp_path,
    )
    assert outcome == "ok"
    assert events[0][0] == "rejected"
    assert events[0][1]["reason"] == "fail_engine_candidate"
    assert fails == []


def test_outcome_error_baseline_infra(monkeypatch, tmp_path):
    outcome, events, fails = _run_injected(
        monkeypatch,
        exit_code=3,
        verdict="error",
        error_role="baseline",
        modules=False,
        tmp_path=tmp_path,
    )
    assert outcome == "bench_engine_baseline_or_unknown"
    assert events == []
    assert fails == ["bench_engine_baseline_or_unknown"]


def test_outcome_exit1_infra(monkeypatch, tmp_path):
    outcome, events, _fails = _run_injected(
        monkeypatch, exit_code=1, verdict="__none__", tmp_path=tmp_path
    )
    assert outcome == "bench_exit_bad_request"
    assert events == []


def test_outcome_exit75_with_report(monkeypatch, tmp_path):
    outcome, events, _fails = _run_injected(
        monkeypatch, exit_code=75, verdict="pass", tmp_path=tmp_path
    )
    assert outcome == "bench_destroy_failed"
    assert [s for s, _ in events] == ["correct", "screened", "benched"]


def test_outcome_exit75_without_report(monkeypatch, tmp_path):
    outcome, events, _fails = _run_injected(
        monkeypatch, exit_code=75, verdict="__none__", tmp_path=tmp_path
    )
    assert outcome == "bench_destroy_failed"
    assert events == []


def test_bind_fail_zero_events(monkeypatch, tmp_path):
    outcome, events, fails = _run_injected(
        monkeypatch,
        exit_code=0,
        verdict="pass",
        bad_task_id="99999999-9999-4999-8999-999999999999",
        tmp_path=tmp_path,
    )
    assert outcome == "bench_report_task_id_mismatch"
    assert events == []
    assert fails == ["bench_report_task_id_mismatch"]


def test_enqueue_terminal_guard(monkeypatch):
    from campaign import store

    monkeypatch.setattr(store, "submission_has_terminal_event", lambda _sid: True)
    assert store.enqueue_bench_job("00000000-0000-0000-0000-000000000000") is False


def test_complete_gates_job_one_transaction(monkeypatch):
    """Gates done + bench enqueue must share one connection/commit (no crash window)."""
    import contextlib

    from campaign import store

    sqls: list[str] = []

    class _Cur:
        def execute(self, sql, args=None):
            sqls.append(" ".join(sql.split()))

        def fetchone(self):
            # terminal check: no row; insert: return id
            if sqls and "INSERT INTO submission_jobs" in sqls[-1]:
                return (1,)
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

    @contextlib.contextmanager
    def fake_db():
        yield _Conn()

    monkeypatch.setattr(store, "db_connection", fake_db)
    assert (
        store.complete_gates_job(
            "00000000-0000-0000-0000-000000000001",
            job_id=9,
            enqueue_bench=True,
        )
        is True
    )
    assert any("status = 'done'" in s for s in sqls)
    assert any("INSERT INTO submission_jobs" in s for s in sqls)
    assert len(sqls) >= 3  # update + terminal check + insert


def test_pipeline_uses_atomic_complete_gates():
    import inspect

    from worker import pipeline

    src = inspect.getsource(pipeline.process_submission)
    assert "complete_gates_job" in src
    assert "enqueue_bench_job(" not in src


def test_manifest_bench_pin_compat():
    now = datetime.now(timezone.utc)
    kwargs = dict(
        campaign_id=None,
        profile_id=None,
        baseline_repo="https://example.com/vllm.git",
        baseline_commit="a" * 40,
        base_image_digest="sha256:" + ("b" * 64),
        gpu_skus=["H200"],
        workload_trace_sha256="sha256:" + ("c" * 64),
        workload_trace_url="https://cdn.test/t.json",
        sla=SLA(p99_ttft_ms=1.0, p99_itl_ms=2.0),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        window_opens_at=now,
        window_closes_at=now,
    )
    fields_no = freeze_manifest_fields(**kwargs)
    assert "bench" not in fields_no
    h0 = compute_manifest_hash(fields_no)
    m0 = build_manifest(**kwargs)
    assert m0.manifest_hash == h0
    fields_yes = freeze_manifest_fields(**kwargs, bench=_bench_spec())
    assert "bench" in fields_yes
    assert compute_manifest_hash(fields_yes) != h0


def test_mock_bench_guard_refuses_without_env(monkeypatch):
    import config
    from worker.main import main

    monkeypatch.setattr(config, "ALLOW_MOCK_BENCH", False)
    assert main(["--mock-bench", "--once"]) == 2


def test_engine_error_writes_error_report(tmp_path, monkeypatch):
    from bench.main import run_bench

    sample_req = json.loads(
        (ROOT / "fixtures" / "bench" / "sample_request.json").read_text()
    )
    sample_req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "bench_request.json"
    req_path.write_text(json.dumps(sample_req), encoding="utf-8")
    out = tmp_path / "out"

    def boom(*_a, **_k):
        raise EngineError("candidate died", error_role="candidate")

    monkeypatch.setattr("bench.main.run_all_modules", boom)
    code = run_bench(req_path, out, mock_engine=True)
    assert code == 3
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["verdict"] == "error"
    assert report["error_role"] == "candidate"
