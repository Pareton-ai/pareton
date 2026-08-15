"""End-to-end Stage 0 with mock fetch + mock build (no Docker/chain)."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from campaign.manifest import build_manifest
from campaign.models import CustomerSignoff, SLA
from campaign.store import (
    get_submission,
    insert_campaign,
    insert_profile,
    insert_submission,
    list_events,
)
from e2e_db import cleanup_e2e_rows, require_e2e_database_url
from worker.pipeline import process_submission

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _bind_e2e_database(monkeypatch: pytest.MonkeyPatch):
    """Point store/connection code at the Neon test branch for this module."""
    url = require_e2e_database_url()
    monkeypatch.setenv("PARETON_DATABASE_URL", url)
    import db.connection as conn

    monkeypatch.setattr(conn, "DATABASE_URL", url)
    monkeypatch.setattr(conn, "_pool", None)
    yield
    cleanup_e2e_rows()


def _patch_for_repo(repo: Path) -> bytes:
    target = repo / "vllm" / "hello.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "vllm/hello.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "base"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    target.write_text("x = 2\n", encoding="utf-8")
    diff = subprocess.run(
        ["git", "diff"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    # restore working tree so apply --check works against HEAD
    subprocess.run(["git", "checkout", "--", "."], cwd=repo, check=True)
    return diff


def test_e2e_mock_commitment_to_built(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.test")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")

    repo = tmp_path / "baseline"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    patch = _patch_for_repo(repo)
    patch_hash = "sha256:" + hashlib.sha256(patch).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    profile_id = insert_profile("e2e", {"fixture": True})
    now = datetime.now(timezone.utc)
    campaign_id = uuid4()
    url = f"https://cdn.test/stage0/campaigns/{campaign_id}/patches/5FakesHotkeyForE2ETesting000000000000000000000/e2e.diff"
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=str(repo),
        baseline_commit=commit,
        base_image_digest="sha256:" + "d" * 64,
        gpu_skus=["H200"],
        workload_trace_sha256="sha256:" + "e" * 64,
        workload_trace_url="https://cdn.test/trace.json",
        sla=SLA(),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="open",
        customer_signoff=CustomerSignoff(
            approved_manifest_hash="pending",
            approver="test",
            timestamp=now,
        ),
    )
    # fix signoff hash
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=str(repo),
        baseline_commit=commit,
        base_image_digest="sha256:" + "d" * 64,
        gpu_skus=["H200"],
        workload_trace_sha256="sha256:" + "e" * 64,
        workload_trace_url="https://cdn.test/trace.json",
        sla=SLA(),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="open",
        customer_signoff=CustomerSignoff(
            approved_manifest_hash=manifest.manifest_hash,
            approver="test",
            timestamp=now,
        ),
        manifest_hash=manifest.manifest_hash,
    )
    insert_campaign(manifest)

    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash=patch_hash,
        hotkey="5FakesHotkeyForE2ETesting000000000000000000000",
        baseline_commit=commit,
        retrieval_url=url,
        commit_block=1,
    )
    assert sid is not None

    from campaign.store import claim_next_job

    row = claim_next_job()
    assert row is not None
    assert str(row["id"]) == str(sid)

    result = process_submission(
        row,
        registered_hotkeys=[row["hotkey"]],
        fetcher=lambda _u: patch,
        mock_build=True,
        local_repo=repo,
        work_root=tmp_path / "gate-work",
    )
    assert result.ok, result.reason
    assert result.state == "built"

    sub = get_submission(patch_hash)
    assert sub is not None
    assert sub["engine_image_ref"]
    assert "@sha256:" in sub["engine_image_ref"]
    events = [e["state"] for e in list_events(sid)]
    assert "committed" in events
    assert "fetched" in events
    assert "verified" in events
    assert "applied" in events
    assert "surface_ok" in events
    assert "built" in events
    # bench=None campaign: no bench job enqueued
    from db.connection import db_connection

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT kind, status FROM submission_jobs WHERE submission_id = %s",
                (str(sid),),
            )
            jobs = cur.fetchall()
    assert jobs == [("gates", "done")]


def _make_repo_and_patch(tmp_path):
    repo = tmp_path / "baseline"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    patch = _patch_for_repo(repo)
    patch_hash = "sha256:" + hashlib.sha256(patch).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, patch, patch_hash, commit


def _bench_campaign_spec(trace_path: Path, trace_sha: str) -> dict:
    return {
        "model": {
            "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
            "hf_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "dtype": "bfloat16",
            "quantization": None,
            "max_model_len": 8192,
        },
        "baseline_engine_image_digest": "sha256:" + ("a" * 64),
        "gpu_count": 1,
        "serve_args": None,
        "correctness": {
            "num_prompts": 2,
            "max_new_tokens": 8,
            "thresholds": {
                "mean_abs_logprob_diff": 0.005,
                "max_abs_logprob_diff": 0.05,
                "argmax_mismatch_rate": 0.001,
            },
        },
        "perf_screen": {
            "num_requests": 2,
            "concurrency": 1,
            "min_throughput_ratio": 1.0,
        },
    }


def _insert_open_campaign(
    *,
    repo,
    commit,
    campaign_id,
    url,
    now,
    bench,
    trace_sha,
    trace_url,
):
    profile_id = insert_profile("e2e-bench", {"fixture": True})
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=str(repo),
        baseline_commit=commit,
        base_image_digest="sha256:" + ("d" * 64),
        gpu_skus=["H200"],
        workload_trace_sha256=trace_sha,
        workload_trace_url=trace_url,
        sla=SLA(p99_ttft_ms=2000.0, p99_itl_ms=50.0),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="open",
        customer_signoff=CustomerSignoff(
            approved_manifest_hash="pending",
            approver="test",
            timestamp=now,
        ),
        bench=bench,
    )
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=str(repo),
        baseline_commit=commit,
        base_image_digest="sha256:" + ("d" * 64),
        gpu_skus=["H200"],
        workload_trace_sha256=trace_sha,
        workload_trace_url=trace_url,
        sla=SLA(p99_ttft_ms=2000.0, p99_itl_ms=50.0),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="open",
        customer_signoff=CustomerSignoff(
            approved_manifest_hash=manifest.manifest_hash,
            approver="test",
            timestamp=now,
        ),
        manifest_hash=manifest.manifest_hash,
        bench=bench,
    )
    insert_campaign(manifest)
    return manifest


def test_e2e_mock_bench_happy_path(tmp_path, monkeypatch):
    import config
    from campaign.store import (
        claim_next_job,
        list_bench_reports,
        list_bench_summaries,
    )
    from db.connection import db_connection
    from worker.bench_job import process_bench_job

    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.test")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "ALLOW_MOCK_BENCH", True)
    monkeypatch.setenv("PARETON_ALLOW_MOCK_BENCH", "1")

    repo, patch, patch_hash, commit = _make_repo_and_patch(tmp_path)
    sample_trace = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "sample_trace.json"
    )
    trace_sha = "sha256:" + hashlib.sha256(sample_trace.read_bytes()).hexdigest()
    trace_url = f"file://{sample_trace.resolve()}"
    now = datetime.now(timezone.utc)
    campaign_id = uuid4()
    url = f"https://cdn.test/stage0/campaigns/{campaign_id}/patches/5FakesHotkeyForE2ETesting000000000000000000001/e2e.diff"
    bench = _bench_campaign_spec(sample_trace, trace_sha)
    _insert_open_campaign(
        repo=repo,
        commit=commit,
        campaign_id=campaign_id,
        url=url,
        now=now,
        bench=bench,
        trace_sha=trace_sha,
        trace_url=trace_url,
    )
    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash=patch_hash,
        hotkey="5FakesHotkeyForE2ETesting000000000000000000001",
        baseline_commit=commit,
        retrieval_url=url,
        commit_block=1,
    )
    row = claim_next_job(kind="gates")
    assert row is not None
    result = process_submission(
        row,
        registered_hotkeys=[row["hotkey"]],
        fetcher=lambda _u: patch,
        mock_build=True,
        local_repo=repo,
        work_root=tmp_path / "gate-work",
    )
    assert result.ok
    sub = get_submission(patch_hash)
    assert "@sha256:" in sub["engine_image_ref"]

    brow = claim_next_job(kind="bench")
    assert brow is not None
    # Stabilize SLA verdict for CI (A/B are deterministic; C is wall-clock noisy).
    from bench.main import run_bench as real_run_bench

    def stable_run(request_path, output_dir, **kwargs):
        code = real_run_bench(request_path, output_dir, **kwargs)
        report_path = Path(output_dir) / "bench_report.json"
        if report_path.is_file():
            import json

            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("correctness", {}).get("verdict") == "pass"
                and report.get("perf_screen", {}).get("verdict") == "pass"
                and isinstance(report.get("sla_bench"), dict)
            ):
                report["sla_bench"]["verdict"] = "pass"
                report["verdict"] = "pass"
                report_path.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
        return code

    outcome = process_bench_job(
        brow,
        mock_bench=True,
        mock_baseline_token_latency_s=0.03,
        mock_candidate_token_latency_s=0.015,
        work_root=tmp_path / "bench-work",
        run_bench_fn=stable_run,
    )
    assert outcome == "ok"
    states = [e["state"] for e in list_events(sid)]
    assert "correct" in states
    assert "screened" in states
    assert "benched" in states
    reports = list_bench_reports(sid)
    assert len(reports) == 3
    assert all(r["mock"] is True for r in reports)
    assert all(r["evidence_s3_url"] is None for r in reports)
    assert len({r["task_id"] for r in reports}) == 1
    summaries = list_bench_summaries(campaign_id)
    assert summaries.get(str(sid)) == "pass"
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT kind, status FROM submission_jobs WHERE submission_id = %s ORDER BY kind",
                (str(sid),),
            )
            assert cur.fetchall() == [("bench", "done"), ("gates", "done")]


def test_e2e_mock_bench_adversarial_and_terminal_guard(tmp_path, monkeypatch):
    import config
    from campaign.store import claim_next_job, enqueue_bench_job, list_bench_reports
    from worker.bench_job import process_bench_job

    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.test")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "ALLOW_MOCK_BENCH", True)

    repo, patch, patch_hash, commit = _make_repo_and_patch(tmp_path)
    sample_trace = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "sample_trace.json"
    )
    trace_sha = "sha256:" + hashlib.sha256(sample_trace.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    campaign_id = uuid4()
    url = f"https://cdn.test/stage0/campaigns/{campaign_id}/patches/5FakesHotkeyForE2ETesting000000000000000000002/adv.diff"
    _insert_open_campaign(
        repo=repo,
        commit=commit,
        campaign_id=campaign_id,
        url=url,
        now=now,
        bench=_bench_campaign_spec(sample_trace, trace_sha),
        trace_sha=trace_sha,
        trace_url=f"file://{sample_trace.resolve()}",
    )
    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash=patch_hash,
        hotkey="5FakesHotkeyForE2ETesting000000000000000000002",
        baseline_commit=commit,
        retrieval_url=url,
        commit_block=2,
    )
    grow = claim_next_job(kind="gates")
    assert process_submission(
        grow,
        registered_hotkeys=[grow["hotkey"]],
        fetcher=lambda _u: patch,
        mock_build=True,
        local_repo=repo,
        work_root=tmp_path / "gate-work",
    ).ok
    brow = claim_next_job(kind="bench")
    outcome = process_bench_job(
        brow,
        mock_bench=True,
        mock_tampered_candidate=True,
        work_root=tmp_path / "bench-work",
    )
    assert outcome == "ok"
    states = [e["state"] for e in list_events(sid)]
    assert "rejected" in states
    assert "screened" not in states
    assert "benched" not in states
    reports = list_bench_reports(sid)
    assert len(reports) == 1
    assert reports[0]["stage"] == "correctness"
    assert enqueue_bench_job(sid) is False


def test_e2e_mock_bench_candidate_engine_failure(tmp_path, monkeypatch):
    import config
    from bench.lifecycle import EngineError
    from campaign.store import claim_next_job
    from worker.bench_job import process_bench_job

    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.test")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "ALLOW_MOCK_BENCH", True)

    repo, patch, patch_hash, commit = _make_repo_and_patch(tmp_path)
    sample_trace = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "sample_trace.json"
    )
    trace_sha = "sha256:" + hashlib.sha256(sample_trace.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    campaign_id = uuid4()
    url = f"https://cdn.test/stage0/campaigns/{campaign_id}/patches/5FakesHotkeyForE2ETesting000000000000000000003/eng.diff"
    _insert_open_campaign(
        repo=repo,
        commit=commit,
        campaign_id=campaign_id,
        url=url,
        now=now,
        bench=_bench_campaign_spec(sample_trace, trace_sha),
        trace_sha=trace_sha,
        trace_url=f"file://{sample_trace.resolve()}",
    )
    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash=patch_hash,
        hotkey="5FakesHotkeyForE2ETesting000000000000000000003",
        baseline_commit=commit,
        retrieval_url=url,
        commit_block=3,
    )
    grow = claim_next_job(kind="gates")
    assert process_submission(
        grow,
        registered_hotkeys=[grow["hotkey"]],
        fetcher=lambda _u: patch,
        mock_build=True,
        local_repo=repo,
        work_root=tmp_path / "gate-work",
    ).ok
    brow = claim_next_job(kind="bench")

    def boom(*_a, **_k):
        raise EngineError("cand crash", error_role="candidate")

    monkeypatch.setattr("bench.main.run_all_modules", boom)
    outcome = process_bench_job(
        brow, mock_bench=True, work_root=tmp_path / "bench-work"
    )
    assert outcome == "ok"
    rejected = [e for e in list_events(sid) if e["state"] == "rejected"]
    assert rejected
    assert rejected[-1]["detail"].get("reason") == "fail_engine_candidate"


def test_e2e_sibling_job_isolation(tmp_path, monkeypatch):
    import config
    from campaign.store import claim_next_job, enqueue_bench_job, set_job_status
    from db.connection import db_connection

    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.test")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")

    repo, patch, patch_hash, commit = _make_repo_and_patch(tmp_path)
    sample_trace = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "sample_trace.json"
    )
    trace_sha = "sha256:" + hashlib.sha256(sample_trace.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    campaign_id = uuid4()
    url = f"https://cdn.test/stage0/campaigns/{campaign_id}/patches/5FakesHotkeyForE2ETesting000000000000000000004/sib.diff"
    _insert_open_campaign(
        repo=repo,
        commit=commit,
        campaign_id=campaign_id,
        url=url,
        now=now,
        bench=_bench_campaign_spec(sample_trace, trace_sha),
        trace_sha=trace_sha,
        trace_url=f"file://{sample_trace.resolve()}",
    )
    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash=patch_hash,
        hotkey="5FakesHotkeyForE2ETesting000000000000000000004",
        baseline_commit=commit,
        retrieval_url=url,
        commit_block=4,
    )
    grow = claim_next_job(kind="gates")
    assert process_submission(
        grow,
        registered_hotkeys=[grow["hotkey"]],
        fetcher=lambda _u: patch,
        mock_build=True,
        local_repo=repo,
        work_root=tmp_path / "gate-work",
    ).ok
    assert enqueue_bench_job(sid) is False  # already enqueued by pipeline
    set_job_status(sid, "failed", kind="gates", last_error="touch_gates")
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT kind, status, last_error FROM submission_jobs WHERE submission_id = %s ORDER BY kind",
                (str(sid),),
            )
            rows = cur.fetchall()
    assert ("gates", "failed", "touch_gates") in rows
    assert ("bench", "pending", None) in rows
    set_job_status(sid, "failed", kind="bench", last_error="touch_bench")
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT kind, status, last_error FROM submission_jobs WHERE submission_id = %s ORDER BY kind",
                (str(sid),),
            )
            rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    assert rows["gates"] == ("failed", "touch_gates")
    assert rows["bench"] == ("failed", "touch_bench")


def test_e2e_mock_bench_cross_env_speedup_api_verdict(tmp_path, monkeypatch):
    """Multi-SKU floor reject: stage rows pass, event-sourced API verdict rejects."""
    import json

    import config
    from campaign.store import (
        claim_next_job,
        derive_bench_verdict_from_events,
        list_bench_reports,
        list_bench_summaries,
        list_events,
    )
    from worker.bench_job import process_bench_job
    from bench.validate import sha256_bytes

    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.test")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "ALLOW_MOCK_BENCH", True)

    repo, patch, patch_hash, commit = _make_repo_and_patch(tmp_path)
    sample_trace = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "sample_trace.json"
    )
    trace_sha = "sha256:" + hashlib.sha256(sample_trace.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    campaign_id = uuid4()
    url = f"https://cdn.test/stage0/campaigns/{campaign_id}/patches/5FakesHotkeyForE2ETesting000000000000000000099/xenv.diff"
    bench = _bench_campaign_spec(sample_trace, trace_sha)
    bench["cross_env"] = {
        "aggregate": "min",
        "min_speedup_each": 1.5,
        "speedup_metric": "output_tokens_per_s_ratio",
    }
    profile_id = insert_profile("e2e-xenv", {"fixture": True})
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=str(repo),
        baseline_commit=commit,
        base_image_digest="sha256:" + ("d" * 64),
        gpu_skus=["mock-a", "mock-b"],
        workload_trace_sha256=trace_sha,
        workload_trace_url=f"file://{sample_trace.resolve()}",
        sla=SLA(p99_ttft_ms=2000.0, p99_itl_ms=50.0),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="open",
        customer_signoff=CustomerSignoff(
            approved_manifest_hash="pending",
            approver="test",
            timestamp=now,
        ),
        bench=bench,
    )
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=str(repo),
        baseline_commit=commit,
        base_image_digest="sha256:" + ("d" * 64),
        gpu_skus=["mock-a", "mock-b"],
        workload_trace_sha256=trace_sha,
        workload_trace_url=f"file://{sample_trace.resolve()}",
        sla=SLA(p99_ttft_ms=2000.0, p99_itl_ms=50.0),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="open",
        customer_signoff=CustomerSignoff(
            approved_manifest_hash=manifest.manifest_hash,
            approver="test",
            timestamp=now,
        ),
        manifest_hash=manifest.manifest_hash,
        bench=bench,
    )
    insert_campaign(manifest)
    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash=patch_hash,
        hotkey="5FakesHotkeyForE2ETesting000000000000000000099",
        baseline_commit=commit,
        retrieval_url=url,
        commit_block=99,
    )
    grow = claim_next_job(kind="gates")
    assert process_submission(
        grow,
        registered_hotkeys=[grow["hotkey"]],
        fetcher=lambda _u: patch,
        mock_build=True,
        local_repo=repo,
        work_root=tmp_path / "gate-work",
    ).ok
    brow = claim_next_job(kind="bench")

    def run_fn(request_path, output_dir, **_k):
        req = json.loads(Path(request_path).read_text(encoding="utf-8"))
        sku = req["hardware"]["gpu_sku_expected"]
        ratio = 1.0 if sku == "mock-a" else 2.0
        report = {
            "schema_version": 1,
            "task_id": req["task_id"],
            "verdict": "pass",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:01:00Z",
            "environment": {
                "gpu": [],
                "driver_version": "x",
                "cuda_version": "x",
                "docker_version": "x",
                "harness_version": "0",
                "hostname_hash": "sha256:" + ("e" * 64),
            },
            "inputs_fingerprint": {
                "baseline_image_digest": bench["baseline_engine_image_digest"],
                "candidate_image_digest": brow["engine_image_ref"].split("@", 1)[-1]
                if "@" in str(brow["engine_image_ref"])
                else "sha256:" + ("b" * 64),
                "model_repo": bench["model"]["hf_repo"],
                "model_revision": bench["model"]["hf_revision"],
                "model_weights_sha256": "sha256:" + ("0" * 64),
                "trace_sha256": trace_sha,
                "request_sha256": sha256_bytes(Path(request_path).read_bytes()),
            },
            "correctness": {
                "verdict": "pass",
                "num_prompts": 1,
                "num_positions_compared": 1,
                "mean_abs_logprob_diff": 0.0,
                "max_abs_logprob_diff": 0.0,
                "argmax_mismatch_rate": 0.0,
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
                "repetitions": 1,
                "candidate": {},
                "baseline": {},
                "speedup": {
                    "output_tokens_per_s_ratio": ratio,
                    "requests_per_s_ratio": 1.0,
                    "p99_ttft_ratio": 1.0,
                    "p99_itl_ratio": 1.0,
                    "p99_e2e_ratio": 1.0,
                },
                "cross_rep_variance": {},
                "evidence": "e",
            },
        }
        # candidate digest from built submission
        sub = get_submission(patch_hash)
        dig = str(sub["engine_image_ref"]).split("@", 1)[-1]
        report["inputs_fingerprint"]["candidate_image_digest"] = dig
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "bench_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    outcome = process_bench_job(
        brow,
        mock_bench=True,
        work_root=tmp_path / "bench-work",
        run_bench_fn=run_fn,
    )
    assert outcome == "ok"
    reports = list_bench_reports(sid)
    assert len(reports) == 6
    assert all(r["verdict"] == "pass" for r in reports)
    assert len({r["task_id"] for r in reports}) == 2
    events = list_events(sid)
    assert derive_bench_verdict_from_events(events) == "fail_cross_env_speedup"
    assert list_bench_summaries(campaign_id).get(str(sid)) == "fail_cross_env_speedup"


def test_e2e_campaign_scoped_lookup_disambiguates(tmp_path, monkeypatch):
    """Same patch_hash in two campaigns: scoped routes resolve, bare routes 409."""
    import config
    from campaign.store import count_submission_campaigns, get_submission_for_campaign
    from fastapi.testclient import TestClient

    profile_id = insert_profile("e2e", {"fixture": True})
    now = datetime.now(timezone.utc)
    patch_hash = "sha256:" + hashlib.sha256(b"reused patch").hexdigest()
    campaign_ids = [uuid4(), uuid4()]
    sids = []
    for i, campaign_id in enumerate(campaign_ids):
        manifest = build_manifest(
            campaign_id=campaign_id,
            profile_id=profile_id,
            baseline_repo="https://example/baseline.git",
            baseline_commit="deadbeef",
            base_image_digest="sha256:" + ("d" * 64),
            gpu_skus=["H200"],
            workload_trace_sha256="sha256:" + ("e" * 64),
            workload_trace_url="https://cdn.test/trace.json",
            sla=SLA(),
            scoring_config_sha256=None,
            scoring_config_url=None,
            allowed_paths=["vllm/**"],
            denied_paths=["tests/**"],
            priority_metric="throughput",
            success_threshold=">=10% at SLA",
            status="open",
            customer_signoff=CustomerSignoff(
                approved_manifest_hash="pending",
                approver="test",
                timestamp=now,
            ),
        )
        insert_campaign(manifest)
        sids.append(
            insert_submission(
                campaign_id=campaign_id,
                patch_hash=patch_hash,
                hotkey=f"5FakesHotkeyForE2ETesting00000000000000000000{i}",
                baseline_commit="deadbeef",
                retrieval_url=(
                    f"https://cdn.test/stage0/campaigns/{campaign_id}"
                    "/patches/hk/e2e.diff"
                ),
                commit_block=1,
            )
        )
    assert all(sids)

    assert count_submission_campaigns(patch_hash) == 2
    for campaign_id, sid in zip(campaign_ids, sids):
        row = get_submission_for_campaign(campaign_id, patch_hash)
        assert row is not None
        assert str(row["id"]) == str(sid)
    assert get_submission_for_campaign(uuid4(), patch_hash) is None

    monkeypatch.setattr(config, "BUILD_LOG_DIR", tmp_path)
    log_dir = tmp_path / str(sids[0])
    log_dir.mkdir()
    (log_dir / "build.log").write_text("scoped-ok\n", encoding="utf-8")

    from api import server

    client = TestClient(server.app)
    resp = client.get(f"/v1/campaigns/{campaign_ids[0]}/submissions/{patch_hash}")
    assert resp.status_code == 200
    assert resp.json()["submission"]["id"] == str(sids[0])
    resp = client.get(f"/v1/campaigns/{campaign_ids[1]}/submissions/{patch_hash}")
    assert resp.status_code == 200
    assert resp.json()["submission"]["id"] == str(sids[1])
    resp = client.get(
        f"/v1/campaigns/{campaign_ids[0]}/submissions/{patch_hash}/build-log"
    )
    assert resp.status_code == 200
    assert resp.text.strip() == "scoped-ok"

    assert client.get(f"/v1/submissions/{patch_hash}").status_code == 409
    assert client.get(f"/v1/submissions/{patch_hash}/build-log").status_code == 409


def _insert_draft_calibration_campaign(*, repo, commit, campaign_id, now, bench):
    """Draft campaign with zero submissions: the only state `apply` accepts."""
    profile_id = insert_profile("e2e-calib", {"fixture": True})
    sample_trace = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "sample_trace.json"
    )
    trace_sha = "sha256:" + hashlib.sha256(sample_trace.read_bytes()).hexdigest()
    kwargs = dict(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=str(repo),
        baseline_commit=commit,
        base_image_digest="sha256:" + ("d" * 64),
        gpu_skus=["H200"],
        workload_trace_sha256=trace_sha,
        workload_trace_url=f"file://{sample_trace.resolve()}",
        sla=SLA(p99_ttft_ms=2000.0, p99_itl_ms=50.0),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="draft",
        bench=bench,
    )
    manifest = build_manifest(
        customer_signoff=CustomerSignoff(
            approved_manifest_hash="pending", approver="test", timestamp=now
        ),
        **kwargs,
    )
    manifest = build_manifest(
        customer_signoff=CustomerSignoff(
            approved_manifest_hash=manifest.manifest_hash,
            approver="test",
            timestamp=now,
        ),
        manifest_hash=manifest.manifest_hash,
        **kwargs,
    )
    insert_campaign(manifest)
    return manifest, sample_trace, trace_sha


def _campaign_row(campaign_id):
    from psycopg2.extras import RealDictCursor

    from db.connection import db_connection

    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM campaigns WHERE id = %s", (str(campaign_id),))
            row = dict(cur.fetchone())
    row["engine_image_ref"] = "ghcr.io/pareton/e2e@sha256:" + ("c" * 64)
    return row


def test_e2e_calibration_missing_then_applied_unblocks_bench(tmp_path, monkeypatch):
    """Full calibration path against the real DB: gate -> prepare -> analyze -> apply.

    The mock bench tests all run with ``require_calibration=False``, so this is
    the only e2e coverage of the enforcement and apply code paths.
    """
    import json

    import config
    from bench.calibrate import (
        analyze_reports,
        correctness_dict_from_summary,
        prepare_campaign_calibration_request,
    )
    from campaign.store import apply_campaign_correctness_calibration
    from test_calibrate import _report
    from worker.bench_job import (
        build_bench_request_dict,
        campaign_calibration_fingerprint,
    )

    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")

    repo, _patch, _patch_hash, commit = _make_repo_and_patch(tmp_path)
    now = datetime.now(timezone.utc)
    campaign_id = uuid4()
    sample_trace = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "sample_trace.json"
    )
    trace_sha = "sha256:" + hashlib.sha256(sample_trace.read_bytes()).hexdigest()
    bench = _bench_campaign_spec(sample_trace, trace_sha)
    assert "calibration" not in bench["correctness"]
    manifest, sample_trace, trace_sha = _insert_draft_calibration_campaign(
        repo=repo, commit=commit, campaign_id=campaign_id, now=now, bench=bench
    )
    old_hash = manifest.manifest_hash

    # 1. An uncalibrated campaign must refuse a real (non-mock) bench.
    row = _campaign_row(campaign_id)
    with pytest.raises(Exception) as excinfo:
        build_bench_request_dict(
            row,
            trace_path=str(sample_trace),
            gpu_sku="H200",
            require_calibration=True,
        )
    assert "calibration_missing" in str(excinfo.value)

    # ...while the mock path still builds, which is why the mock e2e tests
    # cannot catch a regression in the block above.
    assert build_bench_request_dict(row, trace_path=str(sample_trace), gpu_sku="H200")[
        "mode"
    ] in ("full", "correctness", "all")

    # 2. prepare reads the campaign from the DB and pins baseline == candidate.
    prep_dir = tmp_path / "calib-prep"
    req = prepare_campaign_calibration_request(
        campaign_id=str(campaign_id),
        output_dir=prep_dir,
        fetcher=lambda _u: sample_trace.read_bytes(),
    )
    assert req["mode"] == "correctness"
    assert req["engines"]["baseline"]["image"] == req["engines"]["candidate"]["image"]
    assert (prep_dir / "bench_request.json").is_file()

    # 3. analyze three agreeing self-check reports into suggested thresholds.
    report_paths = []
    for i, (mean, max_d, argmax) in enumerate(
        [(0.002, 0.02, 0.0), (0.0025, 0.03, 0.0004), (0.0018, 0.025, 0.0002)]
    ):
        rep = _report(mean=mean, max_d=max_d, argmax=argmax)
        rep["inputs_fingerprint"]["trace_sha256"] = trace_sha
        p = tmp_path / f"report-{i}.json"
        p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        report_paths.append(p)
    summary = analyze_reports(report_paths)

    # 4. apply writes thresholds, recomputes manifest_hash, refreshes signoff.
    fingerprint = campaign_calibration_fingerprint(bench, row)
    correctness = correctness_dict_from_summary(
        summary,
        existing_correctness=bench["correctness"],
        campaign_fingerprint=fingerprint,
    )
    new_hash = apply_campaign_correctness_calibration(campaign_id, correctness)
    assert new_hash != old_hash

    applied = _campaign_row(campaign_id)
    stored = applied["bench"]
    if isinstance(stored, str):
        stored = json.loads(stored)
    assert stored["correctness"]["calibration"]["fingerprint"] == fingerprint
    assert applied["manifest_hash"] == new_hash
    signoff = applied["customer_signoff"]
    if isinstance(signoff, str):
        signoff = json.loads(signoff)
    assert signoff["approved_manifest_hash"] == new_hash

    # 5. the same bench request now builds, with the calibrated thresholds.
    request = build_bench_request_dict(
        applied,
        trace_path=str(sample_trace),
        gpu_sku="H200",
        require_calibration=True,
    )
    applied_thresholds = request["correctness"]["thresholds"]
    suggested = stored["correctness"]["thresholds"]
    assert applied_thresholds["mean_abs_logprob_diff"] == pytest.approx(
        suggested["mean_abs_logprob_diff"]
    )
    assert applied_thresholds["max_abs_logprob_diff"] == pytest.approx(
        suggested["max_abs_logprob_diff"]
    )

    # 6. a serve_args change invalidates the calibration instead of silently
    #    scoring against thresholds measured on a different engine config.
    stale = dict(applied)
    stale_bench = json.loads(json.dumps(stored))
    stale_bench["serve_args"] = ["--no-enable-prefix-caching"]
    stale["bench"] = stale_bench
    with pytest.raises(Exception) as excinfo:
        build_bench_request_dict(
            stale,
            trace_path=str(sample_trace),
            gpu_sku="H200",
            require_calibration=True,
        )
    assert "calibration_stale" in str(excinfo.value)
