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
    # bench=None campaign: settled gates job, and nothing queued for a round
    from db.connection import db_connection

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM submission_jobs WHERE submission_id = %s",
                (str(sid),),
            )
            jobs = cur.fetchall()
    assert jobs == [("done",)]
    assert "bench_queued" not in events


def test_e2e_mock_round_runs_end_to_end(tmp_path, monkeypatch):
    """One --mock-bench round: claim, run, complete. No GPU, no Docker pull."""
    import config
    from campaign.models import SLA
    from campaign.seed import build_seed_bench_spec
    from round.create import create_due_rounds
    from round.store import claim_pending_round, get_round, list_round_entries
    from worker.round_job import process_round

    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.test")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "ALLOW_MOCK_BENCH", True)

    trace = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench" / "sample_trace.json"
    )
    trace_sha = "sha256:" + hashlib.sha256(trace.read_bytes()).hexdigest()
    engine_digest = "sha256:" + "a" * 64
    image_a = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "1" * 64
    image_b = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "2" * 64

    profile_id = insert_profile("e2e", {"fixture": True})
    campaign_id = uuid4()
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo="https://example/baseline.git",
        baseline_commit="deadbeef",
        base_image_digest="sha256:" + "d" * 64,
        gpu_skus=["H200"],
        workload_trace_sha256=trace_sha,
        workload_trace_url=f"file://{trace}",
        sla=SLA(p99_ttft_ms=2000.0, p99_itl_ms=50.0),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="open",
        bench=build_seed_bench_spec(baseline_engine_image_digest=engine_digest),
        scoring_rule={"name": "median_e2e_speedup"},
    )
    insert_campaign(manifest)

    def _queue(image_ref: str, block: int) -> str:
        sid = insert_submission(
            campaign_id=campaign_id,
            patch_hash="sha256:" + uuid4().hex * 2,
            hotkey="5FakesHotkeyForE2ETesting000000000000000000000",
            baseline_commit="deadbeef",
            retrieval_url=(
                f"https://cdn.test/stage0/campaigns/{campaign_id}/p/e2e.diff"
            ),
            commit_block=block,
        )
        assert sid is not None
        from db.connection import db_connection

        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE submissions SET engine_image_ref = %s WHERE id = %s",
                    (image_ref, str(sid)),
                )
                cur.execute(
                    """
                    UPDATE submission_events
                    SET created_at = now() - make_interval(secs => 40060)
                    WHERE submission_id = %s
                    """,
                    (str(sid),),
                )
                cur.execute(
                    """
                    INSERT INTO submission_events
                        (submission_id, state, detail, created_at)
                    VALUES (%s, 'bench_queued', '{}'::jsonb,
                            now() - make_interval(secs => 40000))
                    """,
                    (str(sid),),
                )
        return str(sid)

    _queue(image_a, 10)
    _queue(image_b, 11)

    class _FakeSubtensor:
        block = 1000

        def get_block_hash(self, number: int) -> str:
            return "0x" + f"{number:064x}"

    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    outcome = process_round(
        claimed,
        mock_bench=True,
        work_root=tmp_path / "round-work",
    )
    assert outcome == "ok"
    settled = get_round(claimed["id"])
    assert settled is not None
    assert settled["status"] == "complete"
    assert settled["baseline_drift"] is not None
    entries = list_round_entries(claimed["id"])
    assert entries[0]["role"] == "baseline"
    assert entries[0]["status"] == "scored"
    assert float(entries[0]["score"]) == 0.0
    challengers = [e for e in entries if e["role"] == "challenger"]
    assert len(challengers) == 2
    assert {e["status"] for e in challengers} <= {
        "scored",
        "disqualified",
        "infra_failed",
    }


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
