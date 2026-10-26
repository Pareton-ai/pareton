"""DB-backed: an LLM review outage defers the gates job, then rejects."""

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
    claim_next_job,
    insert_campaign,
    insert_profile,
    insert_submission,
    list_events,
)
from e2e_db import cleanup_e2e_rows, require_e2e_database_url
from gate.integrity import patch_fingerprint_bytes
from gate.types import GateResult, SubmissionState
from worker import pipeline

pytestmark = pytest.mark.e2e

HOTKEY = "5FakesHotkeyForE2ETesting000000000000000000000"
SPEC = {"policy": "dtype_precision_v1", "model": "kimi-k3"}


@pytest.fixture(autouse=True)
def _bind_e2e_database(monkeypatch: pytest.MonkeyPatch):
    url = require_e2e_database_url()
    monkeypatch.setenv("PARETON_DATABASE_URL", url)
    import db.connection as conn

    monkeypatch.setattr(conn, "DATABASE_URL", url)
    monkeypatch.setattr(conn, "_pool", None)
    yield
    cleanup_e2e_rows()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _seed(tmp_path: Path) -> tuple[Path, bytes, str]:
    repo = tmp_path / "baseline"
    repo.mkdir()
    _git(repo, "init")
    target = repo / "vllm" / "kv.py"
    target.parent.mkdir()
    target.write_text("KV_DTYPE = 'fp8'\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    target.write_text("KV_DTYPE = 'fp4'\n", encoding="utf-8")
    patch = _git(repo, "diff").encode()
    _git(repo, "checkout", "--", ".")
    commit = _git(repo, "rev-parse", "HEAD").strip()

    profile_id = insert_profile("e2e", {"fixture": True})
    campaign_id = uuid4()
    fields = dict(
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
        submission_fee={"amount_tao": "0", "recipient": "5Test"},
        status="open",
        bench={"llm_review": SPEC},
    )
    now = datetime.now(timezone.utc)
    draft = build_manifest(
        **fields,
        customer_signoff=CustomerSignoff(
            approved_manifest_hash="pending", approver="test", timestamp=now
        ),
    )
    insert_campaign(
        build_manifest(
            **fields,
            customer_signoff=CustomerSignoff(
                approved_manifest_hash=draft.manifest_hash,
                approver="test",
                timestamp=now,
            ),
            manifest_hash=draft.manifest_hash,
        )
    )
    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash="sha256:" + hashlib.sha256(patch).hexdigest(),
        hotkey=HOTKEY,
        baseline_commit=commit,
        retrieval_url=(
            f"https://cdn.test/stage0/campaigns/{campaign_id}/patches/{HOTKEY}/e2e.diff"
        ),
        commit_block=1,
        patch_fingerprint=patch_fingerprint_bytes(patch),
    )
    return repo, patch, str(sid)


def _job(sid: str) -> tuple:
    from db.connection import db_connection

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, attempts, retry_after > now(), last_error
                FROM submission_jobs WHERE submission_id = %s
                """,
                (sid,),
            )
            return cur.fetchone()


def _make_due(sid: str) -> None:
    from db.connection import db_connection

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE submission_jobs SET retry_after = now() - interval '1 second'"
                " WHERE submission_id = %s",
                (sid,),
            )


def test_outage_defers_then_rejects_and_never_builds(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.test")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "LLM_REVIEW_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(config, "LLM_REVIEW_RETRY_BASE_S", 300.0)
    monkeypatch.setattr(
        pipeline,
        "check_llm_review",
        lambda **_: GateResult.reject(
            "llm_review_unavailable", retryable=True, detail="ReadTimeout"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_engine_image_local_mock",
        lambda **_: pytest.fail("must not build while the review is unavailable"),
    )
    repo, patch, sid = _seed(tmp_path)

    def run(row):
        return pipeline.process_submission(
            row,
            registered_hotkeys=[HOTKEY],
            fetcher=lambda _u: patch,
            mock_build=True,
            local_repo=repo,
            work_root=tmp_path / "gate-work",
        )

    row = claim_next_job()
    assert str(row["id"]) == sid and row["job_attempt"] == 1
    assert run(row).reason == "llm_review_deferred"
    status, attempts, in_future, last_error = _job(sid)
    assert (status, attempts, in_future) == ("pending", 1, True)
    assert last_error == "llm_review_unavailable: ReadTimeout"
    assert SubmissionState.REJECTED not in [e["state"] for e in list_events(sid)]
    # Not claimable until retry_after passes.
    assert claim_next_job() is None

    _make_due(sid)
    row = claim_next_job()
    assert str(row["id"]) == sid and row["job_attempt"] == 2
    result = run(row)
    assert result.reason == "llm_review_unavailable"
    assert _job(sid)[0] == "failed"
    assert list_events(sid)[-1]["state"] == SubmissionState.REJECTED
