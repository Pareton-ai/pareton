"""End-to-end Stage 0 with mock fetch + mock build (no Docker/chain)."""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timedelta, timezone
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
from worker.pipeline import process_submission

# Skip if no DB configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("PARETON_DATABASE_URL"),
    reason="PARETON_DATABASE_URL not set",
)


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
    url = f"https://cdn.test/stage0/campaigns/{campaign_id}/patches/hk/e2e.diff"
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
        window_opens_at=now - timedelta(hours=1),
        window_closes_at=now + timedelta(days=1),
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
        window_opens_at=now - timedelta(hours=1),
        window_closes_at=now + timedelta(days=1),
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
    events = [e["state"] for e in list_events(sid)]
    assert "committed" in events
    assert "fetched" in events
    assert "verified" in events
    assert "applied" in events
    assert "surface_ok" in events
    assert "built" in events
