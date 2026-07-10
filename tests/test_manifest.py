"""Unit tests for campaign manifest hashing."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from datetime import datetime, timezone
from uuid import uuid4

from campaign.manifest import build_manifest, compute_manifest_hash, freeze_manifest_fields
from campaign.models import SLA


def test_manifest_hash_stable():
    cid = uuid4()
    pid = uuid4()
    opens = datetime(2026, 7, 1, tzinfo=timezone.utc)
    closes = datetime(2026, 10, 1, tzinfo=timezone.utc)
    fields = freeze_manifest_fields(
        campaign_id=cid,
        profile_id=pid,
        baseline_repo="https://github.com/vllm-project/vllm.git",
        baseline_commit="A" * 40,
        base_image_digest="sha256:" + "B" * 64,
        gpu_skus=["H200"],
        workload_trace_sha256="sha256:" + "c" * 64,
        workload_trace_url="https://example.com/trace.json",
        sla=SLA(p99_ttft_ms=1.0),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        window_opens_at=opens,
        window_closes_at=closes,
    )
    h1 = compute_manifest_hash(fields)
    h2 = compute_manifest_hash(fields)
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64


def test_build_manifest_normalizes_commit_case():
    m = build_manifest(
        campaign_id=uuid4(),
        profile_id=uuid4(),
        baseline_repo="https://github.com/vllm-project/vllm.git",
        baseline_commit="A" * 40,
        base_image_digest="sha256:" + "B" * 64,
        gpu_skus=[],
        workload_trace_sha256="sha256:" + "c" * 64,
        workload_trace_url="https://example.com/t",
        sla={},
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=[],
        window_opens_at=datetime.now(timezone.utc),
        window_closes_at=datetime.now(timezone.utc),
    )
    assert m.baseline_commit == "a" * 40
