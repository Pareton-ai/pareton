"""Unit tests for campaign manifest hashing."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from datetime import datetime, timezone
from uuid import uuid4

from campaign.manifest import (
    build_manifest,
    compute_manifest_hash,
    freeze_manifest_fields,
)
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
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
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
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
    )
    assert m.baseline_commit == "a" * 40


def _manifest_kwargs(**overrides):
    now = datetime.now(timezone.utc)
    kwargs = dict(
        campaign_id=uuid4(),
        profile_id=uuid4(),
        baseline_repo="https://github.com/vllm-project/vllm.git",
        baseline_commit="a" * 40,
        base_image_digest="sha256:" + "b" * 64,
        gpu_skus=["H200"],
        workload_trace_sha256="sha256:" + "c" * 64,
        workload_trace_url="https://example.com/t",
        sla=SLA(),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        window_opens_at=now,
        window_closes_at=now,
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
    )
    kwargs.update(overrides)
    return kwargs


def test_priority_metric_rejects_unknown_value():
    with pytest.raises(ValueError, match="priority_metric must be one of"):
        build_manifest(**_manifest_kwargs(priority_metric="tokens_per_watt"))


def test_priority_metric_normalizes_case():
    m = build_manifest(**_manifest_kwargs(priority_metric="GPU_Hours"))
    assert m.priority_metric == "gpu_hours"


def test_priority_metric_changes_manifest_hash():
    a = build_manifest(**_manifest_kwargs(priority_metric="throughput"))
    b = build_manifest(
        **_manifest_kwargs(
            campaign_id=a.campaign_id,
            profile_id=a.profile_id,
            priority_metric="gpu_hours",
        )
    )
    assert a.manifest_hash != b.manifest_hash


def test_success_threshold_changes_manifest_hash():
    a = build_manifest(**_manifest_kwargs(success_threshold=">=10% at SLA"))
    b = build_manifest(
        **_manifest_kwargs(
            campaign_id=a.campaign_id,
            profile_id=a.profile_id,
            success_threshold=">=20% at SLA",
        )
    )
    assert a.manifest_hash != b.manifest_hash


def test_workload_pool_absent_keeps_hash_stable():
    kwargs = _manifest_kwargs()
    base = freeze_manifest_fields(**kwargs)
    with_none = freeze_manifest_fields(**kwargs, workload_pool=None)
    assert compute_manifest_hash(base) == compute_manifest_hash(with_none)
    assert "workload_pool" not in base


def test_workload_pool_and_z_threshold_change_hash():
    kwargs = _manifest_kwargs()
    a = build_manifest(**kwargs)
    b = build_manifest(
        **kwargs,
        workload_pool=[
            {"sha256": "sha256:" + ("1" * 64), "url": "https://x/1.json"},
            {"sha256": "sha256:" + ("2" * 64), "url": "https://x/2.json"},
        ],
        sampling_rule={"type": "uniform_index", "seed_block_offset": 1},
        z_threshold=3.0,
    )
    assert a.manifest_hash != b.manifest_hash
    assert b.workload_pool is not None
    assert b.z_threshold == 3.0
