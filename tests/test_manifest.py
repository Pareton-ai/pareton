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
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
    )
    assert m.baseline_commit == "a" * 40


def _manifest_kwargs(**overrides):
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
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
    )
    kwargs.update(overrides)
    return kwargs


def test_window_is_not_in_the_pin_set():
    """Campaigns have no submission window, so nothing about one is hashed."""
    fields = freeze_manifest_fields(**_manifest_kwargs())
    assert "window" not in fields


def test_public_dict_reports_created_at_and_no_window():
    m = build_manifest(
        **_manifest_kwargs(created_at=datetime(2026, 8, 4, tzinfo=timezone.utc))
    )
    out = m.to_public_dict()
    assert "window" not in out
    assert out["created_at"] == "2026-08-04T00:00:00+00:00"
    assert out["workload_trace_url"] == "https://example.com/t"
    assert "sampling_rule" not in out


def test_public_dict_omits_trace_when_sampling_rule_is_set():
    rule = {
        "type": "hf_rows",
        "dataset": "nebius/SWE-agent-trajectories",
        "revision": "a" * 40,
        "n_rows": 1000,
        "n_prompts": 32,
    }
    m = build_manifest(
        **_manifest_kwargs(
            workload_trace_url="file:///Users/xavierlu/Desktop/trace.json",
            sampling_rule=rule,
        )
    )
    out = m.to_public_dict()
    assert "workload_trace_url" not in out
    assert "workload_trace_sha256" not in out
    assert out["sampling_rule"]["type"] == "hf_rows"
    assert out["sampling_rule"]["dataset"] == rule["dataset"]


def test_created_at_is_not_hashed():
    """created_at is read from the DB row, never pinned."""
    kwargs = _manifest_kwargs()
    bare = build_manifest(**kwargs)
    dated = build_manifest(**kwargs, created_at=datetime.now(timezone.utc))
    assert bare.manifest_hash == dated.manifest_hash
    assert bare.to_public_dict()["created_at"] is None


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


def test_workload_pool_and_sampling_rule_change_hash():
    kwargs = _manifest_kwargs()
    a = build_manifest(**kwargs)
    b = build_manifest(
        **kwargs,
        workload_pool=[
            {"sha256": "sha256:" + ("1" * 64), "url": "https://x/1.json"},
            {"sha256": "sha256:" + ("2" * 64), "url": "https://x/2.json"},
        ],
        sampling_rule={"type": "uniform_index", "seed_block_offset": 1},
    )
    assert a.manifest_hash != b.manifest_hash
    assert b.workload_pool is not None


def test_scoring_rule_is_always_pinned_and_defaults():
    """The scoring formula is part of what a miner competes on, so two
    campaigns with the same hash must rank identically."""
    kwargs = _manifest_kwargs()
    default = build_manifest(**kwargs)
    assert default.scoring_rule == {"name": "median_e2e_speedup"}
    fields = freeze_manifest_fields(**_manifest_kwargs())
    assert fields["scoring_rule"] == {"name": "median_e2e_speedup"}


def test_scoring_rule_extras_change_hash():
    kwargs = _manifest_kwargs()
    a = build_manifest(**kwargs)
    b = build_manifest(
        **kwargs, scoring_rule={"name": "median_e2e_speedup", "min_prompts": 8}
    )
    assert a.manifest_hash != b.manifest_hash
    assert b.scoring_rule["min_prompts"] == 8


def test_unknown_scoring_rule_is_refused():
    with pytest.raises(ValueError, match="scoring_rule.name must be one of"):
        build_manifest(**_manifest_kwargs(), scoring_rule={"name": "vibes"})
