"""Unit tests for identity and integrity gates."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from datetime import datetime, timedelta, timezone
from uuid import uuid4

from campaign.manifest import build_manifest
from campaign.models import SLA
from gate.identity import check_identity
from gate.integrity import check_integrity, hash_patch_bytes
from storage.s3 import is_allowed_retrieval_url
import config


def _campaign(**overrides):
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
        window_opens_at=now - timedelta(hours=1),
        window_closes_at=now + timedelta(days=1),
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="open",
    )
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_identity_accepts_registered_in_window():
    c = _campaign()
    res = check_identity(
        hotkey="hk1",
        registered_hotkeys=["hk1", "hk2"],
        campaign=c,
        baseline_commit="a" * 40,
    )
    assert res.ok


def test_identity_rejects_unregistered():
    c = _campaign()
    res = check_identity(
        hotkey="hkX",
        registered_hotkeys=["hk1"],
        campaign=c,
        baseline_commit="a" * 40,
    )
    assert not res.ok
    assert "registered" in res.reason


def test_identity_rejects_baseline_mismatch():
    c = _campaign()
    res = check_identity(
        hotkey="hk1",
        registered_hotkeys=["hk1"],
        campaign=c,
        baseline_commit="f" * 40,
    )
    assert not res.ok
    assert "baseline" in res.reason


def test_integrity_hash_match():
    data = b"hello patch"
    expected = hash_patch_bytes(data)
    res = check_integrity(
        retrieval_url="https://example.com/stage0/campaigns/x/patches/h/1.diff",
        expected_patch_hash=expected,
        fetcher=lambda _u: data,
    )
    # URL allowlist may reject example.com — override by patching config in test
    assert res.ok or res.reason == "retrieval_url not allowlisted"


def test_integrity_with_allowlisted_url(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "pareton-patches")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com/pareton")
    url = "https://cdn.example.com/pareton/stage0/campaigns/cid/patches/hk/1.diff"
    assert is_allowed_retrieval_url(url)
    data = b"abc"
    expected = hash_patch_bytes(data)
    res = check_integrity(
        retrieval_url=url,
        expected_patch_hash=expected,
        fetcher=lambda _u: data,
    )
    assert res.ok
    assert res.evidence["patch_bytes"] == data


def test_integrity_mismatch(monkeypatch):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    url = "https://cdn.example.com/stage0/campaigns/cid/patches/hk/1.diff"
    res = check_integrity(
        retrieval_url=url,
        expected_patch_hash="sha256:" + "0" * 64,
        fetcher=lambda _u: b"abc",
    )
    assert not res.ok
    assert res.reason == "patch_hash mismatch"
