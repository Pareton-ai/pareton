"""Unit tests for Pareton patch commitment parsing."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from chain.commitment import (
    build_patch_commitments,
    encode_patch_commitment,
    parse_patch_commitment,
)


def test_parse_and_encode_roundtrip():
    campaign_id = "123e4567-e89b-12d3-a456-426614174000"
    raw = encode_patch_commitment(
        campaign_id=campaign_id,
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "b" * 64,
        retrieval_url="https://s3.example.com/bucket/stage0/campaigns/x/patches/hk/1.diff",
    )
    parsed = parse_patch_commitment(raw)
    assert parsed is not None
    assert parsed["campaign_id"] == campaign_id
    assert parsed["baseline_commit"] == "a" * 40
    assert parsed["patch_hash"] == "sha256:" + "b" * 64


def test_reject_image_payload():
    raw = '{"image":"docker.io/x/y:v1","digest":"sha256:' + ("c" * 64) + '"}'
    assert parse_patch_commitment(raw) is None


def test_reject_bad_url_scheme():
    raw = encode_patch_commitment(
        campaign_id="123e4567-e89b-12d3-a456-426614174000",
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "b" * 64,
        retrieval_url="ipfs://QmSomething",
    )
    # encode allows any string; parse rejects non-http(s)
    assert parse_patch_commitment(raw) is None


def test_build_patch_commitments_latest_wins():
    class Meta:
        hotkeys = ["hk1"]
        coldkeys = ["ck1"]

    campaign_id = "123e4567-e89b-12d3-a456-426614174000"
    older = encode_patch_commitment(
        campaign_id=campaign_id,
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "1" * 64,
        retrieval_url="https://example.com/a.diff",
    )
    newer = encode_patch_commitment(
        campaign_id=campaign_id,
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "2" * 64,
        retrieval_url="https://example.com/b.diff",
    )
    out = build_patch_commitments(Meta(), {"hk1": [(10, older), (20, newer)]})
    assert 0 in out
    assert out[0].patch_hash == "sha256:" + "2" * 64
    assert out[0].commit_block == 20
