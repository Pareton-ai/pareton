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


def test_parse_legacy_v1_json():
    raw = (
        '{"v":1,"campaign_id":"123e4567-e89b-12d3-a456-426614174000",'
        '"baseline_commit":"' + "a" * 40 + '",'
        '"patch_hash":"sha256:' + "b" * 64 + '",'
        '"retrieval_url":"https://example.com/a.diff"}'
    )
    parsed = parse_patch_commitment(raw)
    assert parsed is not None
    assert parsed["patch_hash"] == "sha256:" + "b" * 64


def test_reject_v2_wrong_field_count():
    assert parse_patch_commitment("v2|only|three|parts") is None


def test_encode_fits_finney_maxfields():
    # CommitmentInfo.fields is a BoundedVec with MaxFields=3 on finney
    # (3 x 128-byte Raw chunks); exceeding it traps validate_transaction.
    raw = encode_patch_commitment(
        campaign_id="123e4567-e89b-12d3-a456-426614174000",
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "b" * 64,
        retrieval_url="https://pareton-s3.s3.us-east-2.amazonaws.com/stage0/campaigns/"
        + "c" * 36
        + "/patches/"
        + "h" * 48
        + "/"
        + "d" * 36
        + ".diff",
    )
    assert len(raw.encode()) <= 3 * 128


def test_encode_rejects_bad_url_scheme():
    with pytest.raises(ValueError, match="http"):
        encode_patch_commitment(
            campaign_id="123e4567-e89b-12d3-a456-426614174000",
            baseline_commit="a" * 40,
            patch_hash="sha256:" + "b" * 64,
            retrieval_url="ipfs://QmSomething",
        )


def test_encode_rejects_pipe_in_url():
    with pytest.raises(ValueError, match="\\|"):
        encode_patch_commitment(
            campaign_id="123e4567-e89b-12d3-a456-426614174000",
            baseline_commit="a" * 40,
            patch_hash="sha256:" + "b" * 64,
            retrieval_url="https://example.com/a|b.diff",
        )


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
