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


CAMPAIGN_ID = "123e4567-e89b-12d3-a456-426614174000"


def _encode(**overrides) -> str:
    kwargs = dict(
        campaign_id=CAMPAIGN_ID,
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "b" * 64,
        retrieval_url="https://example.com/a.diff",
    )
    kwargs.update(overrides)
    return encode_patch_commitment(**kwargs)


def test_payment_proof_roundtrip():
    raw = _encode(payment_block=6_123_456, payment_tx=3)
    assert raw.endswith("|6123456|3")
    parsed = parse_patch_commitment(raw)
    assert parsed is not None
    assert parsed["payment_block"] == 6_123_456
    assert parsed["payment_tx"] == 3


def test_payload_without_proof_carries_no_payment_fields():
    parsed = parse_patch_commitment(_encode())
    assert parsed is not None
    assert "payment_block" not in parsed
    assert "payment_tx" not in parsed


def test_proof_costs_far_less_than_a_tx_hash():
    # The ref is (block, index) precisely because a 0x-prefixed 32-byte hash
    # would cost ~67 bytes against the 384-byte cap.
    overhead = len(_encode(payment_block=6_123_456, payment_tx=3)) - len(_encode())
    assert overhead <= 16


@pytest.mark.parametrize(
    "tail",
    ["|abc|1", "|1|abc", "|0|1", "|-5|1", "|1.5|0", "||1", "|1|", "|1e3|0"],
)
def test_reject_malformed_payment_proof(tail):
    assert parse_patch_commitment(_encode() + tail) is None


def test_reject_v2_with_six_fields():
    assert parse_patch_commitment(_encode() + "|900") is None


def test_encode_rejects_half_set_proof():
    with pytest.raises(ValueError, match="together"):
        _encode(payment_block=900)
    with pytest.raises(ValueError, match="together"):
        _encode(payment_tx=0)


def test_encode_rejects_payload_over_the_field_cap():
    with pytest.raises(ValueError, match="384 bytes"):
        _encode(retrieval_url="https://example.com/" + "d" * 300)


def test_build_patch_commitments_keeps_payment_proof():
    class Meta:
        hotkeys = ["hk1"]
        coldkeys = ["ck1"]

    raw = _encode(payment_block=901, payment_tx=4)
    out = build_patch_commitments(Meta(), {"hk1": [(20, raw)]})
    assert out[0].payment_block == 901
    assert out[0].payment_tx == 4


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
