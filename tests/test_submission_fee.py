"""Campaign submission fee contract tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from campaign.fees import submission_fee_rao, validate_submission_fee
from campaign.manifest import build_manifest
from campaign.models import SLA

FEE = {
    "amount_tao": "0.0005",
    "recipient": "5CiieAa5nzSMbw4LPkh2hqv9rfMPZX9ZfEcSjh3SYWNBzk3K",
}


def _manifest_kwargs(**overrides):
    kwargs = {
        "campaign_id": uuid4(),
        "profile_id": uuid4(),
        "baseline_repo": "https://github.com/vllm-project/vllm.git",
        "baseline_commit": "a" * 40,
        "base_image_digest": "sha256:" + "b" * 64,
        "gpu_skus": ["H200"],
        "workload_trace_sha256": "sha256:" + "c" * 64,
        "workload_trace_url": "https://example.com/t",
        "sla": SLA(),
        "scoring_config_sha256": None,
        "scoring_config_url": None,
        "allowed_paths": ["vllm/**"],
        "denied_paths": ["tests/**"],
        "priority_metric": "throughput",
        "success_threshold": ">=10% at SLA",
        "submission_fee": FEE,
    }
    kwargs.update(overrides)
    return kwargs


def test_submission_fee_is_canonical_and_exact():
    fee = validate_submission_fee(
        {"amount_tao": "0.000500000", "recipient": f"  {FEE['recipient']}  "}
    )
    assert fee == FEE
    assert submission_fee_rao(fee) == 500_000


@pytest.mark.parametrize("amount", ["-1", "nan", "0.0000000001", True, None])
def test_submission_fee_rejects_invalid_amount(amount):
    with pytest.raises(ValueError, match="submission_fee.amount_tao"):
        validate_submission_fee({"amount_tao": amount, "recipient": FEE["recipient"]})


def test_fee_amount_and_recipient_each_change_the_manifest_hash():
    kwargs = _manifest_kwargs()
    base = build_manifest(**kwargs)
    other_amount = build_manifest(
        **{**kwargs, "submission_fee": {**FEE, "amount_tao": "0.001"}}
    )
    other_recipient = build_manifest(
        **{**kwargs, "submission_fee": {**FEE, "recipient": "5Other"}}
    )
    assert (
        len(
            {
                base.manifest_hash,
                other_amount.manifest_hash,
                other_recipient.manifest_hash,
            }
        )
        == 3
    )
    assert base.to_public_dict()["submission_fee"] == FEE
