"""Unit tests for the campaign emission rule and its manifest pin (PAR-102)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from uuid import uuid4

from campaign.manifest import (
    build_manifest,
    compute_manifest_hash,
    freeze_manifest_fields,
)
from campaign.models import SLA, validate_emission_rule

RULE = {
    "name": "linear_decay",
    "start_weight": 0.10,
    "floor_weight": 0.02,
    "decay_blocks": 201600,
}


def _fields_kwargs(**overrides):
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
        submission_fee={"amount_tao": "0", "recipient": "5Test"},
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------
# The pin: absent means unpinned, present means signed
# --------------------------------------------------------------------------


def test_absent_emission_rule_leaves_manifest_hash_unchanged():
    """A campaign pinned before the column existed must hash to the same value.

    ``freeze_manifest_fields`` is called with no ``emission_rule`` kwarg at
    all, exactly as every caller did before PAR-102. The golden pre-emission
    hash itself is asserted in ``test_engine_profile.py``.
    """
    kwargs = _fields_kwargs()
    fields = freeze_manifest_fields(**kwargs)
    assert "emission_rule" not in fields
    assert compute_manifest_hash(fields) == compute_manifest_hash(
        freeze_manifest_fields(**kwargs, emission_rule=None)
    )


def test_a_pinned_emission_rule_changes_manifest_hash():
    """The pay schedule is signed, so it cannot move under a miner silently."""
    kwargs = _fields_kwargs()
    unpaid = compute_manifest_hash(freeze_manifest_fields(**kwargs))
    paid = compute_manifest_hash(freeze_manifest_fields(**kwargs, emission_rule=RULE))
    assert unpaid != paid


def test_every_term_of_the_rule_is_part_of_the_hash():
    kwargs = _fields_kwargs()
    base = compute_manifest_hash(freeze_manifest_fields(**kwargs, emission_rule=RULE))
    for key, value in [
        ("start_weight", 0.2),
        ("floor_weight", 0.01),
        ("decay_blocks", 100800),
    ]:
        moved = compute_manifest_hash(
            freeze_manifest_fields(**kwargs, emission_rule={**RULE, key: value})
        )
        assert moved != base, key


def test_the_hash_is_key_order_independent():
    kwargs = _fields_kwargs()
    a = compute_manifest_hash(freeze_manifest_fields(**kwargs, emission_rule=RULE))
    b = compute_manifest_hash(
        freeze_manifest_fields(
            **kwargs,
            emission_rule={
                "decay_blocks": 201600,
                "floor_weight": 0.02,
                "start_weight": 0.10,
                "name": " linear_decay ",
            },
        )
    )
    assert a == b


def test_build_manifest_carries_the_rule_onto_the_campaign():
    m = build_manifest(**_fields_kwargs(), emission_rule=RULE)
    assert m.emission_rule == RULE
    assert m.to_public_dict()["emission_rule"] == RULE


def test_build_manifest_without_a_rule_pays_nothing():
    m = build_manifest(**_fields_kwargs())
    assert m.emission_rule is None
    assert m.to_public_dict()["emission_rule"] is None


def test_build_manifest_rejects_a_bad_rule():
    with pytest.raises(ValueError, match="emission_rule.name must be one of"):
        build_manifest(**_fields_kwargs(), emission_rule={**RULE, "name": "annuity"})


# --------------------------------------------------------------------------
# validate_emission_rule
# --------------------------------------------------------------------------


def test_validation_returns_exactly_the_known_keys():
    assert validate_emission_rule({**RULE, "start_weight": 1}) == {
        **RULE,
        "start_weight": 1.0,
    }


@pytest.mark.parametrize(
    "rule, match",
    [
        ("linear_decay", "emission_rule must be an object"),
        ({**RULE, "name": "exponential_decay"}, "emission_rule.name must be one of"),
        ({**RULE, "reset_min_rounds": 3}, "emission_rule has unknown keys"),
        (
            {**RULE, "floor_weight": 0.5},
            "emission_rule.floor_weight must be <= start_weight",
        ),
        (
            {**RULE, "start_weight": 1.5},
            r"emission_rule.start_weight must be in \[0, 1\]",
        ),
        (
            {**RULE, "start_weight": -0.1},
            r"emission_rule.start_weight must be in \[0, 1\]",
        ),
        (
            {**RULE, "floor_weight": float("nan")},
            r"emission_rule.floor_weight must be in \[0, 1\]",
        ),
        (
            {**RULE, "start_weight": "0.1"},
            "emission_rule.start_weight must be a number",
        ),
        ({**RULE, "start_weight": True}, "emission_rule.start_weight must be a number"),
        ({**RULE, "decay_blocks": 0}, "emission_rule.decay_blocks must be >= 1"),
        ({**RULE, "decay_blocks": -1}, "emission_rule.decay_blocks must be >= 1"),
        (
            {**RULE, "decay_blocks": 201600.0},
            "emission_rule.decay_blocks must be an integer",
        ),
    ],
)
def test_validation_rejects_malformed(rule, match):
    with pytest.raises(ValueError, match=match):
        validate_emission_rule(rule)


def test_a_flat_rule_is_allowed():
    """floor == start is a campaign that never decays: legal, and explicit."""
    flat = validate_emission_rule({**RULE, "floor_weight": 0.10})
    assert flat["floor_weight"] == flat["start_weight"]
