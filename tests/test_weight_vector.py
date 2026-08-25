"""Unit tests for the pure weight-vector builder (PAR-103).

Nothing here touches a chain, a database, or a network: the builder takes
already-fetched inputs, which is the point of it being pure.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit

from weights.build import (
    CampaignEmission,
    WeightVectorError,
    build_weight_vector,
    emission_curve,
)

RULE = {
    "name": "linear_decay",
    "start_weight": 0.10,
    "floor_weight": 0.02,
    "decay_blocks": 201600,
}
BURN_UID = 63


def rule(start: float, floor: float = 0.0, span: int = 201600) -> dict:
    return {
        "name": "linear_decay",
        "start_weight": start,
        "floor_weight": floor,
        "decay_blocks": span,
    }


def seated(
    campaign_id: str,
    hotkey: str,
    *,
    emission_rule: dict | None = RULE,
    seed_block: int = 1_000,
    status: str = "open",
) -> CampaignEmission:
    return CampaignEmission(
        campaign_id=campaign_id,
        status=status,
        emission_rule=emission_rule,
        leader_hotkey=hotkey,
        seed_block=seed_block,
    )


def build(
    campaigns, *, current_block=1_000, uids=None, uid_count=64, burn_uid=BURN_UID
):
    return build_weight_vector(
        campaigns,
        current_block=current_block,
        hotkey_uids={} if uids is None else uids,
        uid_count=uid_count,
        burn_uid=burn_uid,
    )


# --- the curve ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("blocks_held", "expected"),
    [
        (-5_000, 0.10),  # a leader seated after the head we read
        (0, 0.10),  # the block the crown was won
        (50_400, 0.08),  # a quarter of the way down
        (100_800, 0.06),  # mid-decay
        (201_599, pytest.approx(0.02, abs=1e-6)),  # one block short of the floor
        (201_600, 0.02),  # exactly at decay_blocks
        (10_000_000, 0.02),  # far past it: the floor, never below
    ],
)
def test_linear_decay_curve(blocks_held: int, expected: float) -> None:
    assert emission_curve(RULE)(blocks_held) == pytest.approx(expected)


def test_curve_is_a_function_of_blocks_not_a_value() -> None:
    """Dispatch happens once; the caller never branches on the rule name."""
    curve = emission_curve(RULE)
    assert callable(curve)
    assert curve(0) > curve(100_800) > curve(201_600)


def test_unknown_rule_name_raises() -> None:
    with pytest.raises(WeightVectorError, match="unknown emission_rule.name"):
        emission_curve({"name": "exponential_decay"})


# --- the vector --------------------------------------------------------------


@pytest.mark.parametrize("uid_count", [64, 512])
def test_vector_length_follows_the_metagraph(uid_count: int) -> None:
    """No hardcoded 256 anywhere: the length is whatever the metagraph says."""
    burn_uid = uid_count - 1
    vector = build(
        [seated("c1", "5Alice")],
        uids={"5Alice": uid_count - 2},
        uid_count=uid_count,
        burn_uid=burn_uid,
    )
    assert len(vector.weights) == uid_count
    assert vector.weights[uid_count - 2] == pytest.approx(0.10)
    assert vector.weights[burn_uid] == pytest.approx(0.90)


def test_breakdown_row_matches_the_schema_columns() -> None:
    vector = build(
        [seated("c1", "5Alice", seed_block=1_000)],
        current_block=101_800,
        uids={"5Alice": 3},
    )
    assert vector.breakdown == [
        {
            "campaign_id": "c1",
            "hotkey": "5Alice",
            "uid": 3,
            "blocks_held": 100_800,
            "weight": pytest.approx(0.06),
            "note": None,
        }
    ]


def test_two_campaigns_one_hotkey_add_at_a_single_uid() -> None:
    """A miner who wins two campaigns earned both. No per-UID cap."""
    vector = build(
        [
            seated("c1", "5Alice", emission_rule=rule(0.10)),
            seated("c2", "5Alice", emission_rule=rule(0.25)),
        ],
        uids={"5Alice": 7},
    )
    assert vector.weights[7] == pytest.approx(0.35)
    assert vector.weights[BURN_UID] == pytest.approx(0.65)
    assert [e["uid"] for e in vector.breakdown] == [7, 7]


def test_vacant_crown_burns_the_whole_share() -> None:
    campaign = CampaignEmission(
        campaign_id="c1", status="open", emission_rule=RULE, leader_hotkey=None
    )
    vector = build([campaign])
    assert vector.weights[BURN_UID] == 1.0
    assert vector.breakdown[0]["note"] == "vacant"
    assert vector.breakdown[0]["weight"] == 0.0
    assert vector.breakdown[0]["uid"] is None


def test_closed_campaign_stops_paying_immediately() -> None:
    """Closed means the competition is over; the budget frees at once."""
    vector = build(
        [seated("c1", "5Alice", status="closed")],
        uids={"5Alice": 7},
    )
    assert vector.weights[7] == 0.0
    assert vector.weights[BURN_UID] == 1.0
    assert vector.breakdown[0]["note"] == "closed"
    assert vector.breakdown[0]["hotkey"] == "5Alice"


def test_leader_absent_from_the_metagraph_contributes_nothing() -> None:
    vector = build([seated("c1", "5Gone")], uids={"5Alice": 7})
    assert vector.weights == [0.0] * 63 + [1.0]
    assert vector.breakdown[0]["note"] == "deregistered"
    assert vector.breakdown[0]["uid"] is None


def test_no_campaigns_at_all_is_one_hundred_percent_burn() -> None:
    vector = build([])
    assert vector.weights[BURN_UID] == 1.0
    assert math.fsum(vector.weights) == 1.0
    assert vector.breakdown == []


def test_campaign_without_an_emission_rule_is_left_out_entirely() -> None:
    """NULL emission_rule pays nothing and records nothing."""
    vector = build([seated("c1", "5Alice", emission_rule=None)], uids={"5Alice": 7})
    assert vector.weights[7] == 0.0
    assert vector.breakdown == []


def test_a_leader_holding_the_burn_uid_still_sums_to_one() -> None:
    vector = build([seated("c1", "5Alice")], uids={"5Alice": BURN_UID})
    assert vector.weights[BURN_UID] == pytest.approx(1.0)
    assert math.fsum(vector.weights) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "shares",
    [
        [0.1, 0.2],
        [0.07, 0.13, 0.29, 0.31],
        [1e-9, 0.3333333333333333, 0.0000001],
        [0.1] * 9,
        [0.9999999999],
        [0.0],
        [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
    ],
)
def test_sum_is_one_across_awkward_float_inputs(shares: list[float]) -> None:
    """Burn takes the residual, so float noise never over-pays a miner."""
    campaigns = [
        seated(f"c{i}", f"5Miner{i}", emission_rule=rule(share, floor=share))
        for i, share in enumerate(shares)
    ]
    uids = {f"5Miner{i}": i for i in range(len(shares))}
    vector = build(campaigns, current_block=10_000_000, uids=uids)
    assert math.fsum(vector.weights) == pytest.approx(1.0, abs=1e-12)
    assert vector.weights[BURN_UID] == pytest.approx(1.0 - math.fsum(shares), abs=1e-12)


# --- structural guards -------------------------------------------------------


@pytest.mark.parametrize("burn_uid", [64, 512, -1])
def test_burn_uid_outside_the_metagraph_raises(burn_uid: int) -> None:
    with pytest.raises(WeightVectorError, match="outside the metagraph"):
        build([], uid_count=64, burn_uid=burn_uid)


def test_uid_outside_the_metagraph_raises() -> None:
    with pytest.raises(WeightVectorError, match="outside the metagraph"):
        build([seated("c1", "5Alice")], uids={"5Alice": 99}, uid_count=64)


def test_over_committed_campaigns_raise_rather_than_returning_a_vector() -> None:
    """A negative burn means the sum trigger was bypassed upstream."""
    campaigns = [
        seated(f"c{i}", f"5Miner{i}", emission_rule=rule(0.4)) for i in range(3)
    ]
    uids = {f"5Miner{i}": i for i in range(3)}
    with pytest.raises(WeightVectorError, match="burn is negative"):
        build(campaigns, uids=uids)


def test_a_weight_outside_zero_to_one_raises() -> None:
    """A negative floor leaves burn positive, so only the range guard catches it."""
    campaigns = [seated("c1", "5Alice", emission_rule=rule(0.1, floor=-0.2))]
    with pytest.raises(WeightVectorError, match=r"outside \[0, 1\]"):
        build(campaigns, current_block=10_000_000, uids={"5Alice": 7})
