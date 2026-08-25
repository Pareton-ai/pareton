"""The subnet weight vector, built from already-fetched inputs.

Every function here is pure. Nothing reads the database, the chain, or the
network, which is what makes the pay schedule fully unit-testable: the caller
fetches, this decides, and `chain/weights.py` signs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import config

# The vector must sum to 1.0. Burn takes the residual, so anything past float
# noise on top of that is a bug, not a rounding artifact.
SUM_TOLERANCE = 1e-9


class WeightVectorError(RuntimeError):
    """A structural violation. The vector is never returned, never signed."""


@dataclass(frozen=True)
class CampaignEmission:
    """One campaign's inputs to the vector, as already read from Postgres.

    `seed_block` is `rounds.seed_block` of the round the crown was won in,
    reached through `leaders.won_at_round_id`. That column survives a
    successful defence and resets on a leader change, which is exactly the
    decay clock we want. It is the block the round was *created* at, so the
    clock starts about one round early, uniformly for every leader.
    """

    campaign_id: str
    status: str
    emission_rule: dict[str, Any] | None
    leader_hotkey: str | None = None
    seed_block: int | None = None


class WeightVector(NamedTuple):
    """The dense vector (index is uid) and the per-campaign audit trail."""

    weights: list[float]
    breakdown: list[dict[str, Any]]


def emission_curve(rule: Mapping[str, Any]) -> Callable[[int], float]:
    """Return f(blocks_held) -> weight for a campaign's emission rule.

    Dispatches on `rule["name"]` and hands back a function, so no caller ever
    branches on the rule name. That is the whole reason `emission_rule` is one
    JSONB column instead of two scalars: an exponential or step curve is a new
    branch here and nothing else, with no schema change and no caller change.

    The rule arrives already validated and normalized by
    `campaign.models.validate_emission_rule`, so nothing is re-checked here.
    """
    name = rule["name"]
    if name == "linear_decay":
        start = float(rule["start_weight"])
        floor = float(rule["floor_weight"])
        span = int(rule["decay_blocks"])

        def linear_decay(blocks_held: int) -> float:
            if blocks_held <= 0:
                return start
            if blocks_held >= span:
                return floor
            return floor + (start - floor) * (1.0 - blocks_held / span)

        return linear_decay
    raise WeightVectorError(f"unknown emission_rule.name: {name!r}")


def _entry(
    campaign: CampaignEmission,
    *,
    hotkey: str | None,
    uid: int | None,
    blocks_held: int | None,
    weight: float,
    note: str | None,
) -> dict[str, Any]:
    """One `weight_sets.breakdown` row. Shape is fixed by `db/schema.sql`."""
    return {
        "campaign_id": campaign.campaign_id,
        "hotkey": hotkey,
        "uid": uid,
        "blocks_held": blocks_held,
        "weight": weight,
        "note": note,
    }


def _withheld(
    campaign: CampaignEmission, *, hotkey: str | None, note: str
) -> dict[str, Any]:
    """A campaign that pays nobody this cycle, with the reason recorded."""
    return _entry(
        campaign, hotkey=hotkey, uid=None, blocks_held=None, weight=0.0, note=note
    )


def build_weight_vector(
    campaigns: Iterable[CampaignEmission],
    *,
    current_block: int,
    hotkey_uids: Mapping[str, int],
    uid_count: int,
    burn_uid: int = config.BURN_UID,
) -> WeightVector:
    """The dense weight vector plus its per-campaign breakdown.

    `hotkey_uids` and `uid_count` both come from the live metagraph on every
    compute. No UID is ever persisted: a UID is a lease, not an identity, and
    deregistration reassigns it, so a stored one eventually pays a stranger.
    `uid_count` is `len(metagraph.hotkeys)`, never a hardcoded 256.

    Shares from multiple campaigns add with no per-UID cap: a miner who wins
    two campaigns earned both. What stops the promises over-committing the
    subnet is the `sum(start_weight) <= 1.0` trigger on `campaigns`.

    Raises `WeightVectorError` on a structural violation rather than returning
    a vector, so a bad vector is never signed. Structural only: an
    economically lopsided but valid vector is signed, because a second place
    deciding what is payable can disagree with the formula.
    """
    if not 0 <= burn_uid < uid_count:
        raise WeightVectorError(
            f"burn_uid {burn_uid} is outside the metagraph (uid_count={uid_count})"
        )

    weights = [0.0] * uid_count
    breakdown: list[dict[str, Any]] = []

    for campaign in campaigns:
        if campaign.emission_rule is None:
            # NULL emission_rule means the campaign pays nothing and is left
            # out of the vector entirely, hash and breakdown included.
            continue
        # A withheld share carries its reason and no uid: the crown paid
        # nobody, so there is no lease to record.
        if campaign.leader_hotkey is None:
            breakdown.append(_withheld(campaign, hotkey=None, note="vacant"))
            continue
        if campaign.status != "open":
            breakdown.append(
                _withheld(campaign, hotkey=campaign.leader_hotkey, note="closed")
            )
            continue
        uid = hotkey_uids.get(campaign.leader_hotkey)
        if uid is None:
            breakdown.append(
                _withheld(campaign, hotkey=campaign.leader_hotkey, note="deregistered")
            )
            continue
        if not 0 <= uid < uid_count:
            raise WeightVectorError(
                f"uid {uid} is outside the metagraph (uid_count={uid_count})"
            )

        blocks_held = current_block - campaign.seed_block
        weight = emission_curve(campaign.emission_rule)(blocks_held)
        weights[uid] += weight
        breakdown.append(
            _entry(
                campaign,
                hotkey=campaign.leader_hotkey,
                uid=uid,
                blocks_held=blocks_held,
                weight=weight,
                note=None,
            )
        )

    # Burn absorbs the rounding residual by construction, so the vector sums
    # to exactly 1.0 and no miner is over-paid by a float artifact.
    burn = 1.0 - math.fsum(weights)
    if burn < 0.0:
        raise WeightVectorError(
            f"burn is negative ({burn!r}): the sum cap was violated upstream"
        )
    weights[burn_uid] += burn

    for uid, weight in enumerate(weights):
        if not 0.0 <= weight <= 1.0:
            raise WeightVectorError(f"weight {weight!r} at uid {uid} is outside [0, 1]")
    total = math.fsum(weights)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=SUM_TOLERANCE):
        raise WeightVectorError(f"weights sum to {total!r}, not 1.0")

    return WeightVector(weights, breakdown)
