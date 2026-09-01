"""The pareton-weights cadence: compute, store, and sign one vector.

The builder is pure; this module is the I/O. The ``weight_sets`` row is
inserted before the chain call returns, so `/v1/weights` can serve a vector
that is still in flight, or that never landed if this process dies in between.
"""

from __future__ import annotations

import logging
from typing import Any

import config
from chain.weights import (
    WeightSetError,
    assert_validator_permit,
    set_weights,
)
from observability import events as obs
from round.store import (
    get_latest_weight_set,
    insert_weight_set,
    list_idle_seated_leaders,
    list_open_campaign_emissions,
    mark_weight_set_result,
    vacate_leader_if_idle,
)
from weights.build import CampaignEmission, WeightVectorError, build_weight_vector

logger = logging.getLogger(__name__)


def cycle_due(last_block: int | None, head: int, cadence: int) -> bool:
    """True when no cycle has run yet, or ``cadence`` blocks have passed."""
    if last_block is None:
        return True
    return head >= last_block + cadence


def new_round_due(last_round: str | None, latest_round: str | None) -> bool:
    """True when a scored round exists that this process has not stored."""
    return latest_round is not None and latest_round != last_round


def _hotkey_uids(meta: Any) -> tuple[dict[str, int], int]:
    """Live ``hotkey -> uid`` map from ``by_hotkey().uid``. UIDs are never stored.

    ``enumerate(meta.hotkeys)`` is not an authority. If a listed hotkey is
    missing from ``by_hotkey`` or the uid is outside the metagraph length,
    raise rather than pay the wrong miner.
    """
    hotkeys = [str(hk) for hk in getattr(meta, "hotkeys", [])]
    uid_count = len(hotkeys)
    mapping: dict[str, int] = {}
    for hotkey in hotkeys:
        neuron = meta.by_hotkey(hotkey)
        if neuron is None:
            raise WeightVectorError(
                "metagraph lists a hotkey that by_hotkey does not resolve"
            )
        uid = int(neuron.uid)
        if not 0 <= uid < uid_count:
            raise WeightVectorError(
                f"uid {uid} is outside the metagraph (uid_count={uid_count})"
            )
        mapping[hotkey] = uid
    return mapping, uid_count


def _campaigns(rows: list[dict[str, Any]]) -> list[CampaignEmission]:
    return [
        CampaignEmission(
            campaign_id=str(row["campaign_id"]),
            status=str(row["status"]),
            emission_rule=row["emission_rule"],
            leader_hotkey=row["leader_hotkey"],
            seed_block=row["seed_block"],
        )
        for row in rows
    ]


def reconcile_deregistrations(hotkey_uids: dict[str, int]) -> list[str]:
    """Vacate seated leaders whose hotkey has left the metagraph.

    A campaign with a pending or running round is not in the idle list and is
    retried next cycle. If a round appears between the list and the delete,
    ``vacate_leader_if_idle`` returns False and we log it.
    """
    vacated: list[str] = []
    for row in list_idle_seated_leaders():
        hotkey = row["hotkey"]
        campaign_id = str(row["campaign_id"])
        if hotkey in hotkey_uids:
            continue
        if vacate_leader_if_idle(campaign_id, epsilon=config.OVERTAKE_EPSILON):
            logger.info("vacated deregistered leader on campaign %s", campaign_id)
            vacated.append(campaign_id)
        else:
            logger.info(
                "skipped vacate on campaign %s: live round or crown already gone",
                campaign_id,
            )
    return vacated


def _same_vector(row: dict[str, Any] | None, values: list[float]) -> bool:
    if row is None:
        return False
    return (
        int(row["version_key"]) == config.VERSION_KEY
        and int(row["burn_uid"]) == config.BURN_UID
        and [float(value) for value in row["weights"]] == values
    )


def run_cycle(
    subtensor: Any,
    wallet: Any,
    meta: Any,
    *,
    current_block: int,
    enabled: bool,
    sign: bool = True,
) -> int | None:
    """Compute once, store only when changed, and optionally sign."""
    if sign and enabled:
        try:
            uid = assert_validator_permit(
                meta, wallet.hotkey.ss58_address, config.NETUID
            )
            logger.info("validator permit ok uid=%d", uid)
        except WeightSetError as exc:
            logger.warning("skipping cycle before compute: %s", exc)
            return None

    try:
        hotkey_uids, uid_count = _hotkey_uids(meta)
    except WeightVectorError:
        logger.exception("weight vector failed structural guards; not stored")
        return None

    reconcile_deregistrations(hotkey_uids)

    try:
        vector = build_weight_vector(
            _campaigns(list_open_campaign_emissions()),
            current_block=current_block,
            hotkey_uids=hotkey_uids,
            uid_count=uid_count,
            burn_uid=config.BURN_UID,
        )
    except WeightVectorError:
        logger.exception("weight vector failed structural guards; not stored")
        return None

    values = list(vector.weights)
    stored = not _same_vector(get_latest_weight_set(), values)
    row_id = None
    if stored:
        row_id = insert_weight_set(
            computed_at_block=current_block,
            version_key=config.VERSION_KEY,
            burn_uid=config.BURN_UID,
            weights=values,
            breakdown=list(vector.breakdown),
        )
    burn_share = float(values[config.BURN_UID])

    if not sign:
        logger.info(
            "weights API refresh stored=%s at block %d",
            stored,
            current_block,
        )
        return current_block

    if not enabled:
        logger.info(
            "weights kill switch on: stored=%s at block %d, did not sign",
            stored,
            current_block,
        )
        obs.weights_computed(
            computed_at_block=current_block,
            version_key=config.VERSION_KEY,
            uid_count=uid_count,
            burn_share=burn_share,
            enabled=False,
        )
        return current_block

    try:
        set_weights(
            subtensor,
            wallet,
            meta,
            netuid=config.NETUID,
            uids=list(range(len(values))),
            weights=values,
            version_key=config.VERSION_KEY,
            burn_uid=config.BURN_UID,
        )
    except WeightSetError as exc:
        if row_id is not None:
            mark_weight_set_result(row_id, ok=False, error=str(exc))
        logger.warning("set_weights failed: %s", exc)
        obs.weights_computed(
            computed_at_block=current_block,
            version_key=config.VERSION_KEY,
            uid_count=uid_count,
            burn_share=burn_share,
            set_ok=False,
        )
        return current_block

    if row_id is not None:
        mark_weight_set_result(row_id, ok=True, error=None)
    obs.weights_computed(
        computed_at_block=current_block,
        version_key=config.VERSION_KEY,
        uid_count=uid_count,
        burn_share=burn_share,
        set_ok=True,
    )
    return current_block


class WeightsProcess:
    """Track API publication and chain-set cadence independently."""

    def __init__(
        self,
        *,
        last_stored_round: str | None = None,
        last_chain_set_block: int | None = None,
        cadence: int,
    ):
        self.last_stored_round = last_stored_round
        self.last_chain_set_block = last_chain_set_block
        self.cadence = cadence

    def tick(
        self,
        *,
        head: int,
        subtensor: Any,
        wallet: Any,
        meta: Any,
        enabled: bool,
        round_marker: str | None = None,
        sign: bool = True,
    ) -> str:
        completed = run_cycle(
            subtensor,
            wallet,
            meta,
            current_block=head,
            enabled=enabled,
            sign=sign,
        )
        if sign:
            self.last_chain_set_block = head
        if completed is not None:
            self.last_stored_round = round_marker
        return "computed" if completed is not None else "aborted"
