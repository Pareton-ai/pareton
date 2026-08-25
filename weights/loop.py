"""The pareton-weights cadence: compute, store, and sign one vector.

One process owns all three so `/v1/weights` can never serve a vector that was
not also sent to the chain. The builder is pure; this module is the I/O.
"""

from __future__ import annotations

import logging
from typing import Any

import config
from chain.weights import (
    WeightSetError,
    assert_validator_permit,
    dense_to_sparse,
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


def _hotkey_uids(meta: Any) -> tuple[dict[str, int], int]:
    """Live ``hotkey -> uid`` map and metagraph length. UIDs are never stored."""
    hotkeys = [str(hk) for hk in getattr(meta, "hotkeys", [])]
    return {hk: uid for uid, hk in enumerate(hotkeys)}, len(hotkeys)


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


def run_cycle(
    subtensor: Any,
    wallet: Any,
    meta: Any,
    *,
    current_block: int,
    enabled: bool,
) -> int | None:
    """One compute cycle. Returns the block used, or None if nothing was stored.

    A builder guard abort leaves no ``weight_sets`` row. A chain rejection
    stores the row with ``set_ok=false`` and the loop keeps running.
    """
    hotkey_uids, uid_count = _hotkey_uids(meta)
    reconcile_deregistrations(hotkey_uids)

    if enabled:
        try:
            uid = assert_validator_permit(
                meta, wallet.hotkey.ss58_address, config.NETUID
            )
            logger.info("validator permit ok uid=%d", uid)
        except WeightSetError as exc:
            logger.warning("skipping cycle before compute: %s", exc)
            return None

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

    row_id = insert_weight_set(
        computed_at_block=current_block,
        version_key=config.VERSION_KEY,
        burn_uid=config.BURN_UID,
        weights=list(vector.weights),
        breakdown=list(vector.breakdown),
    )
    uids, values = dense_to_sparse(vector.weights)
    burn_share = next(
        (float(w) for uid, w in zip(uids, values) if uid == config.BURN_UID),
        0.0,
    )

    if not enabled:
        logger.info(
            "weights kill switch on: stored row %d at block %d, did not sign",
            row_id,
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
            uids=uids,
            weights=values,
            version_key=config.VERSION_KEY,
            burn_uid=config.BURN_UID,
        )
    except WeightSetError as exc:
        mark_weight_set_result(row_id, ok=False, error=str(exc))
        logger.warning("set_weights failed; stored set_ok=false: %s", exc)
        obs.weights_computed(
            computed_at_block=current_block,
            version_key=config.VERSION_KEY,
            uid_count=uid_count,
            burn_share=burn_share,
            set_ok=False,
        )
        return current_block

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
    """Single-flight cadence. A long cycle skips the next tick, never overlaps."""

    def __init__(self, *, last_block: int | None, cadence: int):
        self.last_block = last_block
        self.cadence = cadence
        self._running = False

    def tick(
        self,
        *,
        head: int,
        subtensor: Any,
        wallet: Any,
        meta: Any,
        enabled: bool,
        force: bool = False,
    ) -> str:
        if self._running:
            return "skipped_overlap"
        if not force and not cycle_due(self.last_block, head, self.cadence):
            return "waiting"
        self._running = True
        try:
            block = run_cycle(
                subtensor,
                wallet,
                meta,
                current_block=head,
                enabled=enabled,
            )
            if block is not None:
                self.last_block = block
                return "computed"
            return "aborted"
        finally:
            self._running = False


def last_computed_block() -> int | None:
    row = get_latest_weight_set()
    if row is None:
        return None
    return int(row["computed_at_block"])
