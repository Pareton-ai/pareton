"""Cohort selection, seeding, and round creation, driven by the watcher.

The watcher owns the subtensor connection and the head block, which is why
round creation lives on its side of the process split. Import this as
``from round.create import create_due_rounds``: binding the package itself
would shadow the builtin.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from bench.sampler import (
    compute_sample_seed,
    fetch_hf_row,
    generate_trace,
    parse_sampling_rule,
)
from builder.registry import baseline_engine_image_ref, normalize_digest
from campaign.store import get_campaign
import config
from chain.rpc import fetch_finalized_block_hash
from round.store import campaigns_with_queue, create_round

logger = logging.getLogger(__name__)


def dedupe_by_digest(
    cohort: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a cohort into keepers and duplicate images.

    Two miners can push byte-identical images from different patches. The
    earliest ``commit_block`` keeps the slot, the rest are duplicates carrying
    the id that displaced them. Cohort order is already
    ``commit_block ASC, id ASC``, so the first row seen wins.

    A ref that carries no digest cannot collide with another image, so it
    keys on itself rather than failing the whole round.
    """
    keepers: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    kept_by_digest: dict[str, Any] = {}
    for row in cohort:
        ref = str(row.get("engine_image_ref") or "")
        try:
            key = normalize_digest(ref)
        except ValueError:
            key = ref.strip().lower()
        kept = kept_by_digest.get(key)
        if kept is None:
            kept_by_digest[key] = row["id"]
            keepers.append(row)
        else:
            duplicates.append({"id": row["id"], "kept_id": kept})
    return keepers, duplicates


def should_create(
    queued: int,
    oldest_queued_at: datetime | None,
    *,
    size: int,
    max_wait_s: int,
    now: datetime | None = None,
) -> bool:
    """Whether the queue has earned a round.

    Fires at ``size`` queued submissions, or once the oldest has waited
    ``max_wait_s``. Without the second rule one miner submitting alone is
    never benched and never learns anything. A round needs one challenger.
    """
    if queued < 1:
        return False
    if queued >= size:
        return True
    if oldest_queued_at is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - oldest_queued_at).total_seconds() >= max_wait_s


def try_create_round(
    campaign: Any,
    queue: dict[str, Any],
    *,
    seed_block: int,
    seed_block_hash: str,
    row_fetcher: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Create one round for one campaign, or return None if it is not due."""
    campaign_id = str(campaign.campaign_id)
    if not should_create(
        int(queue["queued"]),
        queue["oldest_queued_at"],
        size=config.ROUND_SIZE,
        max_wait_s=config.ROUND_MAX_WAIT_S,
    ):
        return None

    bench = campaign.bench or {}
    engine_digest = bench.get("baseline_engine_image_digest")
    if not engine_digest:
        logger.warning(
            "campaign %s: no baseline engine pin; cannot run a round", campaign_id
        )
        return None
    if not campaign.gpu_skus:
        logger.warning("campaign %s: no gpu_skus; cannot run a round", campaign_id)
        return None

    seed_hex = compute_sample_seed(block_hash=seed_block_hash, campaign_id=campaign_id)
    if not campaign.sampling_rule:
        logger.warning("campaign %s: no sampling_rule; cannot run a round", campaign_id)
        return None
    # The default row fetcher indexes config/split directly; parsing fills
    # the defaults a minimal rule omits.
    rule = parse_sampling_rule(campaign.sampling_rule)
    sampled = generate_trace(
        rule=rule,
        seed_hex=seed_hex,
        row_fetcher=row_fetcher or (lambda idx: fetch_hf_row(rule, idx)),
        sample_seed_block=seed_block,
        sample_seed_block_hash=seed_block_hash,
    )
    sampled_trace_sha256 = sampled.sha256
    sampling_receipt = sampled.receipt

    return create_round(
        campaign_id=campaign_id,
        cohort_limit=config.ROUND_SIZE,
        dedupe=dedupe_by_digest,
        gpu_sku=str(campaign.gpu_skus[0]),
        baseline_image_ref=baseline_engine_image_ref(engine_digest),
        seed_block=seed_block,
        seed_block_hash=seed_block_hash,
        seed_hex=seed_hex,
        sampled_trace_sha256=sampled_trace_sha256,
        sampling_receipt=sampling_receipt,
        scoring_rule=campaign.scoring_rule,
    )


def create_due_rounds(subtensor: Any) -> list[dict[str, Any]]:
    """Create a round for every campaign whose queue has earned one.

    The seed block is the head minus ``PARETON_CHAIN_FINALITY_DEPTH``, so it
    is already settled when its hash is read. Seeding on the head itself would
    make the watcher sleep on a block the chain has not produced yet.
    """
    queues = campaigns_with_queue()
    if not queues:
        return []
    seed_block = int(subtensor.block) - config.CHAIN_FINALITY_DEPTH
    seed_block_hash = fetch_finalized_block_hash(
        subtensor, seed_block, finality_depth=config.CHAIN_FINALITY_DEPTH
    )
    created: list[dict[str, Any]] = []
    for queue in queues:
        # One campaign's bad pin or dead HF fetch must not cost every other
        # campaign its round on this cycle, and every cycle after it.
        try:
            campaign = get_campaign(queue["campaign_id"])
            if campaign is None:
                continue
            row = try_create_round(
                campaign,
                queue,
                seed_block=seed_block,
                seed_block_hash=seed_block_hash,
            )
        except Exception:
            logger.exception(
                "campaign %s: round creation failed; continuing with the rest",
                queue["campaign_id"],
            )
            continue
        if row is not None:
            logger.info(
                "campaign %s: created round %d with %d challenger(s)",
                queue["campaign_id"],
                row["ordinal"],
                len(row["challenger_ids"]),
            )
            created.append(row)
    return created
