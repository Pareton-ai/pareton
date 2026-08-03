"""Poll chain for Pareton patch commitments and enqueue submissions."""

from __future__ import annotations

import logging
from typing import Any, Callable

from campaign.store import get_campaign, insert_submission
from chain.commitment import (
    PatchCommitment,
    build_patch_commitments,
)
from chain.rpc import fetch_chain_view

logger = logging.getLogger(__name__)


def ingest_commitment(com: PatchCommitment) -> str | None:
    """Insert a submission from a commitment. Returns submission id or None if dupe/invalid."""
    campaign = get_campaign(com.campaign_id)
    if campaign is None:
        logger.info(
            "skip commitment: unknown campaign_id=%s hotkey=%s",
            com.campaign_id,
            com.hotkey[:16],
        )
        return None
    if campaign.status != "open":
        logger.info(
            "skip commitment: campaign %s status=%s",
            com.campaign_id,
            campaign.status,
        )
        return None

    sid = insert_submission(
        campaign_id=com.campaign_id,
        patch_hash=com.patch_hash,
        hotkey=com.hotkey,
        baseline_commit=com.baseline_commit,
        retrieval_url=com.retrieval_url,
        commit_block=com.commit_block,
    )
    if sid is None:
        logger.info(
            "skip commitment: duplicate patch_hash=%s campaign=%s",
            com.patch_hash,
            com.campaign_id,
        )
        return None
    logger.info(
        "enqueued submission %s patch_hash=%s hotkey=%s",
        sid,
        com.patch_hash,
        com.hotkey[:16],
    )
    return str(sid)


def scan_chain(
    subtensor: Any,
    netuid: int,
    *,
    network: str = "finney",
    ingest: Callable[[PatchCommitment], str | None] = ingest_commitment,
) -> tuple[list[str], list[str]]:
    """Fetch revealed commitments and enqueue new submissions.

    Returns (new submission ids, registered hotkeys from the metagraph).
    """
    meta, revealed, _block, _block_hash = fetch_chain_view(
        subtensor, netuid, network=network
    )
    commitments = build_patch_commitments(meta, revealed)
    created: list[str] = []
    for com in commitments.values():
        sid = ingest(com)
        if sid:
            created.append(sid)
    hotkeys = [str(hk) for hk in getattr(meta, "hotkeys", [])]
    return created, hotkeys


def ingest_mock_commitment(
    *,
    campaign_id: str,
    hotkey: str,
    baseline_commit: str,
    patch_hash: str,
    retrieval_url: str,
    commit_block: int = 1,
    uid: int = 0,
) -> str | None:
    """Test/helper path that does not touch the chain."""
    com = PatchCommitment(
        uid=uid,
        hotkey=hotkey,
        coldkey="",
        commit_block=commit_block,
        campaign_id=campaign_id,
        baseline_commit=baseline_commit.lower(),
        patch_hash=patch_hash.lower(),
        retrieval_url=retrieval_url,
        raw="",
    )
    return ingest_commitment(com)
