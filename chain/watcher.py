"""Poll chain for Pareton patch commitments and enqueue submissions."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable

import config
from campaign.store import (
    get_campaign,
    get_submission_for_campaign,
    insert_submission,
    payment_ref_consumed,
)
from chain.commitment import (
    PatchCommitment,
    build_patch_commitments,
)
from chain.payment import (
    BlockPaymentView,
    PaymentCheck,
    fee_rao,
    fetch_block_payment_view,
    verify_payment,
)
from chain.rpc import fetch_chain_view
from gate.integrity import check_integrity, patch_fingerprint_bytes
from observability import events as obs
from storage.s3 import fetch_patch_bytes, is_allowed_retrieval_url, patch_url_hotkey

logger = logging.getLogger(__name__)

BlockFetcher = Callable[[int], BlockPaymentView | None]
PatchFetcher = Callable[[str], bytes]


def check_fee_proof(
    com: PatchCommitment,
    fetch_block: BlockFetcher | None,
) -> PaymentCheck:
    """Verify the commitment's fee proof. Only called when the fee is on."""
    if com.payment_block is None or com.payment_tx is None:
        return PaymentCheck.reject("payment_proof_missing")
    if payment_ref_consumed(com.payment_block, com.payment_tx):
        return PaymentCheck.reject("payment_ref_already_used")
    if fetch_block is None:
        return PaymentCheck.reject("payment_no_chain_access")
    view = fetch_block(com.payment_block)
    if view is None:
        return PaymentCheck.reject("payment_block_unavailable")
    return verify_payment(
        extrinsics=view.extrinsics,
        events=view.events,
        extrinsic_index=com.payment_tx,
        recipient=config.PAYMENT_RECIPIENT_ADDRESS,
        min_amount_rao=fee_rao(config.SUBMISSION_FEE_TAO),
        hotkey=com.hotkey,
        coldkey=com.coldkey,
    )


def ingest_commitment(
    com: PatchCommitment,
    *,
    fetch_block: BlockFetcher | None = None,
    fetcher: PatchFetcher | None = None,
) -> str | None:
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
    if get_submission_for_campaign(com.campaign_id, com.patch_hash) is not None:
        return None
    # Reject before insert so a mismatched/junk URL cannot burn the
    # (campaign_id, patch_hash) first-seen dedupe slot.
    if not is_allowed_retrieval_url(com.retrieval_url):
        logger.info(
            "skip commitment: retrieval_url not allowlisted hotkey=%s url=%s",
            com.hotkey[:16],
            com.retrieval_url,
        )
        return None
    url_hotkey = patch_url_hotkey(com.retrieval_url)
    if url_hotkey != com.hotkey:
        logger.info(
            "skip commitment: retrieval_url hotkey mismatch signer=%s url_hotkey=%s",
            com.hotkey[:16],
            (url_hotkey or "")[:16],
        )
        return None

    # No GPU spend without proof the miner paid: reject before insert so a
    # missing or junk proof cannot burn the first-seen dedupe slot either.
    payment_block = payment_tx = None
    if config.SUBMISSION_FEE_TAO > 0:
        check = check_fee_proof(com, fetch_block)
        if not check.ok:
            logger.info(
                "skip commitment: %s hotkey=%s patch_hash=%s",
                check.reason,
                com.hotkey[:16],
                com.patch_hash,
            )
            return None
        payment_block, payment_tx = com.payment_block, com.payment_tx

    integrity = check_integrity(
        retrieval_url=com.retrieval_url,
        expected_patch_hash=com.patch_hash,
        hotkey=com.hotkey,
        fetcher=fetcher or fetch_patch_bytes,
    )
    if not integrity.ok:
        logger.info(
            "skip commitment: integrity failed patch_hash=%s reason=%s",
            com.patch_hash,
            integrity.reason,
        )
        return None
    patch_bytes = integrity.evidence["patch_bytes"]
    patch_fingerprint = patch_fingerprint_bytes(patch_bytes)

    sid = insert_submission(
        campaign_id=com.campaign_id,
        patch_hash=com.patch_hash,
        hotkey=com.hotkey,
        baseline_commit=com.baseline_commit,
        retrieval_url=com.retrieval_url,
        commit_block=com.commit_block,
        payment_block=payment_block,
        payment_tx=payment_tx,
        patch_fingerprint=patch_fingerprint,
    )
    if sid is None:
        logger.info(
            "skip commitment: duplicate fingerprint=%s campaign=%s",
            patch_fingerprint,
            com.campaign_id,
        )
        return None
    logger.info(
        "enqueued submission %s patch_hash=%s hotkey=%s",
        sid,
        com.patch_hash,
        com.hotkey[:16],
    )
    obs.submission_ingested(
        submission_id=str(sid),
        campaign_id=com.campaign_id,
        patch_sha256=com.patch_hash,
        hotkey=com.hotkey,
    )
    return str(sid)


def scan_chain(
    subtensor: Any,
    netuid: int,
    *,
    network: str = "finney",
    ingest: Callable[[PatchCommitment], str | None] | None = None,
) -> tuple[list[str], list[str]]:
    """Fetch revealed commitments and enqueue new submissions.

    Returns (new submission ids, registered hotkeys from the metagraph).
    """
    if ingest is None:
        ingest = partial(
            ingest_commitment,
            fetch_block=partial(fetch_block_payment_view, subtensor),
        )
    meta, revealed, block, _block_hash = fetch_chain_view(
        subtensor, netuid, network=network
    )
    commitments = build_patch_commitments(meta, revealed)
    # Plaintext makes patch_hash public at commit time; ingest in chain
    # chronology so a later lower-UID copycat cannot win first-seen dedupe.
    ordered = sorted(commitments.values(), key=lambda c: (c.commit_block, c.hotkey))
    created: list[str] = []
    for com in ordered:
        sid = ingest(com)
        if sid:
            created.append(sid)
    hotkeys = [str(hk) for hk in getattr(meta, "hotkeys", [])]
    obs.chain_scanned(
        block=block,
        commitments_seen=len(ordered),
        ingested=len(created),
    )
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
