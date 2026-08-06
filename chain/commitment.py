"""Pareton patch commitment payload (reveal-commitment compatible)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

logger = logging.getLogger(__name__)

PATCH_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PatchCommitment:
    """One miner's most recent Pareton patch commitment."""

    uid: int
    hotkey: str
    coldkey: str
    commit_block: int
    campaign_id: str
    baseline_commit: str
    patch_hash: str
    retrieval_url: str
    raw: str

    def as_key(self) -> tuple[str, str]:
        return (self.campaign_id, self.patch_hash)


class _MetagraphLike(Protocol):
    hotkeys: list[str]


def _parse_v2(raw: str) -> dict[str, str] | None:
    """Parse `v2|campaign_id|baseline_commit|sha256:<hex>|retrieval_url`."""
    parts = raw.split("|")
    if len(parts) != 5:
        return None
    _v, campaign_id, baseline_commit, patch_hash, retrieval_url = parts
    if not _UUID_RE.match(campaign_id.strip()):
        return None
    if not _GIT_SHA_RE.match(baseline_commit.strip().lower()):
        return None
    if not PATCH_HASH_RE.match(patch_hash.strip().lower()):
        return None
    if not retrieval_url.strip().startswith(("https://", "http://")):
        return None
    return {
        "campaign_id": str(UUID(campaign_id.strip())),
        "baseline_commit": baseline_commit.strip().lower(),
        "patch_hash": patch_hash.strip().lower(),
        "retrieval_url": retrieval_url.strip(),
    }


def parse_patch_commitment(raw: str) -> dict[str, str] | None:
    """Parse a Pareton patch commitment (v2 pipe format or legacy v1 JSON)."""
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("v2|"):
        return _parse_v2(raw)
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("v") != 1:
        return None

    campaign_id = obj.get("campaign_id")
    baseline_commit = obj.get("baseline_commit")
    patch_hash = obj.get("patch_hash")
    retrieval_url = obj.get("retrieval_url")

    if not isinstance(campaign_id, str) or not _UUID_RE.match(campaign_id.strip()):
        return None
    if not isinstance(baseline_commit, str) or not _GIT_SHA_RE.match(
        baseline_commit.strip().lower()
    ):
        return None
    if not isinstance(patch_hash, str) or not PATCH_HASH_RE.match(
        patch_hash.strip().lower()
    ):
        return None
    if not isinstance(retrieval_url, str) or not retrieval_url.strip().startswith(
        ("https://", "http://")
    ):
        return None

    return {
        "campaign_id": str(UUID(campaign_id.strip())),
        "baseline_commit": baseline_commit.strip().lower(),
        "patch_hash": patch_hash.strip().lower(),
        "retrieval_url": retrieval_url.strip(),
    }


def encode_patch_commitment(
    *,
    campaign_id: str,
    baseline_commit: str,
    patch_hash: str,
    retrieval_url: str,
) -> str:
    """Canonical v2 wire string for Commitments.set_commitment.

    Positional pipe format: finney's CommitmentInfo allows MaxFields=3
    (3 x 128-byte Raw chunks = 384 bytes), and the v1 JSON skeleton alone
    costs ~75 bytes; this format keeps full payloads (~356 bytes) inside
    the bound. None of the values may contain '|'.
    """
    cid = str(UUID(campaign_id))
    baseline = baseline_commit.lower()
    phash = patch_hash.lower()
    url = retrieval_url.strip()
    for name, value in (
        ("campaign_id", cid),
        ("baseline_commit", baseline),
        ("patch_hash", phash),
        ("retrieval_url", url),
    ):
        if "|" in value:
            raise ValueError(f"{name} must not contain '|'")
    if not url.startswith(("https://", "http://")):
        raise ValueError("retrieval_url must start with http:// or https://")
    raw = "|".join(["v2", cid, baseline, phash, url])
    if parse_patch_commitment(raw) is None:
        raise ValueError("encoded commitment failed parse round-trip")
    return raw


def build_patch_commitments(
    metagraph: _MetagraphLike,
    revealed: dict[str, list[tuple[int, str]]],
) -> dict[int, PatchCommitment]:
    """Fold revealed commitments into `{uid: PatchCommitment}` (latest per hotkey)."""
    out: dict[int, PatchCommitment] = {}
    hotkeys = list(metagraph.hotkeys)
    coldkeys = list(getattr(metagraph, "coldkeys", []) or [])

    for uid, hotkey in enumerate(hotkeys):
        hotkey_str = str(hotkey)
        reveals = revealed.get(hotkey_str) or []
        if not reveals:
            continue
        block, raw = max(reveals, key=lambda p: p[0])
        parsed = parse_patch_commitment(raw)
        if parsed is None:
            logger.debug(
                "UID %d (%s): commitment at block %d is not valid pareton JSON",
                uid,
                hotkey_str[:16] + "...",
                block,
            )
            continue
        coldkey_str = str(coldkeys[uid]) if uid < len(coldkeys) else ""
        out[uid] = PatchCommitment(
            uid=uid,
            hotkey=hotkey_str,
            coldkey=coldkey_str,
            commit_block=int(block),
            campaign_id=parsed["campaign_id"],
            baseline_commit=parsed["baseline_commit"],
            patch_hash=parsed["patch_hash"],
            retrieval_url=parsed["retrieval_url"],
            raw=raw,
        )
    return out


def fetch_revealed_commitments(
    subtensor: Any,
    netuid: int,
    *,
    network: str = "finney",
    attempts: int = 3,
    delay_s: float = 30.0,
) -> dict[str, list[tuple[int, str]]]:
    """Fetch revealed commitments from the subnet."""
    from chain.rpc import fetch_revealed_commitments as _fetch

    return _fetch(
        subtensor, netuid, network=network, attempts=attempts, delay_s=delay_s
    )


def fetch_metagraph(subtensor: Any, netuid: int, *, network: str = "finney") -> Any:
    """Fetch metagraph only (drops block/hash from the RPC tuple)."""
    from chain.rpc import fetch_metagraph as _fetch

    metagraph, _block, _block_hash = _fetch(subtensor, netuid, network=network)
    return metagraph
