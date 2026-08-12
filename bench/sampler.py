"""Deterministic per-submission workload sampling from a pre-baked pool.

Seed = sha256(block_hash(commit_block + offset) || patch_hash || campaign_id).
Index = int(seed, 16) % len(pool). Pure given fixed hashes (unit-testable offline).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

_SHA256_HEX_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


class SamplerError(ValueError):
    """Invalid sampler inputs (empty pool, bad rule, bad hex)."""


@dataclass(frozen=True)
class SampledTrace:
    index: int
    sha256: str
    url: str
    seed_hex: str
    sample_seed_block: int
    sample_seed_block_hash: str
    receipt: dict[str, Any]


def normalize_sha256(value: str) -> str:
    m = _SHA256_HEX_RE.fullmatch(str(value).strip())
    if not m:
        raise SamplerError(f"expected sha256:<64 hex> or 64 hex, got {value!r}")
    return f"sha256:{m.group(1).lower()}"


def resolve_workload_pool(
    *,
    workload_pool: Sequence[dict[str, Any]] | None,
    workload_trace_sha256: str,
    workload_trace_url: str,
) -> list[dict[str, str]]:
    """Return normalized pool; absent pool ⇒ single-entry back-compat pool."""
    raw = list(workload_pool) if workload_pool else None
    if not raw:
        return [
            {
                "sha256": normalize_sha256(workload_trace_sha256),
                "url": str(workload_trace_url),
            }
        ]
    out: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SamplerError(f"workload_pool[{i}] must be an object")
        sha = item.get("sha256")
        url = item.get("url")
        if not sha or not url:
            raise SamplerError(f"workload_pool[{i}] requires sha256 and url")
        out.append({"sha256": normalize_sha256(str(sha)), "url": str(url)})
    return out


def parse_sampling_rule(rule: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize sampling_rule; default offset 10 when rule is a uniform_index."""
    if rule is None:
        return {"type": "uniform_index", "seed_block_offset": 10}
    if not isinstance(rule, dict):
        raise SamplerError("sampling_rule must be an object")
    rtype = str(rule.get("type") or "uniform_index")
    if rtype != "uniform_index":
        raise SamplerError(f"unsupported sampling_rule.type: {rtype!r}")
    offset = int(rule.get("seed_block_offset", 10))
    if offset < 0:
        raise SamplerError("seed_block_offset must be >= 0")
    return {"type": "uniform_index", "seed_block_offset": offset}


def compute_sample_seed(
    *,
    block_hash: str,
    patch_hash: str,
    campaign_id: str,
) -> str:
    """Return 64-char lowercase hex seed (no sha256: prefix)."""
    bh = str(block_hash).strip().lower()
    if bh.startswith("0x"):
        bh = bh[2:]
    if not bh or not _HEX_RE.fullmatch(bh):
        raise SamplerError(f"block_hash must be hex, got {block_hash!r}")
    ph = str(patch_hash).strip().lower()
    if ph.startswith("sha256:"):
        ph = ph[len("sha256:") :]
    if not _SHA256_HEX_RE.fullmatch(ph) and not _HEX_RE.fullmatch(ph):
        raise SamplerError(f"patch_hash must be hex, got {patch_hash!r}")
    # Fixed-width hex fields + campaign_id; order matches plan:
    # sha256(block_hash || patch_hash || campaign_id).
    material = (bh + ph + str(campaign_id).strip().lower()).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def sample_from_pool(
    pool: Sequence[dict[str, str]],
    *,
    seed_hex: str,
) -> tuple[int, dict[str, str]]:
    if not pool:
        raise SamplerError("workload pool is empty")
    cleaned = str(seed_hex).strip().lower()
    if cleaned.startswith("sha256:"):
        cleaned = cleaned[len("sha256:") :]
    if not _HEX_RE.fullmatch(cleaned):
        raise SamplerError(f"seed_hex must be hex, got {seed_hex!r}")
    index = int(cleaned, 16) % len(pool)
    return index, dict(pool[index])


def sample_workload(
    *,
    pool: Sequence[dict[str, str]],
    commit_block: int,
    seed_block_offset: int,
    block_hash: str,
    patch_hash: str,
    campaign_id: str,
) -> SampledTrace:
    """Pick one pool entry; return receipt fields for submissions + events."""
    if commit_block is None:
        raise SamplerError("commit_block is required for sampling")
    seed_block = int(commit_block) + int(seed_block_offset)
    seed_hex = compute_sample_seed(
        block_hash=block_hash,
        patch_hash=patch_hash,
        campaign_id=campaign_id,
    )
    index, entry = sample_from_pool(pool, seed_hex=seed_hex)
    receipt = {
        "type": "uniform_index",
        "seed_block_offset": int(seed_block_offset),
        "sample_seed_block": seed_block,
        "sample_seed_block_hash": str(block_hash).strip().lower(),
        "seed_hex": seed_hex,
        "pool_size": len(pool),
        "index": index,
        "sampled_trace_sha256": entry["sha256"],
        "sampled_trace_url": entry["url"],
        "patch_hash": str(patch_hash).strip().lower(),
        "campaign_id": str(campaign_id).strip().lower(),
    }
    return SampledTrace(
        index=index,
        sha256=entry["sha256"],
        url=entry["url"],
        seed_hex=seed_hex,
        sample_seed_block=seed_block,
        sample_seed_block_hash=str(block_hash).strip().lower(),
        receipt=receipt,
    )
