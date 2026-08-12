"""Deterministic on-the-fly workload traces from a pinned HuggingFace dataset.

Seed (bench) = sha256(block_hash(commit_block + offset) || patch_hash || campaign_id).
Seed (calib) = sha256(campaign_id || "calib" || i).
Row i = sha256(seed || counter) % n_rows. Empty/too-long rows skip via counter.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Sequence

_SHA256_HEX_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

ALGO_VERSION = 1
MAX_PROMPT_CHARS = 8000
DEFAULT_N_PROMPTS = 32
DEFAULT_MAX_TOKENS = 128
DEFAULT_SEED_BLOCK_OFFSET = 1
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"


class SamplerError(ValueError):
    """Invalid sampler inputs or generation failure."""


@dataclass(frozen=True)
class SampledTrace:
    sha256: str
    body: bytes
    seed_hex: str
    sample_seed_block: int
    sample_seed_block_hash: str
    row_indices: tuple[int, ...]
    receipt: dict[str, Any]


def normalize_sha256(value: str) -> str:
    m = _SHA256_HEX_RE.fullmatch(str(value).strip())
    if not m:
        raise SamplerError(f"expected sha256:<64 hex> or 64 hex, got {value!r}")
    return f"sha256:{m.group(1).lower()}"


def parse_sampling_rule(rule: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize and validate an hf_rows sampling_rule."""
    if not isinstance(rule, dict):
        raise SamplerError("sampling_rule must be an object")
    rtype = str(rule.get("type") or "")
    if rtype != "hf_rows":
        raise SamplerError(f"unsupported sampling_rule.type: {rtype!r}")
    offset = int(rule.get("seed_block_offset", DEFAULT_SEED_BLOCK_OFFSET))
    if offset < 0:
        raise SamplerError("seed_block_offset must be >= 0")
    dataset = str(rule.get("dataset") or "").strip()
    revision = str(rule.get("revision") or "").strip()
    if not dataset or not revision:
        raise SamplerError("hf_rows rule requires dataset and revision")
    n_rows = int(rule.get("n_rows") or 0)
    n_prompts = int(rule.get("n_prompts") or DEFAULT_N_PROMPTS)
    max_tokens = int(rule.get("max_tokens") or DEFAULT_MAX_TOKENS)
    algo_version = int(rule.get("algo_version") or ALGO_VERSION)
    if n_rows < 1:
        raise SamplerError("n_rows must be >= 1")
    if n_prompts < 1:
        raise SamplerError("n_prompts must be >= 1")
    if n_prompts > n_rows:
        raise SamplerError("n_prompts must be <= n_rows")
    if max_tokens < 1:
        raise SamplerError("max_tokens must be >= 1")
    if algo_version != ALGO_VERSION:
        raise SamplerError(f"unsupported algo_version: {algo_version}")
    return {
        "type": "hf_rows",
        "seed_block_offset": offset,
        "dataset": dataset,
        "revision": revision,
        "config": str(rule.get("config") or "default"),
        "split": str(rule.get("split") or "train"),
        "n_rows": n_rows,
        "n_prompts": n_prompts,
        "max_tokens": max_tokens,
        "algo_version": algo_version,
    }


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
    material = (bh + ph + str(campaign_id).strip().lower()).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def calib_seed(campaign_id: str, index: int) -> str:
    """Deterministic calibration seed: sha256(campaign_id || 'calib' || i)."""
    if int(index) < 0:
        raise SamplerError("calib index must be >= 0")
    material = f"{str(campaign_id).strip().lower()}calib{int(index)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def extract_prompt(row: dict[str, Any]) -> str:
    """First non-empty user message from a SWE-agent trajectory row."""
    traj = row.get("trajectory")
    if traj is None:
        return ""
    msgs = traj if isinstance(traj, list) else json.loads(traj)
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("text") or m.get("content") or ""
        content = str(content)
        if content:
            return content
    return ""


def prompt_ok(prompt: str) -> bool:
    return bool(prompt) and len(prompt) <= MAX_PROMPT_CHARS


def _index_at(seed_hex: str, counter: int, n_rows: int) -> int:
    material = f"{seed_hex}:{counter}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest(), 16) % n_rows


def select_row_indices(
    *,
    seed_hex: str,
    n_rows: int,
    n_prompts: int,
    row_ok: Callable[[int], bool],
) -> list[int]:
    """Pick n_prompts distinct row indices; skip rows where row_ok is False."""
    cleaned = str(seed_hex).strip().lower()
    if cleaned.startswith("sha256:"):
        cleaned = cleaned[len("sha256:") :]
    if not _HEX_RE.fullmatch(cleaned):
        raise SamplerError(f"seed_hex must be hex, got {seed_hex!r}")
    if n_rows < 1 or n_prompts < 1:
        raise SamplerError("n_rows and n_prompts must be >= 1")
    chosen: list[int] = []
    seen: set[int] = set()
    counter = 0
    max_tries = n_rows * 8
    while len(chosen) < n_prompts:
        if counter >= max_tries:
            raise SamplerError(
                f"could not fill {n_prompts} prompts from {n_rows} rows"
            )
        idx = _index_at(cleaned, counter, n_rows)
        counter += 1
        if idx in seen:
            continue
        seen.add(idx)
        if not row_ok(idx):
            continue
        chosen.append(idx)
    return chosen


def build_trace_json(
    prompts: Sequence[str],
    *,
    max_tokens: int,
    meta_name: str,
) -> dict[str, Any]:
    requests = []
    for j, p in enumerate(prompts):
        requests.append(
            {
                "id": f"hf-{j:03d}",
                "arrival_offset_ms": j * 200,
                "prompt": p,
                "max_tokens": int(max_tokens),
                "sampling": {"temperature": 0.0, "top_p": 1.0},
            }
        )
    return {
        "schema_version": 1,
        "meta": {
            "name": meta_name,
            "description": f"{len(requests)} prompts sampled on the fly",
        },
        "requests": requests,
    }


def encode_trace(trace: dict[str, Any]) -> bytes:
    return (json.dumps(trace, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def fetch_hf_row(rule: dict[str, Any], index: int) -> dict[str, Any]:
    """Fetch one row from the HuggingFace datasets-server. Network path."""
    params = urllib.parse.urlencode(
        {
            "dataset": rule["dataset"],
            "config": rule["config"],
            "split": rule["split"],
            "revision": rule["revision"],
            "offset": int(index),
            "length": 1,
        }
    )
    url = f"{HF_ROWS_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "pareton-sampler/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise SamplerError(f"hf row fetch failed at offset {index}: {exc}") from exc
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise SamplerError(f"hf row fetch empty at offset {index}")
    row = rows[0].get("row") if isinstance(rows[0], dict) else None
    if not isinstance(row, dict):
        raise SamplerError(f"hf row missing at offset {index}")
    return row


def generate_trace(
    *,
    rule: dict[str, Any],
    seed_hex: str,
    row_fetcher: Callable[[int], dict[str, Any]],
    sample_seed_block: int = 0,
    sample_seed_block_hash: str = "",
) -> SampledTrace:
    """Build a trace from hash-selected rows. row_fetcher is injected in tests."""
    parsed = parse_sampling_rule(rule)
    cache: dict[int, str] = {}

    def row_ok(idx: int) -> bool:
        if idx not in cache:
            try:
                cache[idx] = extract_prompt(row_fetcher(idx))
            except SamplerError:
                raise
            except Exception as exc:
                raise SamplerError(f"row fetch failed at {idx}: {exc}") from exc
        return prompt_ok(cache[idx])

    indices = select_row_indices(
        seed_hex=seed_hex,
        n_rows=int(parsed["n_rows"]),
        n_prompts=int(parsed["n_prompts"]),
        row_ok=row_ok,
    )
    prompts = [cache[i] for i in indices]
    trace = build_trace_json(
        prompts,
        max_tokens=int(parsed["max_tokens"]),
        meta_name=f"hf-rows-{seed_hex[:12]}",
    )
    body = encode_trace(trace)
    sha = "sha256:" + hashlib.sha256(body).hexdigest()
    receipt = {
        "type": "hf_rows",
        "algo_version": parsed["algo_version"],
        "dataset": parsed["dataset"],
        "revision": parsed["revision"],
        "config": parsed["config"],
        "split": parsed["split"],
        "n_rows": parsed["n_rows"],
        "n_prompts": parsed["n_prompts"],
        "max_tokens": parsed["max_tokens"],
        "seed_block_offset": parsed["seed_block_offset"],
        "sample_seed_block": int(sample_seed_block),
        "sample_seed_block_hash": str(sample_seed_block_hash).strip().lower(),
        "seed_hex": str(seed_hex).strip().lower(),
        "row_indices": list(indices),
        "sampled_trace_sha256": sha,
    }
    return SampledTrace(
        sha256=sha,
        body=body,
        seed_hex=str(seed_hex).strip().lower(),
        sample_seed_block=int(sample_seed_block),
        sample_seed_block_hash=str(sample_seed_block_hash).strip().lower(),
        row_indices=tuple(indices),
        receipt=receipt,
    )


def sample_workload(
    *,
    rule: dict[str, Any],
    commit_block: int,
    block_hash: str,
    patch_hash: str,
    campaign_id: str,
    row_fetcher: Callable[[int], dict[str, Any]] | None = None,
) -> SampledTrace:
    """Generate one submission trace from a future-block seed."""
    if commit_block is None:
        raise SamplerError("commit_block is required for sampling")
    parsed = parse_sampling_rule(rule)
    seed_block = int(commit_block) + int(parsed["seed_block_offset"])
    seed_hex = compute_sample_seed(
        block_hash=block_hash,
        patch_hash=patch_hash,
        campaign_id=campaign_id,
    )
    fetcher = row_fetcher or (lambda idx: fetch_hf_row(parsed, idx))
    return generate_trace(
        rule=parsed,
        seed_hex=seed_hex,
        row_fetcher=fetcher,
        sample_seed_block=seed_block,
        sample_seed_block_hash=str(block_hash).strip().lower(),
    )
