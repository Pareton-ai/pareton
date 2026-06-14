"""In-container content fingerprinting for duplicate submission detection."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import config as validator_config
from .chain import CommitmentRecord
from .state import DUPLICATE_SUBMISSION_REASON, EvaluationRecord, _atomic_write_json

logger = logging.getLogger(__name__)

FINGERPRINT_FILE_NAME = "content_fingerprints.json"

FingerprintEntry = dict[str, Any]
FingerprintRegistry = dict[str, Any]

_EMPTY_FINGERPRINT = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class FingerprintCheckResult:
    """Outcome of a fingerprint check for one participant."""

    dq_reason: str | None = None
    fingerprint: str | None = None
    superseded_owner: FingerprintEntry | None = None


def _registry_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / FINGERPRINT_FILE_NAME


def load_fingerprint_registry(state_dir: str | Path) -> FingerprintRegistry:
    try:
        from cacheon_db.loaders import load_fingerprint_registry_dict

        data = load_fingerprint_registry_dict()
        if data is not None:
            return data
    except Exception:
        logger.debug("Postgres fingerprint registry load failed", exc_info=True)

    path = _registry_path(state_dir)
    if not path.exists():
        return {"entries": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load %s (%s); starting fresh", path, exc)
        return {"entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return {"entries": {}}
    return {"entries": entries}


def save_fingerprint_registry(
    state_dir: str | Path, registry: FingerprintRegistry
) -> None:
    _atomic_write_json(_registry_path(state_dir), registry)


def composite_fingerprint(file_hashes: dict[str, str]) -> str:
    """SHA-256 of sorted ``path:hash`` lines."""
    if not file_hashes:
        return _EMPTY_FINGERPRINT
    lines = [f"{path}:{digest}" for path, digest in sorted(file_hashes.items())]
    payload = "\n".join(lines).encode()
    return hashlib.sha256(payload).hexdigest()


def _docker_exec(
    container_id: str, args: list[str], *, timeout_s: float = 120
) -> str | None:
    import subprocess

    cmd = ["docker", "exec", container_id, *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("docker exec failed for %s: %s", container_id[:12], exc)
        return None
    if result.returncode != 0:
        logger.warning(
            "docker exec rc=%d for %s: %s",
            result.returncode,
            " ".join(args[:3]),
            (result.stderr or result.stdout or "").strip()[:200],
        )
        return None
    return result.stdout


def _list_fingerprint_files(container_id: str) -> list[str] | None:
    parts = []
    for root in validator_config.FINGERPRINT_PATHS:
        if root == "/start.sh":
            parts.append("if [ -f /start.sh ]; then echo /start.sh; fi")
        else:
            parts.append(
                f'if [ -e "{root}" ]; then find "{root}" -type f 2>/dev/null; fi'
            )
    script = "; ".join(parts)
    out = _docker_exec(container_id, ["sh", "-c", script], timeout_s=60)
    if out is None:
        return None
    files: list[str] = []
    for line in out.splitlines():
        path = line.strip()
        if not path or path.startswith("/models/") or path == "/models":
            continue
        files.append(path)
    return sorted(set(files))


def _file_size_bytes(container_id: str, path: str) -> int | None:
    out = _docker_exec(
        container_id,
        ["sh", "-c", f'stat -c%s "{path}" 2>/dev/null || stat -f%z "{path}"'],
        timeout_s=30,
    )
    if out is None:
        return None
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _hash_file(container_id: str, path: str) -> str | None:
    out = _docker_exec(
        container_id,
        ["sh", "-c", f'sha256sum "{path}" 2>/dev/null | cut -d" " -f1'],
        timeout_s=600,
    )
    if out is None:
        return None
    digest = out.strip().splitlines()[-1].strip()
    if len(digest) != 64:
        return None
    return digest


def compute_content_fingerprint(container_id: str) -> str | None:
    """Hash allowlisted in-container files. Returns None on infra failure."""
    files = _list_fingerprint_files(container_id)
    if files is None:
        return None

    max_bytes = validator_config.FINGERPRINT_MAX_FILE_BYTES
    file_hashes: dict[str, str] = {}
    for path in files:
        size = _file_size_bytes(container_id, path)
        if size is None:
            logger.warning("Skipping fingerprint file %s (stat failed)", path)
            continue
        if size > max_bytes:
            logger.warning(
                "Skipping fingerprint file %s (%d bytes > cap %d)",
                path,
                size,
                max_bytes,
            )
            continue
        digest = _hash_file(container_id, path)
        if digest is None:
            logger.warning("Skipping fingerprint file %s (hash failed)", path)
            continue
        file_hashes[path] = digest

    fp = composite_fingerprint(file_hashes)
    logger.info(
        "Content fingerprint for %s: %s (%d file(s))",
        container_id[:12],
        fp[:16],
        len(file_hashes),
    )
    return fp


def check_duplicate(
    registry: FingerprintRegistry,
    fingerprint: str,
    hotkey: str,
    commit_block: int,
) -> FingerprintEntry | None:
    """Return the canonical owner entry if this submission is a later copy."""
    entry = registry.get("entries", {}).get(fingerprint)
    if not entry:
        return None
    if entry.get("hotkey") == hotkey:
        return None
    owner_block = int(entry.get("commit_block", 0))
    if commit_block > owner_block:
        return entry
    return None


def find_registry_supersession(
    registry: FingerprintRegistry,
    fingerprint: str,
    hotkey: str,
    commit_block: int,
) -> FingerprintEntry | None:
    """Return the registry entry displaced when current is an earlier canonical owner."""
    entry = registry.get("entries", {}).get(fingerprint)
    if not entry:
        return None
    if entry.get("hotkey") == hotkey:
        return None
    owner_block = int(entry.get("commit_block", 0))
    if commit_block < owner_block:
        return entry
    return None


def register_fingerprint(
    registry: FingerprintRegistry,
    fingerprint: str,
    *,
    hotkey: str,
    commit_block: int,
    uid: int,
    image: str,
    digest: str,
) -> None:
    entries = registry.setdefault("entries", {})
    existing = entries.get(fingerprint)
    new_entry: FingerprintEntry = {
        "hotkey": hotkey,
        "commit_block": commit_block,
        "uid": uid,
        "image": image,
        "digest": digest,
        "registered_at": time.time(),
    }
    if existing is None:
        entries[fingerprint] = new_entry
        _mirror_fingerprint(fingerprint, new_entry)
        return
    if existing.get("hotkey") == hotkey:
        entries[fingerprint] = new_entry
        _mirror_fingerprint(fingerprint, new_entry)
        return
    if commit_block < int(existing.get("commit_block", 0)):
        entries[fingerprint] = new_entry
        _mirror_fingerprint(fingerprint, new_entry)


def _mirror_fingerprint(fingerprint: str, entry: FingerprintEntry) -> None:
    try:
        from cacheon_db import sync_fingerprint

        sync_fingerprint(
            fingerprint,
            uid=int(entry["uid"]),
            hotkey=str(entry["hotkey"]),
            commit_block=int(entry["commit_block"]),
            image=str(entry["image"]),
            digest=str(entry["digest"]),
            registered_at=float(entry["registered_at"]),
        )
    except Exception:
        logger.debug("Postgres fingerprint mirror failed", exc_info=True)


def duplicate_submission_reason(owner: FingerprintEntry) -> str:
    hotkey = owner.get("hotkey", "?")
    block = owner.get("commit_block", "?")
    return (
        f"{DUPLICATE_SUBMISSION_REASON}: matches hotkey {hotkey} (commit_block {block})"
    )


def canonical_owner_entry(com: CommitmentRecord) -> FingerprintEntry:
    return {
        "uid": com.uid,
        "commit_block": com.commit_block,
        "hotkey": com.hotkey,
    }


def retro_dq_record(
    record: EvaluationRecord,
    canonical_owner: FingerprintEntry,
) -> EvaluationRecord:
    """Mark an already-evaluated participant as duplicate of ``canonical_owner``."""
    return replace(
        record,
        score=0.0,
        speed_improvement=0.0,
        disqualified=True,
        disqualify_reason=duplicate_submission_reason(canonical_owner),
    )


def entry_matches_record(entry: FingerprintEntry, record: EvaluationRecord) -> bool:
    return record.hotkey == entry.get("hotkey") and record.commit_block == int(
        entry.get("commit_block", -1)
    )


def apply_fingerprint_supersede_records(
    result: FingerprintCheckResult,
    com: CommitmentRecord,
    *,
    leader_record: EvaluationRecord | None,
    ru_record: EvaluationRecord | None,
    challenger_records: list[tuple[int, EvaluationRecord]],
) -> tuple[
    EvaluationRecord | None,
    EvaluationRecord | None,
    list[tuple[int, EvaluationRecord]],
    list[str],
]:
    """Retro-DQ participants registered earlier in-round with a later copy.

    Returns updated leader, runner-up, challengers, and ``eval_key`` values
    removed from teacher-forcing.
    """
    if result.superseded_owner is None:
        return leader_record, ru_record, challenger_records, []

    canonical = canonical_owner_entry(com)
    superseded = result.superseded_owner
    removed_keys: list[str] = []

    def _maybe_dq(record: EvaluationRecord | None) -> EvaluationRecord | None:
        if record is None or not entry_matches_record(superseded, record):
            return record
        if record.disqualified:
            return record
        dq = retro_dq_record(record, canonical)
        removed_keys.append(dq.eval_key)
        logger.info(
            "Retro-DQ UID %d (%s): %s",
            dq.uid,
            dq.hotkey[:16],
            dq.disqualify_reason,
        )
        return dq

    new_challengers = [(idx, _maybe_dq(rec)) for idx, rec in challenger_records]
    return (
        _maybe_dq(leader_record),
        _maybe_dq(ru_record),
        new_challengers,
        removed_keys,
    )


def check_and_register_fingerprint(
    registry: FingerprintRegistry | None,
    *,
    container_id: str,
    com: CommitmentRecord,
    state_dir: str | Path,
) -> FingerprintCheckResult:
    """Check for duplicates, register on success, surface incumbent supersession."""
    if registry is None:
        return FingerprintCheckResult()

    fingerprint = compute_content_fingerprint(container_id)
    if fingerprint is None:
        logger.warning(
            "UID %d (%s): fingerprint check skipped (infra failure)",
            com.uid,
            com.hotkey[:16],
        )
        return FingerprintCheckResult()

    owner = check_duplicate(registry, fingerprint, com.hotkey, com.commit_block)
    if owner is not None:
        reason = duplicate_submission_reason(owner)
        logger.info(
            "UID %d (%s) DQ'd: %s (fingerprint=%s)",
            com.uid,
            com.hotkey[:16],
            reason,
            fingerprint[:16],
        )
        return FingerprintCheckResult(dq_reason=reason, fingerprint=fingerprint)

    superseded = find_registry_supersession(
        registry, fingerprint, com.hotkey, com.commit_block
    )
    register_fingerprint(
        registry,
        fingerprint,
        hotkey=com.hotkey,
        commit_block=com.commit_block,
        uid=com.uid,
        image=com.image,
        digest=com.digest,
    )
    if state_dir:
        save_fingerprint_registry(state_dir, registry)

    if superseded is not None:
        logger.info(
            "UID %d (%s) supersedes earlier-registered duplicate UID %s "
            "(commit_block %s -> %s, fingerprint=%s)",
            com.uid,
            com.hotkey[:16],
            superseded.get("uid"),
            superseded.get("commit_block"),
            com.commit_block,
            fingerprint[:16],
        )

    return FingerprintCheckResult(
        fingerprint=fingerprint,
        superseded_owner=superseded,
    )
