"""GET /api/leader -- current leader record.
GET /api/leader/history -- overtake timeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import STATE_DIR
from api.helpers.state_reader import safe_json_load, safe_jsonl_load, sanitize_floats

router = APIRouter()

# Read legacy winner/king files until state-mainnet is renamed on S3 + VPS.
_HISTORY_FILES = (
    "leader-history.jsonl",
    "winner-history.jsonl",
    "king-history.jsonl",
)


@router.get(
    "/api/leader",
    tags=["Leader"],
    summary="Current leader",
    description="Full record of the reigning champion: UID, score, image, per-prompt stats.",
)
def get_leader():
    state = safe_json_load(STATE_DIR / "state.json", {})
    winner = state.get("winner") or state.get("king")
    if winner is None:
        return JSONResponse(
            content={"leader": None, "runner_up": None, "message": "No leader yet"},
            headers={"Cache-Control": "public, max-age=30"},
        )
    if "crowned_at_block" in winner and "won_at_block" not in winner:
        winner = {**winner, "won_at_block": winner["crowned_at_block"]}
    runner_up = state.get("runner_up")
    return JSONResponse(
        content=sanitize_floats({"leader": winner, "runner_up": runner_up}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/leader/history",
    tags=["Leader"],
    summary="Overtake history",
    description="Chronological list of leader changes. Each entry shows the new leader, the previous leader, and the margin.",
)
def get_leader_history():
    entries = _load_leader_history_entries(STATE_DIR)
    normalized = [_normalize_history_entry(e) for e in entries]
    return JSONResponse(
        content=sanitize_floats({"history": normalized, "total": len(normalized)}),
        headers={"Cache-Control": "public, max-age=30"},
    )


def _load_leader_history_entries(state_dir: Path) -> list[dict]:
    """Merge leader/winner/king history files, dedupe, oldest-first by ts."""
    entries: list[dict] = []
    seen: set[tuple] = set()
    for name in _HISTORY_FILES:
        path = state_dir / name
        if not path.is_file():
            continue
        for entry in safe_jsonl_load(path):
            key = _history_dedupe_key(entry)
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    entries.sort(key=lambda e: e.get("ts") or 0)
    return entries


def _history_dedupe_key(entry: dict) -> tuple:
    uid = _optional_field(
        entry,
        "new_leader_uid",
        "new_winner_uid",
        "new_king_uid",
    )
    return (entry.get("block"), uid)


def _optional_field(entry: dict, *names: str):
    for name in names:
        if name in entry and entry[name] is not None:
            return entry[name]
    return None


def _normalize_history_entry(e: dict) -> dict:
    """Translate on-disk leader/winner/king field names to API leader keys."""
    out: dict = {
        "ts": e.get("ts"),
        "block": e.get("block"),
        "new_leader_uid": _optional_field(
            e, "new_leader_uid", "new_winner_uid", "new_king_uid"
        ),
        "new_leader_hotkey": (
            e.get("new_leader_hotkey")
            or e.get("new_winner_hotkey")
            or e.get("new_king_hotkey")
        ),
        "new_leader_score": _optional_field(
            e, "new_leader_score", "new_winner_score", "new_king_score"
        ),
        "new_leader_image": (
            e.get("new_leader_image")
            or e.get("new_winner_image")
            or e.get("new_king_image")
        ),
        "new_leader_digest": (
            e.get("new_leader_digest")
            or e.get("new_winner_digest")
            or e.get("new_king_digest")
        ),
        "overtake_threshold": _optional_field(
            e, "overtake_threshold", "dethrone_threshold"
        ),
    }
    prev_uid = _optional_field(e, "prev_leader_uid", "prev_winner_uid", "prev_king_uid")
    if prev_uid is not None:
        out["prev_leader_uid"] = prev_uid
        out["prev_leader_hotkey"] = (
            e.get("prev_leader_hotkey")
            or e.get("prev_winner_hotkey")
            or e.get("prev_king_hotkey")
        )
        out["prev_leader_score"] = _optional_field(
            e, "prev_leader_score", "prev_winner_score", "prev_king_score"
        )
    return out
