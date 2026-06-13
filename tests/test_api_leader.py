"""Unit tests for api.routes.leader history loading and normalization."""

from __future__ import annotations

import json

import pytest

from api.routes.leader import (
    _load_leader_history_entries,
    _normalize_history_entry,
)

pytestmark = pytest.mark.unit


def test_normalize_new_leader_fields():
    entry = {
        "ts": 1.0,
        "block": 100,
        "new_leader_uid": 3,
        "new_leader_hotkey": "hk3",
        "new_leader_score": 0.12,
        "new_leader_image": "img3",
        "new_leader_digest": "sha256:abc",
        "overtake_threshold": 0.01,
        "prev_leader_uid": 1,
        "prev_leader_hotkey": "hk1",
        "prev_leader_score": 0.10,
    }
    out = _normalize_history_entry(entry)
    assert out["new_leader_uid"] == 3
    assert out["new_leader_hotkey"] == "hk3"
    assert out["prev_leader_uid"] == 1


def test_normalize_legacy_winner_fields():
    entry = {
        "ts": 2.0,
        "block": 200,
        "new_winner_uid": 5,
        "new_winner_hotkey": "hk5",
        "new_winner_score": 0.20,
        "new_winner_image": "img5",
        "new_winner_digest": "sha256:def",
        "overtake_threshold": 0.02,
        "prev_winner_uid": 3,
        "prev_winner_hotkey": "hk3",
        "prev_winner_score": 0.15,
    }
    out = _normalize_history_entry(entry)
    assert out["new_leader_uid"] == 5
    assert out["new_leader_hotkey"] == "hk5"
    assert out["prev_leader_uid"] == 3


def test_merge_history_files_dedupes_and_keeps_legacy(tmp_path):
    leader = tmp_path / "leader-history.jsonl"
    winner = tmp_path / "winner-history.jsonl"
    leader.write_text(
        json.dumps(
            {
                "ts": 3.0,
                "block": 300,
                "new_leader_uid": 9,
                "new_leader_hotkey": "hk9",
                "new_leader_score": 0.3,
                "new_leader_image": "img9",
                "new_leader_digest": "sha256:new",
                "overtake_threshold": 0.0,
            }
        )
        + "\n"
    )
    winner.write_text(
        json.dumps(
            {
                "ts": 1.0,
                "block": 100,
                "new_winner_uid": 1,
                "new_winner_hotkey": "hk1",
                "new_winner_score": 0.1,
                "new_winner_image": "img1",
                "new_winner_digest": "sha256:old",
                "overtake_threshold": 0.0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "ts": 3.0,
                "block": 300,
                "new_winner_uid": 9,
                "new_winner_hotkey": "hk9",
                "new_winner_score": 0.3,
                "new_winner_image": "img9",
                "new_winner_digest": "sha256:dup",
                "overtake_threshold": 0.0,
            }
        )
        + "\n"
    )

    entries = _load_leader_history_entries(tmp_path)
    assert len(entries) == 2
    assert entries[0]["block"] == 100
    assert entries[1]["block"] == 300
    assert entries[1]["new_leader_uid"] == 9
