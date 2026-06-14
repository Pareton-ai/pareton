"""Unit tests for cacheon_db.readers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cacheon_db import config as db_config
from cacheon_db.readers import (
    count_evaluations,
    get_eval_progress,
    get_leader_history,
    get_leader_state,
    get_pending_eval_job,
    get_validator_meta,
    list_evaluations,
    list_rounds,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _db_url(monkeypatch):
    monkeypatch.setenv("CACHEON_DATABASE_URL", "postgresql://test/db")
    db_config.DATABASE_URL = "postgresql://test/db"


def _mock_read_conn(rows):
    cursor = MagicMock()
    if rows is None:
        cursor.fetchone.return_value = None
        cursor.fetchall.return_value = []
    elif isinstance(rows, dict):
        cursor.fetchone.return_value = rows
        cursor.fetchall.return_value = [rows]
    else:
        cursor.fetchone.return_value = rows[0] if rows else None
        cursor.fetchall.return_value = rows
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


@patch("cacheon_db.readers.read_db_connection")
def test_get_validator_meta_defaults(mock_conn):
    mock_conn.return_value.__enter__.return_value = _mock_read_conn(None)
    meta = get_validator_meta()
    assert meta["last_scan_block"] == 0
    assert meta["schema_version"] == 1


@patch("cacheon_db.readers.read_db_connection")
def test_get_leader_state_maps_roles(mock_conn):
    rows = [
        {
            "role": "leader",
            "uid": 1,
            "hotkey": "hk1",
            "commit_block": 10,
            "image": "img1",
            "digest": "sha256:a",
            "score": 0.1,
            "speed_improvement": 0.1,
            "token_match_rate": 0.9,
            "evaluated_at": 1.0,
            "evaluation_block": 100,
            "won_at_block": 100,
        },
        {
            "role": "runner_up",
            "uid": 2,
            "hotkey": "hk2",
            "commit_block": 11,
            "image": "img2",
            "digest": "sha256:b",
            "score": 0.05,
            "speed_improvement": 0.05,
            "token_match_rate": 0.95,
            "evaluated_at": 2.0,
            "evaluation_block": 100,
            "won_at_block": 0,
        },
    ]
    mock_conn.return_value.__enter__.return_value = _mock_read_conn(rows)
    leader, runner_up = get_leader_state()
    assert leader["uid"] == 1
    assert runner_up["uid"] == 2
    assert leader["won_at_block"] == 100


@patch("cacheon_db.readers.read_db_connection")
def test_get_leader_history_omits_null_prev(mock_conn):
    rows = [
        {
            "ts": 1.0,
            "block": 100,
            "new_leader_uid": 3,
            "new_leader_hotkey": "hk3",
            "new_leader_score": 0.2,
            "new_leader_image": "img3",
            "new_leader_digest": "sha256:c",
            "overtake_threshold": 0.01,
            "prev_leader_uid": None,
            "prev_leader_hotkey": None,
            "prev_leader_score": None,
        }
    ]
    mock_conn.return_value.__enter__.return_value = _mock_read_conn(rows)
    history = get_leader_history()
    assert len(history) == 1
    assert "prev_leader_uid" not in history[0]


@patch("cacheon_db.readers.read_db_connection")
def test_list_evaluations_active_filter(mock_conn):
    conn = _mock_read_conn([])
    mock_conn.return_value.__enter__.return_value = conn
    list_evaluations(status="active")
    cursor = conn.cursor.return_value
    sql = cursor.execute.call_args[0][0]
    assert "disqualified = FALSE" in sql


@patch("cacheon_db.readers.read_db_connection")
def test_count_evaluations(mock_conn):
    mock_conn.return_value.__enter__.return_value = _mock_read_conn(
        {"total": 5, "active": 4, "dq": 1, "last_eval_ts": 123.0}
    )
    total, active, dq, last_ts = count_evaluations()
    assert total == 5
    assert active == 4
    assert dq == 1
    assert last_ts == 123.0


@patch("cacheon_db.readers.list_evaluations")
def test_list_rounds_groups_by_block(mock_list):
    mock_list.return_value = [
        {
            "uid": 1,
            "hotkey": "hk1",
            "image": "img1",
            "commit_block": 10,
            "score": 0.1,
            "speed_improvement": 0.1,
            "token_match_rate": 0.9,
            "disqualified": False,
            "disqualify_reason": None,
            "evaluated_at": 200.0,
            "evaluation_block": 100,
        },
        {
            "uid": 2,
            "hotkey": "hk2",
            "image": "img2",
            "commit_block": 11,
            "score": 0.2,
            "speed_improvement": 0.2,
            "token_match_rate": 0.95,
            "disqualified": True,
            "disqualify_reason": "timeout",
            "evaluated_at": 150.0,
            "evaluation_block": 100,
        },
    ]
    rounds = list_rounds()
    assert len(rounds) == 1
    assert rounds[0]["evaluation_block"] == 100
    assert rounds[0]["n_challengers"] == 2
    assert rounds[0]["evaluated_at"] == 200.0


@patch("cacheon_db.readers.read_db_connection")
def test_get_pending_eval_job(mock_conn):
    row = {
        "block": 500,
        "block_hash": "0xabc",
        "challengers": [{"uid": 1}],
        "leader": {"uid": 9},
        "runner_up": None,
        "created_at": 1.0,
    }
    mock_conn.return_value.__enter__.return_value = _mock_read_conn(row)
    job = get_pending_eval_job()
    assert job["block"] == 500
    assert job["challengers"] == [{"uid": 1}]


@patch("cacheon_db.readers.read_db_connection")
def test_get_eval_progress_idle_row(mock_conn):
    mock_conn.return_value.__enter__.return_value = _mock_read_conn(
        {"status": "idle", "phase": None, "updated_at": 1.0}
    )
    assert get_eval_progress() is None
