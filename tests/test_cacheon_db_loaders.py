"""Unit tests for cacheon_db.loaders."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cacheon_db import config as db_config
from cacheon_db.loaders import (
    load_eval_progress_payload,
    load_pending_eval_job_dict,
    load_validator_state_dict,
)


@pytest.fixture(autouse=True)
def _enable_db(monkeypatch):
    monkeypatch.setenv("CACHEON_DATABASE_URL", "postgresql://test/db")
    monkeypatch.setenv("CACHEON_SKIP_DB", "0")
    db_config.DATABASE_URL = "postgresql://test/db"
    db_config.SKIP_DB = False


def _mock_conn(rows_by_query: dict[str, list[dict]]):
    cursor = MagicMock()

    def execute(sql, params=None):
        for key, rows in rows_by_query.items():
            if key in sql:
                cursor.fetchone = MagicMock(
                    side_effect=([dict(r) for r in rows] + [None] * 10).__iter__
                    if "LIMIT 1" in sql
                    else None
                )
                cursor.fetchall = MagicMock(return_value=[dict(r) for r in rows])
                return
        cursor.fetchone = MagicMock(return_value=None)
        cursor.fetchall = MagicMock(return_value=[])

    cursor.execute.side_effect = execute
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn


@patch("cacheon_db.loaders.db_connection")
def test_load_eval_progress_payload_running(mock_conn):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = {
        "status": "running",
        "phase": "baseline_running",
        "challengers": [{"idx": 0, "status": "pending"}],
        "updated_at": 1.0,
    }
    conn.cursor.return_value.__enter__.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    payload = load_eval_progress_payload()
    assert payload is not None
    assert payload["status"] == "running"
    assert payload["phase"] == "baseline_running"


@patch("cacheon_db.loaders.db_connection")
def test_load_eval_progress_payload_idle_returns_none(mock_conn):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = {"status": "idle"}
    conn.cursor.return_value.__enter__.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    assert load_eval_progress_payload() is None


@patch("cacheon_db.loaders.db_connection")
def test_load_pending_eval_job_dict(mock_conn):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = {
        "block": 100,
        "block_hash": "0xabc",
        "challengers": [
            {"uid": 1, "hotkey": "hk", "commit_block": 1, "image": "i", "digest": "d"}
        ],
        "leader": None,
        "runner_up": None,
        "created_at": 1.0,
    }
    conn.cursor.return_value.__enter__.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    job = load_pending_eval_job_dict()
    assert job is not None
    assert job["block"] == 100
    assert len(job["challengers"]) == 1


@patch("cacheon_db.loaders.db_connection")
def test_load_validator_state_dict(mock_conn):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = {
        "last_scan_block": 10,
        "last_weights_set_block": 9,
        "schema_version": 1,
    }
    cursor.fetchall.side_effect = [
        [
            {
                "role": "leader",
                "uid": 1,
                "hotkey": "hk",
                "commit_block": 1,
                "image": "img",
                "digest": "dig",
                "score": 0.5,
                "speed_improvement": 0.5,
                "token_match_rate": 0.99,
                "evaluated_at": 1.0,
                "evaluation_block": 10,
                "won_at_block": 10,
            }
        ],
        [
            {
                "hotkey": "hk",
                "commit_block": 1,
                "uid": 1,
                "image": "img",
                "digest": "dig",
                "score": 0.5,
                "speed_improvement": 0.5,
                "token_match_rate": 0.99,
                "disqualified": False,
                "disqualify_reason": None,
                "evaluated_at": 1.0,
                "evaluation_block": 10,
                "per_prompt": None,
            }
        ],
        [],
    ]
    conn.cursor.return_value.__enter__.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    data = load_validator_state_dict()
    assert data is not None
    assert data["last_scan_block"] == 10
    assert data["winner"]["uid"] == 1
    assert "hk:1" in data["evaluations"]
