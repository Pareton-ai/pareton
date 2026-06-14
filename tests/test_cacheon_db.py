"""Unit tests for cacheon_db mirror helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cacheon_db import config as db_config
from cacheon_db.mirror import sync_evaluation, sync_validator_state
from validator.state import EvaluationRecord, ValidatorState, WinnerRecord

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _skip_db(monkeypatch):
    monkeypatch.setenv("CACHEON_SKIP_DB", "1")
    monkeypatch.delenv("CACHEON_DATABASE_URL", raising=False)
    db_config.SKIP_DB = True
    db_config.DATABASE_URL = ""


def test_sync_evaluation_noop_when_disabled():
    ev = EvaluationRecord(
        uid=1,
        hotkey="hk1",
        commit_block=100,
        image="img",
        digest="sha256:abc",
        score=0.1,
        speed_improvement=0.1,
        token_match_rate=0.9,
        disqualified=False,
        disqualify_reason=None,
        evaluated_at=1.0,
        evaluation_block=200,
    )
    with patch("cacheon_db.connection.db_connection") as mock_conn:
        sync_evaluation(ev)
        mock_conn.assert_not_called()


def test_sync_validator_state_noop_when_disabled():
    state = ValidatorState()
    with patch("cacheon_db.connection.db_connection") as mock_conn:
        sync_validator_state(state)
        mock_conn.assert_not_called()


@patch("cacheon_db.connection.enabled", return_value=True)
@patch("cacheon_db.connection.db_connection")
def test_sync_evaluation_executes_sql(mock_conn, _enabled):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    ev = EvaluationRecord(
        uid=2,
        hotkey="hk2",
        commit_block=101,
        image="img2",
        digest="sha256:def",
        score=0.2,
        speed_improvement=0.2,
        token_match_rate=0.95,
        disqualified=False,
        disqualify_reason=None,
        evaluated_at=2.0,
        evaluation_block=201,
    )
    sync_evaluation(ev)
    assert cursor.execute.call_count == 2
    sql = cursor.execute.call_args_list[0].args[0]
    assert "INSERT INTO evaluations" in sql


@patch("cacheon_db.connection.enabled", return_value=True)
@patch("cacheon_db.connection.db_connection")
def test_sync_validator_state_writes_meta_and_leader(mock_conn, _enabled):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    state = ValidatorState(
        winner=WinnerRecord(
            uid=3,
            hotkey="hk3",
            commit_block=50,
            image="img3",
            digest="sha256:ghi",
            score=0.3,
            speed_improvement=0.3,
            token_match_rate=0.99,
            evaluated_at=3.0,
            evaluation_block=300,
            won_at_block=300,
        ),
        last_scan_block=400,
        last_weights_set_block=401,
    )
    sync_validator_state(state)
    assert cursor.execute.call_count >= 2
    sqls = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("validator_meta" in sql for sql in sqls)


@patch("cacheon_db.connection.enabled", return_value=True)
@patch("cacheon_db.connection.db_connection")
def test_sync_eval_job_deletes_pending_first(mock_conn, _enabled):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    job = MagicMock()
    job.block = 500
    job.block_hash = "0xabc"
    job.challengers = []
    job.leader = None
    job.runner_up = None
    job.created_at = 1.0

    from cacheon_db.mirror import sync_eval_job

    sync_eval_job(job)
    assert cursor.execute.call_count == 2
    delete_sql = cursor.execute.call_args_list[0].args[0]
    assert "DELETE FROM eval_jobs" in delete_sql


@patch("cacheon_db.connection.enabled", return_value=True)
@patch("cacheon_db.connection.db_connection")
def test_sync_validator_state_reconciles_precheck(mock_conn, _enabled):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    state = ValidatorState(precheck_failures={"hk1:100": "unpaid"})
    sync_validator_state(state)
    sqls = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("DELETE FROM precheck_failures" in sql for sql in sqls)


@patch("cacheon_db.connection.enabled", return_value=True)
@patch("cacheon_db.connection.db_connection")
def test_sync_evaluation_clears_precheck(mock_conn, _enabled):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    mock_conn.return_value.__enter__.return_value = conn

    ev = EvaluationRecord(
        uid=1,
        hotkey="hk1",
        commit_block=100,
        image="img",
        digest="sha256:abc",
        score=0.1,
        speed_improvement=0.1,
        token_match_rate=0.9,
        disqualified=False,
        disqualify_reason=None,
        evaluated_at=1.0,
        evaluation_block=200,
    )
    sync_evaluation(ev)
    assert cursor.execute.call_count == 2
    delete_sql = cursor.execute.call_args_list[1].args[0]
    assert "DELETE FROM precheck_failures" in delete_sql
