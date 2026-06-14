"""Unit tests for validator.eval_progress and api/routes/eval_progress."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from api.helpers.eval_progress import progress_expired
from validator.eval_progress import (
    clear_progress,
    complete_progress,
    seed_progress_from_eval_job,
    update_challenger_status,
    update_incumbent_status,
    update_progress,
)

pytestmark = pytest.mark.unit

SAMPLE_CHALLENGERS = [
    {"uid": 10, "hotkey": "5Abc", "image": "docker.io/a:v1"},
    {"uid": 20, "hotkey": "5Def", "image": "docker.io/b:v2"},
]


@pytest.fixture
def progress_store(monkeypatch):
    store: dict = {"payload": None}

    def _load():
        return store["payload"]

    def _sync(payload):
        store["payload"] = dict(payload)

    def _clear():
        store["payload"] = None

    monkeypatch.setattr("cacheon_db.loaders.load_eval_progress_payload", _load)
    monkeypatch.setattr("cacheon_db.sync_eval_progress", _sync)
    monkeypatch.setattr("cacheon_db.clear_eval_progress", _clear)
    return store


def test_update_progress_challengers_found(progress_store):
    update_progress(
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
    )
    payload = progress_store["payload"]
    assert payload["status"] == "running"
    assert payload["phase"] == "challengers_found"
    assert len(payload["challengers"]) == 2
    assert payload["challengers"][0]["status"] == "pending"


def test_update_progress_phase_transition(progress_store):
    update_progress(
        phase="challengers_found", round_block=100, challengers=SAMPLE_CHALLENGERS
    )
    update_progress(phase="gpu_searching")
    update_progress(phase="gpu_ready", pod_id="wrk-123")
    payload = progress_store["payload"]
    assert payload["phase"] == "gpu_ready"
    assert any(step["phase"] == "gpu_ready" for step in payload["steps"])


def test_update_challenger_status(progress_store):
    update_progress(
        phase="challengers_found", round_block=100, challengers=SAMPLE_CHALLENGERS
    )
    update_challenger_status(0, status="pulling", detail="pulling_image")
    payload = progress_store["payload"]
    assert payload["challengers"][0]["status"] == "pulling"
    assert payload["phase"] == "challenger_eval"


def test_update_incumbent_status(progress_store):
    update_progress(
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
        leader={"uid": 1, "hotkey": "hk1", "image": "img:v1"},
    )
    update_incumbent_status("leader", status="evaluating")
    payload = progress_store["payload"]
    assert payload["leader"]["status"] == "evaluating"


def test_complete_and_clear(progress_store):
    update_progress(
        phase="challengers_found", round_block=100, challengers=SAMPLE_CHALLENGERS
    )
    complete_progress()
    assert progress_store["payload"]["status"] == "complete"
    clear_progress()
    assert progress_store["payload"] is None


def test_progress_expired():
    payload = {
        "status": "complete",
        "completed_at": time.time() - 100,
    }
    assert progress_expired(payload) is False

    old = {
        "status": "complete",
        "completed_at": time.time() - 2000,
    }
    with patch("api.helpers.eval_progress.COMPLETE_LINGER_S", 900):
        assert progress_expired(old) is True


def test_progress_running_not_expired():
    assert progress_expired({"status": "running"}) is False


def test_seed_progress_swallows_db_error(monkeypatch):
    from cacheon_db.connection import DatabaseUnavailable

    def _raise():
        raise DatabaseUnavailable("load failed")

    monkeypatch.setattr(
        "cacheon_db.loaders.load_eval_progress_payload",
        _raise,
    )

    class _Job:
        block = 100
        challengers = []
        leader = None
        runner_up = None

    seed_progress_from_eval_job(_Job())
