"""Unit tests for validator.eval_progress and api/routes/eval_progress."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from api.helpers.eval_progress import progress_expired
from validator.eval_progress import (
    PROGRESS_FILE,
    clear_progress,
    complete_progress,
    update_challenger_status,
    update_incumbent_status,
    update_progress,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read(state_dir: Path) -> dict:
    path = state_dir / PROGRESS_FILE
    with open(path) as f:
        return json.load(f)


SAMPLE_CHALLENGERS = [
    {"uid": 10, "hotkey": "5Abc", "image": "docker.io/a:v1"},
    {"uid": 20, "hotkey": "5Def", "image": "docker.io/b:v2"},
]


# --------------------------------------------------------------------------- #
# update_progress
# --------------------------------------------------------------------------- #


def test_read_progress_falls_back_to_postgres(tmp_path, monkeypatch):
    db_payload = {
        "status": "running",
        "phase": "gpu_searching",
        "challengers": [{"idx": 0, "uid": 1, "status": "pending"}],
        "updated_at": 1.0,
    }
    monkeypatch.setattr(
        "cacheon_db.loaders.load_eval_progress_payload",
        lambda: db_payload,
    )
    from validator.eval_progress import _read_progress

    assert _read_progress(tmp_path) == db_payload


def test_update_progress_creates_file(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
    )
    data = _read(tmp_path)
    assert data["phase"] == "challengers_found"
    assert data["round_block"] == 100
    assert data["status"] == "running"
    assert len(data["challengers"]) == 2
    assert all(c["status"] == "pending" for c in data["challengers"])
    assert data["challengers"][0]["uid"] == 10
    assert data["challengers"][1]["idx"] == 1
    assert len(data["steps"]) == 1
    assert data["steps"][0]["phase"] == "challengers_found"
    assert data["current_idx"] is None
    assert data["started_at"] > 0
    assert data["updated_at"] > 0


def test_update_progress_appends_steps(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_progress(tmp_path, phase="gpu_searching")
    update_progress(tmp_path, phase="gpu_ready", pod_id="wrk-123")
    data = _read(tmp_path)
    assert data["phase"] == "gpu_ready"
    assert len(data["steps"]) == 3
    assert data["steps"][2]["pod_id"] == "wrk-123"
    assert data["challengers"][0]["status"] == "pending"


def test_challengers_found_resets_steps(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_progress(tmp_path, phase="gpu_searching")
    assert len(_read(tmp_path)["steps"]) == 2

    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=200,
        challengers=[{"uid": 99, "hotkey": "5Xyz", "image": "img:v1"}],
    )
    data = _read(tmp_path)
    assert data["round_block"] == 200
    assert len(data["steps"]) == 1
    assert len(data["challengers"]) == 1


def test_update_progress_preserves_gpu(tmp_path):
    gpu = {"provider": "targon", "pod_id": "wrk-1", "cost_per_hr": 10.0}
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_progress(tmp_path, phase="gpu_ready", gpu=gpu)
    update_progress(tmp_path, phase="gpu_setup")
    data = _read(tmp_path)
    assert data["gpu"]["pod_id"] == "wrk-1"


def test_update_progress_resets_complete_status(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=100,
        challengers=SAMPLE_CHALLENGERS,
    )
    complete_progress(tmp_path)
    assert _read(tmp_path)["status"] == "complete"

    update_progress(tmp_path, phase="gpu_searching")
    data = _read(tmp_path)
    assert data["status"] == "running"
    assert data["phase"] == "gpu_searching"
    assert "completed_at" not in data


# --------------------------------------------------------------------------- #
# update_challenger_status
# --------------------------------------------------------------------------- #


def test_update_challenger_status(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_challenger_status(tmp_path, 0, status="pulling", detail="pulling_image")
    data = _read(tmp_path)
    assert data["challengers"][0]["status"] == "pulling"
    assert data["challengers"][1]["status"] == "pending"
    assert data["current_idx"] == 0
    assert data["phase"] == "challenger_eval"

    update_challenger_status(
        tmp_path, 0, status="dq", score=0.0, dq_reason="pull_timeout"
    )
    data = _read(tmp_path)
    assert data["challengers"][0]["status"] == "dq"
    assert data["challengers"][0]["score"] == 0.0
    assert data["challengers"][0]["dq_reason"] == "pull_timeout"


def test_update_challenger_status_scored(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_challenger_status(tmp_path, 1, status="scored", score=0.85)
    data = _read(tmp_path)
    assert data["challengers"][1]["status"] == "scored"
    assert data["challengers"][1]["score"] == 0.85


# --------------------------------------------------------------------------- #
# update_incumbent_status
# --------------------------------------------------------------------------- #


def test_update_incumbent_status(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
        leader={"uid": 1, "hotkey": "5Lead", "image": "docker.io/l:v1"},
        runner_up={"uid": 2, "hotkey": "5Run", "image": "docker.io/r:v1"},
    )
    data = _read(tmp_path)
    assert data["leader"]["status"] == "pending"
    assert data["runner_up"]["status"] == "pending"

    update_incumbent_status(tmp_path, "leader", status="evaluating")
    data = _read(tmp_path)
    assert data["leader"]["status"] == "evaluating"
    assert data["runner_up"]["status"] == "pending"

    update_incumbent_status(tmp_path, "leader", status="scored", score=0.0188)
    data = _read(tmp_path)
    assert data["leader"]["status"] == "scored"
    assert data["leader"]["score"] == 0.0188

    update_incumbent_status(
        tmp_path, "runner_up", status="dq", score=0.0, dq_reason="pull_timeout"
    )
    data = _read(tmp_path)
    assert data["runner_up"]["status"] == "dq"
    assert data["runner_up"]["dq_reason"] == "pull_timeout"

    update_incumbent_status(
        tmp_path,
        "leader",
        status="skipped",
        detail="scoring_baseline_unavailable",
    )
    data = _read(tmp_path)
    assert data["leader"]["status"] == "skipped"
    assert data["detail"] == "scoring_baseline_unavailable"


def test_update_incumbent_status_no_file(tmp_path):
    update_incumbent_status(tmp_path, "leader", status="evaluating")
    assert not (tmp_path / PROGRESS_FILE).exists()


def test_update_incumbent_status_bad_role(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
        leader={"uid": 1, "hotkey": "5Lead", "image": "docker.io/l:v1"},
    )
    update_incumbent_status(tmp_path, "winner", status="scored", score=1.0)
    data = _read(tmp_path)
    assert "score" not in data["leader"]


# --------------------------------------------------------------------------- #
# complete_progress
# --------------------------------------------------------------------------- #


def test_complete_progress_sets_status(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    update_challenger_status(tmp_path, 0, status="scored", score=0.05)
    complete_progress(tmp_path)
    data = _read(tmp_path)
    assert data["status"] == "complete"
    assert data["phase"] == "eval_complete"
    assert data["completed_at"] > 0
    assert data["challengers"][0]["score"] == 0.05


def test_progress_expired_only_for_complete(tmp_path):
    update_progress(
        tmp_path,
        phase="gpu_searching",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    assert progress_expired(_read(tmp_path)) is False

    complete_progress(tmp_path)
    data = _read(tmp_path)
    assert progress_expired(data) is False

    data["completed_at"] = time.time() - 3600
    assert progress_expired(data) is True


# --------------------------------------------------------------------------- #
# clear_progress
# --------------------------------------------------------------------------- #


def test_clear_progress_removes_file(tmp_path):
    update_progress(
        tmp_path,
        phase="challengers_found",
        round_block=1,
        challengers=SAMPLE_CHALLENGERS,
    )
    assert (tmp_path / PROGRESS_FILE).exists()
    clear_progress(tmp_path)
    assert not (tmp_path / PROGRESS_FILE).exists()


# --------------------------------------------------------------------------- #
# API endpoint
# --------------------------------------------------------------------------- #


try:
    from api.server import app as _app  # noqa: F401

    _has_api_deps = True
except ImportError:
    _has_api_deps = False


@pytest.mark.skipif(not _has_api_deps, reason="API dependencies not installed")
class TestEvalProgressAPI:
    """Tests for GET /api/eval-progress using mocked Postgres readers."""

    @pytest.fixture(autouse=True)
    def _client(self, monkeypatch):
        monkeypatch.setenv("CACHEON_DATABASE_URL", "postgresql://test/db")
        from cacheon_db import config as db_config

        db_config.DATABASE_URL = "postgresql://test/db"
        from api.server import app

        from starlette.testclient import TestClient

        self.client = TestClient(app)

    @patch("api.routes.eval_progress.get_eval_progress")
    def test_idle_when_no_progress(self, mock_get):
        mock_get.return_value = None
        resp = self.client.get("/api/eval-progress")
        assert resp.status_code == 200
        assert resp.json() == {"status": "idle"}

    @patch("api.routes.eval_progress.get_eval_progress")
    def test_returns_data_when_running(self, mock_get):
        mock_get.return_value = {
            "status": "running",
            "phase": "challengers_found",
            "round_block": 100,
            "challengers": SAMPLE_CHALLENGERS,
            "updated_at": time.time(),
        }
        resp = self.client.get("/api/eval-progress")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["phase"] == "challengers_found"
        assert len(body["challengers"]) == 2

    @patch("api.routes.eval_progress.get_eval_progress")
    def test_stale_flag(self, mock_get):
        mock_get.return_value = {
            "status": "running",
            "phase": "challengers_found",
            "round_block": 100,
            "challengers": SAMPLE_CHALLENGERS,
            "updated_at": time.time() - 3600,
        }
        resp = self.client.get("/api/eval-progress")
        body = resp.json()
        assert body.get("possibly_stale") is True

    @patch("api.routes.eval_progress.get_eval_progress")
    def test_complete_status(self, mock_get):
        mock_get.return_value = {
            "status": "complete",
            "phase": "eval_complete",
            "completed_at": time.time(),
            "updated_at": time.time(),
            "challengers": SAMPLE_CHALLENGERS,
        }
        resp = self.client.get("/api/eval-progress")
        body = resp.json()
        assert body["status"] == "complete"
        assert body["phase"] == "eval_complete"

    @patch("api.routes.eval_progress.get_eval_progress")
    def test_idle_when_reader_returns_none_for_expired(self, mock_get):
        mock_get.return_value = None
        resp = self.client.get("/api/eval-progress")
        assert resp.json() == {"status": "idle"}
