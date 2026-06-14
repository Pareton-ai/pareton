"""API integration tests for Postgres-backed structured endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cacheon_db import config as db_config

pytestmark = pytest.mark.unit

try:
    from api.server import app as _app  # noqa: F401
    from starlette.testclient import TestClient

    _has_api_deps = True
except ImportError:
    _has_api_deps = False


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CACHEON_DATABASE_URL", "postgresql://test/db")
    db_config.DATABASE_URL = "postgresql://test/db"
    return TestClient(_app)


@pytest.mark.skipif(not _has_api_deps, reason="API dependencies not installed")
class TestApiPostgresRoutes:
    @patch("api.routes.status.get_validator_meta")
    @patch("api.routes.status.get_leader_state")
    @patch("api.routes.status.count_evaluations")
    def test_status(self, mock_counts, mock_leader, mock_meta, client):
        mock_meta.return_value = {
            "last_scan_block": 1000,
            "last_weights_set_block": 999,
        }
        mock_leader.return_value = (
            {"uid": 5, "score": 0.12, "image": "img:v1"},
            None,
        )
        mock_counts.return_value = (10, 8, 2, 1700000000.0)
        resp = client.get("/api/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["leader_uid"] == 5
        assert body["n_evaluated"] == 10
        assert body["last_scan_block"] == 1000

    @patch("api.routes.leader.get_leader_state")
    def test_leader_no_leader(self, mock_state, client):
        mock_state.return_value = (None, None)
        resp = client.get("/api/leader")
        assert resp.status_code == 200
        assert resp.json()["message"] == "No leader yet"

    @patch("api.routes.leader.get_leader_history")
    def test_leader_history(self, mock_history, client):
        mock_history.return_value = [
            {
                "ts": 1.0,
                "block": 100,
                "new_leader_uid": 3,
                "new_leader_hotkey": "hk3",
                "new_leader_score": 0.1,
                "new_leader_image": "img",
                "new_leader_digest": "sha256:x",
                "overtake_threshold": 0.01,
            }
        ]
        resp = client.get("/api/leader/history")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    @patch("api.routes.evaluations.list_evaluations")
    def test_evaluations_list(self, mock_list, client):
        mock_list.return_value = [{"uid": 1, "hotkey": "hk1", "score": 0.1}]
        resp = client.get("/api/evaluations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    @patch("api.routes.rounds.list_rounds")
    def test_rounds(self, mock_rounds, client):
        mock_rounds.return_value = []
        resp = client.get("/api/rounds")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


@pytest.mark.skipif(not _has_api_deps, reason="API dependencies not installed")
class TestApiDatabaseRequired:
    def test_status_503_when_db_unconfigured(self, monkeypatch):
        monkeypatch.delenv("CACHEON_DATABASE_URL", raising=False)
        db_config.DATABASE_URL = ""
        client = TestClient(_app)
        resp = client.get("/api/status")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "CACHEON_DATABASE_URL is not configured"

    def test_leader_503_when_db_unconfigured(self, monkeypatch):
        monkeypatch.delenv("CACHEON_DATABASE_URL", raising=False)
        db_config.DATABASE_URL = ""
        client = TestClient(_app)
        resp = client.get("/api/leader")
        assert resp.status_code == 503

    def test_evaluations_503_when_db_unconfigured(self, monkeypatch):
        monkeypatch.delenv("CACHEON_DATABASE_URL", raising=False)
        db_config.DATABASE_URL = ""
        client = TestClient(_app)
        resp = client.get("/api/evaluations")
        assert resp.status_code == 503

    def test_health_ok_without_db(self, monkeypatch):
        monkeypatch.delenv("CACHEON_DATABASE_URL", raising=False)
        db_config.DATABASE_URL = ""
        client = TestClient(_app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
