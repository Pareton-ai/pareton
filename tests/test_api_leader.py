"""API integration tests for Postgres-backed leader routes."""

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


@pytest.mark.skipif(not _has_api_deps, reason="API dependencies not installed")
class TestLeaderAPI:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("CACHEON_DATABASE_URL", "postgresql://test/db")
        db_config.DATABASE_URL = "postgresql://test/db"
        self.client = TestClient(_app)

    @patch("api.routes.leader.get_leader_state")
    def test_leader_with_runner_up(self, mock_state):
        mock_state.return_value = (
            {
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
        )
        resp = self.client.get("/api/leader")
        body = resp.json()
        assert body["leader"]["uid"] == 1
        assert body["runner_up"]["uid"] == 2
