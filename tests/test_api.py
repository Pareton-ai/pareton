"""API contract tests (no live DB)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from campaign.store import KNOWN_CAMPAIGN_STATUSES, KNOWN_SUBMISSION_STATES


@pytest.fixture()
def client(monkeypatch):
    from api import server

    monkeypatch.setattr(
        server,
        "list_campaigns",
        lambda status=None: [],
    )
    return TestClient(server.app)


def test_health_no_store(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == "no-store"


def test_campaigns_cache_control(client: TestClient):
    resp = client.get("/v1/campaigns")
    assert resp.status_code == 200
    assert (
        resp.headers.get("Cache-Control")
        == "public, max-age=30, stale-while-revalidate=300"
    )


def test_submissions_pagination_envelope(monkeypatch, client: TestClient):
    from api import server

    campaign = SimpleNamespace(status="open", to_public_dict=lambda: {"id": "c1"})
    monkeypatch.setattr(server, "get_campaign", lambda _cid: campaign)
    monkeypatch.setattr(
        server,
        "list_submissions",
        lambda _cid, *, limit=50, offset=0: {
            "total": 120,
            "items": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "campaign_id": "c1",
                    "patch_hash": "abc",
                    "hotkey": "hk",
                    "baseline_commit": "deadbeef",
                    "retrieval_url": "https://example/p.diff",
                    "commit_block": 1,
                    "committed_at": "2026-07-22T00:00:00+00:00",
                    "engine_image_ref": None,
                }
            ],
        },
    )
    monkeypatch.setattr(
        server,
        "list_bench_summaries",
        lambda _cid, submission_ids=None: {
            "11111111-1111-1111-1111-111111111111": "pass"
        },
    )
    monkeypatch.setattr(
        server,
        "list_latest_states",
        lambda ids: {str(ids[0]): "benched"},
    )

    resp = client.get("/v1/campaigns/c1/submissions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 120
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["submissions"]) == 1
    row = body["submissions"][0]
    assert row["latest_state"] == "benched"
    assert row["bench_verdict"] == "pass"
    assert (
        resp.headers.get("Cache-Control")
        == "public, max-age=30, stale-while-revalidate=300"
    )


def test_submissions_offset_past_end(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(
        server,
        "get_campaign",
        lambda _cid: SimpleNamespace(status="open"),
    )
    monkeypatch.setattr(
        server,
        "list_submissions",
        lambda _cid, *, limit=50, offset=0: {"total": 3, "items": []},
    )
    monkeypatch.setattr(server, "list_bench_summaries", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "list_latest_states", lambda _ids: {})

    resp = client.get("/v1/campaigns/c1/submissions?limit=50&offset=100")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["submissions"] == []


@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=201", "offset=-1"],
)
def test_submissions_bad_params_422(monkeypatch, client: TestClient, query: str):
    from api import server

    monkeypatch.setattr(
        server,
        "get_campaign",
        lambda _cid: SimpleNamespace(status="open"),
    )
    resp = client.get(f"/v1/campaigns/c1/submissions?{query}")
    assert resp.status_code == 422


def test_submissions_campaign_404(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(server, "get_campaign", lambda _cid: None)
    resp = client.get("/v1/campaigns/missing/submissions")
    assert resp.status_code == 404


def test_stats_shape(monkeypatch, client: TestClient):
    from api import server

    payload = {
        "campaigns": {
            "total": 0,
            "by_status": {s: 0 for s in KNOWN_CAMPAIGN_STATUSES},
        },
        "submissions": {
            "total": 0,
            "by_latest_state": {s: 0 for s in KNOWN_SUBMISSION_STATES},
        },
    }
    monkeypatch.setattr(server, "get_public_stats", lambda: payload)
    resp = client.get("/v1/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["campaigns"]["total"] == 0
    assert set(body["campaigns"]["by_status"]) == set(KNOWN_CAMPAIGN_STATUSES)
    assert set(body["submissions"]["by_latest_state"]) == set(KNOWN_SUBMISSION_STATES)
    assert (
        resp.headers.get("Cache-Control")
        == "public, max-age=30, stale-while-revalidate=300"
    )
