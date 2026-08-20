"""API contract tests (no live DB)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from campaign.store import (
    KNOWN_CAMPAIGN_STATUSES,
    KNOWN_SUBMISSION_STATES,
    derive_bench_verdict_from_events,
)

V1_CACHE_CONTROL_EXPECTED = "public, max-age=30, stale-while-revalidate=300"


@pytest.fixture()
def client(monkeypatch):
    from api import server

    monkeypatch.setattr(
        server,
        "list_campaigns",
        lambda status=None: [],
    )
    return TestClient(server.app)


def test_duplicate_image_is_a_terminal_bench_verdict():
    """round/store.py rejects a duplicate image; the API must not call it pending."""
    events = [
        {"state": "bench_queued", "detail": {}},
        {"state": "rejected", "detail": {"reason": "duplicate_image"}},
    ]
    assert derive_bench_verdict_from_events(events) == "duplicate_image"


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


def test_campaign_detail_db_unavailable_is_503(monkeypatch, client: TestClient):
    from api import server
    from db.exceptions import DatabaseUnavailable

    def boom(_cid):
        raise DatabaseUnavailable("database connection failed")

    monkeypatch.setattr(server, "get_campaign", boom)
    resp = client.get("/v1/campaigns/c1")
    assert resp.status_code == 503


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
        lambda ids: {str(ids[0]): "scored"},
    )

    resp = client.get("/v1/campaigns/c1/submissions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 120
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["submissions"]) == 1
    row = body["submissions"][0]
    assert row["latest_state"] == "scored"
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


def test_openapi_publishes_the_submission_state_vocabulary():
    """The generated frontend union derives from this enum; keep them bound.

    If this fails, either a model stopped referencing SubmissionState or the
    class was renamed, and pareton-frontend's `types.ts` import will break at
    the next `schema.d.ts` regeneration (PAR-46).
    """
    from api.server import app
    from gate.types import SubmissionState

    schema = app.openapi()["components"]["schemas"]["SubmissionState"]
    assert schema["type"] == "string"
    assert schema["enum"] == [s.value for s in SubmissionState]


def test_openapi_publishes_the_bench_phase_vocabulary():
    """BenchPhase is a named OpenAPI enum."""
    from api.server import app
    from bench.phases import BenchPhase

    schema = app.openapi()["components"]["schemas"]["BenchPhase"]
    assert schema["type"] == "string"
    assert schema["enum"] == [p.value for p in BenchPhase]


def test_detail_exposes_live_phase_of_a_running_job(monkeypatch, client: TestClient):
    from api import server

    sid = "77777777-7777-7777-7777-777777777777"
    monkeypatch.setattr(
        server,
        "get_submission_for_campaign",
        lambda _c, _h: _detail_row(sid, "c1", "sha256:live"),
    )
    monkeypatch.setattr(server, "list_events", lambda _id: [])
    monkeypatch.setattr(
        server, "list_latest_states", lambda _ids: {sid: "bench_queued"}
    )
    monkeypatch.setattr(
        server,
        "list_submission_jobs",
        lambda _id: [
            {
                "status": "running",
                "last_error": None,
                "phase": "downloading_model",
                "phase_started_at": datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
                "heartbeat_at": datetime(2026, 8, 17, 12, 4, tzinfo=timezone.utc),
                "progress": {"gpu_sku": "H200-SXM-141GB"},
            }
        ],
    )

    body = client.get("/v1/campaigns/c1/submissions/sha256:live").json()
    job = body["jobs"][0]
    assert job["phase"] == "downloading_model"
    assert job["phase_started_at"] == "2026-08-17T12:00:00+00:00"
    assert job["heartbeat_at"] == "2026-08-17T12:04:00+00:00"
    assert job["progress"] == {"gpu_sku": "H200-SXM-141GB"}
    server.SubmissionDetailModel.model_validate(body)


def test_detail_drops_phase_text_outside_the_vocabulary(
    monkeypatch, client: TestClient
):
    """A hand-edited row must not put arbitrary phase text on a public endpoint."""
    from api import server

    sid = "88888888-8888-8888-8888-888888888888"
    monkeypatch.setattr(
        server,
        "get_submission_for_campaign",
        lambda _c, _h: _detail_row(sid, "c1", "sha256:junk"),
    )
    monkeypatch.setattr(server, "list_events", lambda _id: [])
    monkeypatch.setattr(
        server, "list_latest_states", lambda _ids: {sid: "bench_queued"}
    )
    monkeypatch.setattr(
        server,
        "list_submission_jobs",
        lambda _id: [
            {
                "status": "running",
                "last_error": None,
                "phase": "<script>alert(1)</script>",
                "phase_started_at": None,
                "heartbeat_at": None,
                "progress": {"deep": {"nope": 1}},
            }
        ],
    )

    job = client.get("/v1/campaigns/c1/submissions/sha256:junk").json()["jobs"][0]
    assert job["phase"] is None
    assert job["progress"] is None


def test_submissions_payload_matches_the_documented_model(
    monkeypatch, client: TestClient
):
    """`responses=` documents but does not validate; assert the real payload."""
    from api import server

    campaign = SimpleNamespace(status="open", to_public_dict=lambda: {"id": "c1"})
    sid = "11111111-1111-1111-1111-111111111111"
    monkeypatch.setattr(server, "get_campaign", lambda _cid: campaign)
    monkeypatch.setattr(
        server,
        "list_submissions",
        lambda _cid, *, limit=50, offset=0: {
            "total": 1,
            "items": [
                {
                    "id": sid,
                    "campaign_id": "c1",
                    "patch_hash": "abc",
                    "hotkey": "hk",
                    "baseline_commit": "deadbeef",
                    "retrieval_url": "https://example/p.diff",
                    "commit_block": 1,
                    "committed_at": "2026-08-14T00:00:00+00:00",
                    "engine_image_ref": None,
                }
            ],
        },
    )
    monkeypatch.setattr(
        server, "list_bench_summaries", lambda _cid, submission_ids=None: {}
    )
    monkeypatch.setattr(
        server, "list_latest_states", lambda _ids: {sid: "round_assigned"}
    )

    resp = client.get("/v1/campaigns/c1/submissions")
    assert resp.status_code == 200
    page = server.SubmissionsPageModel.model_validate(resp.json())
    assert page.submissions[0].latest_state == "round_assigned"
    # The documented contract must not have moved the timestamp format.
    assert resp.json()["submissions"][0]["committed_at"].endswith("+00:00")


def test_build_log_endpoint(monkeypatch, client: TestClient, tmp_path):
    from api import server

    sid = "22222222-2222-2222-2222-222222222222"
    row = {"id": sid, "patch_hash": "sha256:abc"}
    monkeypatch.setattr(server, "get_submission", lambda _h: row)
    monkeypatch.setattr(server, "count_submission_campaigns", lambda _h: 1)
    monkeypatch.setattr(server, "list_latest_states", lambda _ids: {sid: "building"})
    log_dir = tmp_path / sid
    log_dir.mkdir(parents=True)
    (log_dir / "build.log").write_bytes(b"step1\n\x1b[31mcolored\x1b[0m\nstep3\n")
    monkeypatch.setattr(server.config, "BUILD_LOG_DIR", tmp_path)

    resp = client.get("/v1/submissions/sha256:abc/build-log")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.headers.get("Cache-Control") == "no-store"
    body = resp.text
    assert "colored" in body
    assert "\x1b" not in body
    assert body.count("\n") == 3

    resp = client.get("/v1/submissions/sha256:abc/build-log?tail=1")
    assert resp.text.strip() == "step3"

    resp = client.get("/v1/submissions/sha256:abc/build-log?tail=99999")
    assert resp.status_code == 422


def test_build_log_404s(monkeypatch, client: TestClient, tmp_path):
    from api import server

    monkeypatch.setattr(server, "get_submission", lambda _h: None)
    assert client.get("/v1/submissions/nope/build-log").status_code == 404

    row = {"id": "33333333-3333-3333-3333-333333333333", "patch_hash": "p"}
    monkeypatch.setattr(server, "get_submission", lambda _h: row)
    monkeypatch.setattr(server, "count_submission_campaigns", lambda _h: 1)
    monkeypatch.setattr(server, "list_latest_states", lambda _ids: {})
    monkeypatch.setattr(server.config, "BUILD_LOG_DIR", tmp_path)
    assert client.get("/v1/submissions/p/build-log").status_code == 404


def _detail_row(sid: str, campaign_id: str, patch_hash: str) -> dict:
    return {
        "id": sid,
        "campaign_id": campaign_id,
        "patch_hash": patch_hash,
        "hotkey": "hk",
        "baseline_commit": "deadbeef",
        "retrieval_url": "https://example/p.diff",
        "commit_block": 1,
        "committed_at": "2026-08-07T00:00:00+00:00",
        "engine_image_ref": None,
        "created_at": "2026-08-07T00:00:00+00:00",
    }


def test_campaign_scoped_submission_detail(monkeypatch, client: TestClient):
    from api import server

    sid = "44444444-4444-4444-4444-444444444444"
    row = _detail_row(sid, "c1", "sha256:dup")
    monkeypatch.setattr(server, "get_submission_for_campaign", lambda _c, _h: row)
    monkeypatch.setattr(server, "list_events", lambda _id: [])
    monkeypatch.setattr(
        server, "list_latest_states", lambda _ids: {sid: "bench_queued"}
    )
    monkeypatch.setattr(
        server,
        "list_submission_jobs",
        lambda _id: [
            {
                "status": "failed",
                "last_error": "bench_exit_bad_request",
            },
            {"status": "done", "last_error": None},
        ],
    )

    resp = client.get("/v1/campaigns/c1/submissions/sha256:dup")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == "no-store"
    body = resp.json()
    assert body["submission"]["id"] == sid
    assert body["submission"]["campaign_id"] == "c1"
    assert body["events"] == []
    assert body["bench_verdict"] is None
    assert body["latest_state"] == "bench_queued"
    assert body["jobs"] == [
        {
            "status": "failed",
            "last_error": "bench_exit_bad_request",
            "phase": None,
            "phase_started_at": None,
            "heartbeat_at": None,
            "progress": None,
        },
        {
            "status": "done",
            "last_error": None,
            "phase": None,
            "phase_started_at": None,
            "heartbeat_at": None,
            "progress": None,
        },
    ]

    monkeypatch.setattr(server, "get_submission_for_campaign", lambda _c, _h: None)
    assert client.get("/v1/campaigns/c2/submissions/sha256:dup").status_code == 404


def test_campaign_scoped_build_log(monkeypatch, client: TestClient, tmp_path):
    from api import server

    sid = "55555555-5555-5555-5555-555555555555"
    row = {"id": sid, "patch_hash": "sha256:dup"}
    monkeypatch.setattr(server, "get_submission_for_campaign", lambda _c, _h: row)
    monkeypatch.setattr(server, "list_latest_states", lambda _ids: {sid: "building"})
    log_dir = tmp_path / sid
    log_dir.mkdir(parents=True)
    (log_dir / "build.log").write_bytes(b"line1\n\x1b[32mline2\x1b[0m\n")
    monkeypatch.setattr(server.config, "BUILD_LOG_DIR", tmp_path)

    resp = client.get("/v1/campaigns/c1/submissions/sha256:dup/build-log")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.headers.get("Cache-Control") == "no-store"
    assert "line2" in resp.text
    assert "\x1b" not in resp.text

    monkeypatch.setattr(server, "get_submission_for_campaign", lambda _c, _h: None)
    resp = client.get("/v1/campaigns/c2/submissions/sha256:dup/build-log")
    assert resp.status_code == 404


def test_bare_submission_routes_409_when_ambiguous(monkeypatch, client: TestClient):
    from api import server

    row = {"id": "66666666-6666-6666-6666-666666666666", "patch_hash": "sha256:dup"}
    monkeypatch.setattr(server, "get_submission", lambda _h: row)
    monkeypatch.setattr(server, "count_submission_campaigns", lambda _h: 2)

    resp = client.get("/v1/submissions/sha256:dup")
    assert resp.status_code == 409
    assert "multiple campaigns" in resp.json()["detail"]
    resp = client.get("/v1/submissions/sha256:dup/build-log")
    assert resp.status_code == 409


def test_bare_submission_detail_unique_hash_unchanged(monkeypatch, client: TestClient):
    from api import server

    sid = "77777777-7777-7777-7777-777777777777"
    row = _detail_row(sid, "c1", "sha256:solo")
    monkeypatch.setattr(server, "get_submission", lambda _h: row)
    monkeypatch.setattr(server, "count_submission_campaigns", lambda _h: 1)
    monkeypatch.setattr(server, "list_events", lambda _id: [])
    monkeypatch.setattr(server, "list_latest_states", lambda _ids: {})
    monkeypatch.setattr(server, "list_submission_jobs", lambda _id: [])

    resp = client.get("/v1/submissions/sha256:solo")
    assert resp.status_code == 200
    # Missing latest_state is treated as still live → no-store.
    assert resp.headers.get("Cache-Control") == "no-store"
    body = resp.json()
    assert body["submission"]["id"] == sid
    assert body["latest_state"] is None
    assert body["jobs"] == []


@pytest.mark.parametrize(
    ("latest_state", "expected_cache"),
    [
        ("building", "no-store"),
        ("bench_queued", "no-store"),
        ("built", V1_CACHE_CONTROL_EXPECTED),
        ("scored", V1_CACHE_CONTROL_EXPECTED),
        ("rejected", V1_CACHE_CONTROL_EXPECTED),
    ],
)
def test_submission_detail_cache_control_by_state(
    monkeypatch, client: TestClient, latest_state: str, expected_cache: str
):
    from api import server

    sid = "88888888-8888-8888-8888-888888888888"
    row = _detail_row(sid, "c1", "sha256:live")
    monkeypatch.setattr(server, "get_submission_for_campaign", lambda _c, _h: row)
    monkeypatch.setattr(server, "list_events", lambda _id: [])
    monkeypatch.setattr(server, "list_latest_states", lambda _ids: {sid: latest_state})
    monkeypatch.setattr(server, "list_submission_jobs", lambda _id: [])

    resp = client.get("/v1/campaigns/c1/submissions/sha256:live")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == expected_cache


@pytest.mark.parametrize(
    ("latest_state", "expected_cache"),
    [
        ("building", "no-store"),
        ("built", V1_CACHE_CONTROL_EXPECTED),
        ("scored", V1_CACHE_CONTROL_EXPECTED),
        ("rejected", V1_CACHE_CONTROL_EXPECTED),
    ],
)
def test_build_log_cache_control_by_state(
    monkeypatch, client: TestClient, tmp_path, latest_state: str, expected_cache: str
):
    from api import server

    sid = "99999999-9999-9999-9999-999999999999"
    row = {"id": sid, "patch_hash": "sha256:log"}
    monkeypatch.setattr(server, "get_submission_for_campaign", lambda _c, _h: row)
    monkeypatch.setattr(server, "list_latest_states", lambda _ids: {sid: latest_state})
    log_dir = tmp_path / sid
    log_dir.mkdir(parents=True)
    (log_dir / "build.log").write_bytes(b"ok\n")
    monkeypatch.setattr(server.config, "BUILD_LOG_DIR", tmp_path)

    resp = client.get("/v1/campaigns/c1/submissions/sha256:log/build-log")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == expected_cache
