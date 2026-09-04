"""API contract tests (no live DB)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from campaign.store import KNOWN_CAMPAIGN_STATUSES, KNOWN_SUBMISSION_STATES
from round.rank import ENTRY_ROLES, ENTRY_STATUSES

V1_CACHE_CONTROL_EXPECTED = "public, max-age=30, stale-while-revalidate=300"


@pytest.fixture()
def client(monkeypatch):
    from api import server

    monkeypatch.setattr(
        server,
        "list_campaigns",
        lambda status=None: [],
    )
    # No unit test may reach the database. Tests that care override this.
    monkeypatch.setattr(server, "list_submission_round_entries", lambda _ids: {})
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

    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
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
                    "latest_state": "scored",
                    "round": {
                        "round_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                        "ordinal": 3,
                        "status": "scored",
                        "score": 0.31,
                        "disqualify_reason": None,
                    },
                }
            ],
        },
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
    assert row["round"]["ordinal"] == 3
    assert row["round"]["score"] == 0.31
    assert (
        resp.headers.get("Cache-Control")
        == "public, max-age=30, stale-while-revalidate=300"
    )


def test_submissions_offset_past_end(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
        lambda _cid, *, limit=50, offset=0: {"total": 3, "items": []},
    )

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

    monkeypatch.setattr(server, "list_campaign_submissions", lambda *_a, **_k: None)
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

    sid = "11111111-1111-1111-1111-111111111111"
    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
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
                    "latest_state": "round_assigned",
                    "round": None,
                }
            ],
        },
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
    assert body["round"] is None
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
        ("rejected_duplicate", V1_CACHE_CONTROL_EXPECTED),
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
        ("rejected_duplicate", V1_CACHE_CONTROL_EXPECTED),
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


HOTKEY = "5FakesHotkeyForE2ETesting000000000000000000000"
# The round read endpoints type campaign_id as a UUID, so these paths
# cannot use a short placeholder id.
CAMPAIGN_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def _leader_row() -> dict:
    return {
        "submission_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "patch_hash": "sha256:lead",
        "hotkey": HOTKEY,
        "engine_image_ref": "ghcr.io/x/e@sha256:" + "1" * 64,
        "won_at_round_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "won_at_ordinal": 2,
        "last_score": 0.31,
        "last_scored_round_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "updated_at": "2026-08-20T00:00:00+00:00",
    }


def _open_campaign(monkeypatch) -> None:
    from api import server

    monkeypatch.setattr(
        server,
        "get_campaign",
        lambda _cid: SimpleNamespace(
            status="open", to_public_dict=lambda: {"id": "c1"}
        ),
    )


def test_presign_rejects_campaign_disqualified_hotkey(monkeypatch, client: TestClient):
    from api import server

    _open_campaign(monkeypatch)
    monkeypatch.setattr(server, "campaign_hotkey_is_disqualified", lambda *_a: True)
    called = {"presign": False}
    monkeypatch.setattr(
        server,
        "create_presigned_patch_upload",
        lambda **_k: called.__setitem__("presign", True),
    )

    resp = client.post(
        "/v1/uploads/patch",
        json={"campaign_id": CAMPAIGN_ID, "hotkey": HOTKEY},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "hotkey is disqualified from campaign"
    assert called["presign"] is False


def test_presign_response_is_typed_in_openapi(monkeypatch, client: TestClient):
    from api import server

    _open_campaign(monkeypatch)
    monkeypatch.setattr(server, "campaign_hotkey_is_disqualified", lambda *_a: False)
    monkeypatch.setattr(
        server,
        "create_presigned_patch_upload",
        lambda **_k: SimpleNamespace(
            upload_url="https://upload.example/patch",
            retrieval_url="https://cdn.example/patch",
            object_key="stage0/campaigns/c/patches/h/p.diff",
            expires_in=900,
        ),
    )

    resp = client.post(
        "/v1/uploads/patch",
        json={"campaign_id": CAMPAIGN_ID, "hotkey": HOTKEY},
    )
    assert resp.status_code == 200
    server.PresignResponse.model_validate(resp.json())
    operation = client.get("/openapi.json").json()["paths"]["/v1/uploads/patch"]["post"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"] == "#/components/schemas/PresignResponse"
    assert "403" in operation["responses"]


def test_leader_is_404_when_vacant(monkeypatch, client: TestClient):
    """A vacant crown has no leaders row; the API must not invent an empty one."""
    from api import server

    _open_campaign(monkeypatch)
    monkeypatch.setattr(server, "get_leader", lambda _cid: None)
    resp = client.get(f"/v1/campaigns/{CAMPAIGN_ID}/leader")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "leader is vacant"


def test_leader_404_when_campaign_is_unknown(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(server, "get_campaign", lambda _cid: None)
    monkeypatch.setattr(server, "list_rounds", lambda *_a, **_k: None)
    for path in ("leader", "rounds", "score-progress"):
        resp = client.get(f"/v1/campaigns/{CAMPAIGN_ID}/{path}")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "campaign not found"


def test_round_reads_reject_a_malformed_campaign_id(monkeypatch, client: TestClient):
    """A bad id is a 422 from the path type, before any store call."""
    from api import server

    def _boom(*_a, **_kw):
        raise AssertionError("store must not be reached")

    monkeypatch.setattr(server, "get_campaign", _boom)
    monkeypatch.setattr(server, "list_rounds", _boom)
    for path in ("leader", "rounds", "score-progress"):
        resp = client.get(f"/v1/campaigns/not-a-uuid/{path}")
        assert resp.status_code == 422, path


def test_leader_detail_carries_the_full_hotkey(monkeypatch, client: TestClient):
    from api import server

    _open_campaign(monkeypatch)
    monkeypatch.setattr(server, "get_leader", lambda _cid: _leader_row())
    resp = client.get(f"/v1/campaigns/{CAMPAIGN_ID}/leader")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hotkey"] == HOTKEY
    assert body["campaign_id"] == CAMPAIGN_ID
    server.LeaderModel.model_validate(body)


def _round_summary(ordinal: int, status: str, **over) -> dict:
    row = {
        "id": f"cccccccc-cccc-cccc-cccc-{ordinal:012d}",
        "ordinal": ordinal,
        "status": status,
        "void_reason": None,
        "void_detail": None,
        "gpu_sku": "H200",
        "seed_block": 1000 + ordinal,
        "seed_block_hash": "0x" + f"{ordinal:064x}",
        "leader_changed": False,
        "created_at": "2026-08-20T00:00:00+00:00",
        "completed_at": None,
        "entry_count": 7,
    }
    row.update(over)
    return row


@pytest.mark.parametrize(
    ("status", "expected_cache"),
    [
        ("pending", "no-store"),
        ("running", "no-store"),
        ("complete", V1_CACHE_CONTROL_EXPECTED),
        ("void", V1_CACHE_CONTROL_EXPECTED),
    ],
)
def test_rounds_list_cache_control_follows_the_live_round(
    monkeypatch, client: TestClient, status: str, expected_cache: str
):
    from api import server

    _open_campaign(monkeypatch)
    monkeypatch.setattr(
        server,
        "list_rounds",
        lambda _cid, *, limit, offset: {
            "total": 1,
            "items": [_round_summary(1, status)],
        },
    )
    resp = client.get(f"/v1/campaigns/{CAMPAIGN_ID}/rounds")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == expected_cache
    page = server.RoundsPageModel.model_validate(resp.json())
    assert page.rounds[0].entry_count == 7
    assert page.limit == 50 and page.offset == 0


def test_rounds_list_keeps_void_ordinals(monkeypatch, client: TestClient):
    from api import server

    _open_campaign(monkeypatch)
    rows = [
        _round_summary(3, "complete", leader_changed=True),
        _round_summary(
            2,
            "void",
            void_reason="baseline_drift",
            void_detail="leader image vanished from ghcr",
        ),
        _round_summary(1, "complete"),
    ]
    monkeypatch.setattr(
        server,
        "list_rounds",
        lambda _cid, *, limit, offset: {"total": 3, "items": rows},
    )
    body = client.get(f"/v1/campaigns/{CAMPAIGN_ID}/rounds").json()
    assert [r["ordinal"] for r in body["rounds"]] == [3, 2, 1]
    assert body["rounds"][1]["void_reason"] == "baseline_drift"
    # The list carries the detail too: a miner scanning rounds should not have
    # to open each void to learn it was the same infra fault every time.
    assert body["rounds"][1]["void_detail"] == "leader image vanished from ghcr"


def test_rounds_list_campaign_404(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(server, "list_rounds", lambda *_a, **_k: None)
    resp = client.get(f"/v1/campaigns/{CAMPAIGN_ID}/rounds")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "campaign not found"


def test_round_detail_explains_why_it_voided(monkeypatch, client: TestClient):
    """void_reason is a bare code; void_detail is the sentence behind it."""
    from api import server

    monkeypatch.setattr(
        server,
        "get_round",
        lambda _rid: _round_row(
            status="void",
            phase=None,
            void_reason="pod_failed",
            void_detail="provider returned 503 after 3 retries",
        ),
    )
    monkeypatch.setattr(server, "list_round_entries", lambda _rid: [])
    body = client.get(f"/v1/rounds/{ROUND_ID}").json()
    server.RoundDetailModel.model_validate(body)
    assert body["void_reason"] == "pod_failed"
    assert body["void_detail"] == "provider returned 503 after 3 retries"


def test_a_round_that_did_not_void_carries_no_detail(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(server, "get_round", lambda _rid: _round_row())
    monkeypatch.setattr(server, "list_round_entries", lambda _rid: [])
    body = client.get(f"/v1/rounds/{ROUND_ID}").json()
    assert body["void_detail"] is None


ROUND_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"


def _round_row(**over) -> dict:
    row = {
        "id": ROUND_ID,
        "campaign_id": "c1",
        "ordinal": 4,
        "status": "running",
        "void_reason": None,
        "void_detail": None,
        "gpu_sku": "H200",
        "seed_block": 1004,
        "seed_block_hash": "0x" + "a" * 64,
        "seed_hex": "b" * 64,
        "sampled_trace_sha256": "sha256:" + "c" * 64,
        "scoring_rule": {"name": "median_e2e_speedup"},
        "incumbent_submission_id": None,
        "winner_submission_id": None,
        "leader_changed": None,
        "baseline_drift": None,
        "phase": "sla_bench",
        "phase_started_at": "2026-08-20T00:00:00+00:00",
        "heartbeat_at": "2026-08-20T00:01:00+00:00",
        "progress": {"entry": 2},
        "created_at": "2026-08-20T00:00:00+00:00",
        "started_at": "2026-08-20T00:00:00+00:00",
        "completed_at": None,
    }
    row.update(over)
    return row


def _round_entries() -> list[dict]:
    return [
        {
            "id": 1,
            "submission_id": None,
            "role": "baseline",
            "engine_image_ref": "ghcr.io/x/e@sha256:" + "0" * 64,
            "status": "scored",
            "score": 0.0,
            "disqualify_reason": None,
            "started_at": None,
            "completed_at": None,
            "patch_hash": None,
            "hotkey": None,
        },
        {
            "id": 2,
            "submission_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
            "role": "challenger",
            "engine_image_ref": "ghcr.io/x/e@sha256:" + "1" * 64,
            "status": "disqualified",
            "score": None,
            "disqualify_reason": "fail_correctness",
            "started_at": None,
            "completed_at": None,
            "patch_hash": "sha256:dq",
            "hotkey": HOTKEY,
        },
    ]


def test_round_detail_entries_and_live_phase(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(server, "get_round", lambda _rid: _round_row())
    monkeypatch.setattr(server, "list_round_entries", lambda _rid: _round_entries())

    resp = client.get(f"/v1/rounds/{ROUND_ID}")
    assert resp.status_code == 200
    # A running round is live and must not be cached.
    assert resp.headers.get("Cache-Control") == "no-store"
    body = resp.json()
    server.RoundDetailModel.model_validate(body)
    assert body["phase"] == "sla_bench"
    assert body["progress"] == {"entry": 2}
    baseline, challenger = body["entries"]
    # Serialized enums are the round/rank.py vocabularies, not a respelling.
    assert {e["role"] for e in body["entries"]} <= set(ENTRY_ROLES)
    assert {e["status"] for e in body["entries"]} <= set(ENTRY_STATUSES)
    # 0.0 is a real score; a disqualified entry has none.
    assert baseline["score"] == 0.0
    assert challenger["score"] is None
    assert challenger["disqualify_reason"] == "fail_correctness"
    # Detail page: full hotkey. Evidence stays behind its gate.
    assert challenger["hotkey"] == HOTKEY
    assert "evidence_s3_url" not in challenger
    assert "report" not in challenger


def test_round_detail_drops_phase_text_outside_the_vocabulary(
    monkeypatch, client: TestClient
):
    from api import server

    monkeypatch.setattr(
        server,
        "get_round",
        lambda _rid: _round_row(phase="<script>", progress={"deep": {"no": 1}}),
    )
    monkeypatch.setattr(server, "list_round_entries", lambda _rid: [])
    body = client.get(f"/v1/rounds/{ROUND_ID}").json()
    assert body["phase"] is None
    assert body["progress"] is None


def test_round_detail_404_and_bad_uuid(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(server, "get_round", lambda _rid: None)
    assert client.get(f"/v1/rounds/{ROUND_ID}").status_code == 404
    assert client.get("/v1/rounds/not-a-uuid").status_code == 422


def test_round_detail_of_a_terminal_round_is_cacheable(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(
        server, "get_round", lambda _rid: _round_row(status="void", phase=None)
    )
    monkeypatch.setattr(server, "list_round_entries", lambda _rid: [])
    resp = client.get(f"/v1/rounds/{ROUND_ID}")
    assert resp.headers.get("Cache-Control") == V1_CACHE_CONTROL_EXPECTED


# --- GET /v1/rounds/{id}/entries/{id}/report -------------------------------


def _score_report_row(**over) -> dict:
    """One scored challenger, as get_round_entry_report returns it."""
    row = {
        "id": 2,
        "round_id": ROUND_ID,
        "submission_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        "role": "challenger",
        "engine_image_ref": "ghcr.io/x/e@sha256:" + "1" * 64,
        "status": "scored",
        "score": 0.7194,
        "disqualify_reason": None,
        "started_at": None,
        "completed_at": None,
        "patch_hash": "sha256:win",
        "hotkey": HOTKEY,
        "round_status": "complete",
        "round_ordinal": 12,
        "scoring_rule": {"name": "median_e2e_speedup", "tolerance": 0.9},
        "report": {
            "index": 0,
            "image_digest": "sha256:" + "1" * 64,
            "status": "scored",
            "score": 0.7194,
            "reason": None,
            "score_report": {
                "rule": "median_e2e_speedup",
                "score": 0.7194,
                "prompts": [
                    {
                        "request_id": "req-0",
                        "speedup": 0.7194,
                        "aligned_tokens": 44,
                        "baseline_e2e_s": 1.811,
                        "candidate_e2e_s": 0.508,
                        "reason": None,
                    },
                    {
                        "request_id": "req-1",
                        "speedup": 0.0,
                        "aligned_tokens": 38,
                        "baseline_e2e_s": None,
                        "candidate_e2e_s": None,
                        "reason": "candidate output below tolerance",
                    },
                ],
            },
            "sla": {"role": "candidate", "metrics": {"output_tokens_per_s": 91.2}},
            "correctness": {"verdict": "pass", "mean_logprob": -0.21},
        },
    }
    row.update(over)
    return row


def test_entry_report_serves_the_per_prompt_breakdown(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(
        server, "get_round_entry_report", lambda _rid, _eid: _score_report_row()
    )
    resp = client.get(f"/v1/rounds/{ROUND_ID}/entries/2/report")
    assert resp.status_code == 200
    body = resp.json()
    server.RoundEntryReportModel.model_validate(body)

    assert body["round_ordinal"] == 12
    assert body["entry_id"] == 2
    assert body["hotkey"] == HOTKEY
    assert body["scoring_rule"] == {"name": "median_e2e_speedup", "tolerance": 0.9}
    # Absolute seconds travel with the ratio: a speedup alone cannot be
    # checked against a local run.
    assert body["prompts"][0]["baseline_e2e_s"] == 1.811
    assert body["prompts"][0]["candidate_e2e_s"] == 0.508
    # The count that was asked for by name: prompts the tolerance gate zeroed.
    assert body["prompt_summary"]["total"] == 2
    assert body["prompt_summary"]["scored"] == 1
    assert body["prompt_summary"]["below_tolerance"] == 1
    assert body["correctness"]["verdict"] == "pass"
    assert body["sla"]["metrics"]["output_tokens_per_s"] == 91.2
    # Evidence keeps its own gate; the report does not carry it out.
    assert "evidence_s3_url" not in body


def test_entry_report_of_a_live_round_is_not_cached(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(
        server,
        "get_round_entry_report",
        lambda _rid, _eid: _score_report_row(round_status="running"),
    )
    resp = client.get(f"/v1/rounds/{ROUND_ID}/entries/2/report")
    assert resp.headers.get("Cache-Control") == "no-store"

    monkeypatch.setattr(
        server, "get_round_entry_report", lambda _rid, _eid: _score_report_row()
    )
    resp = client.get(f"/v1/rounds/{ROUND_ID}/entries/2/report")
    assert resp.headers.get("Cache-Control") == V1_CACHE_CONTROL_EXPECTED


def test_entry_report_carries_the_reason_for_a_non_scored_entry(
    monkeypatch, client: TestClient
):
    """A disqualified entry never reached scoring, so it has no prompts."""
    from api import server

    row = _score_report_row(
        status="disqualified",
        score=None,
        disqualify_reason="mean_logprob -3.9 below -2.0",
        report={
            "index": 0,
            "image_digest": "sha256:" + "2" * 64,
            "status": "disqualified",
            "score": None,
            "reason": "mean_logprob -3.9 below -2.0",
            "correctness": {"verdict": "fail_correctness", "mean_logprob": -3.9},
        },
    )
    monkeypatch.setattr(server, "get_round_entry_report", lambda _rid, _eid: row)
    body = client.get(f"/v1/rounds/{ROUND_ID}/entries/2/report").json()
    server.RoundEntryReportModel.model_validate(body)
    assert body["score"] is None
    assert body["reason"] == "mean_logprob -3.9 below -2.0"
    assert body["prompts"] == []
    assert body["prompt_summary"]["total"] == 0
    assert body["correctness"]["verdict"] == "fail_correctness"


def test_entry_report_reads_the_baseline_row_as_an_sla_replay(
    monkeypatch, client: TestClient
):
    """The baseline entry stores its replay, not a comparison against itself."""
    from api import server

    row = _score_report_row(
        id=1,
        role="baseline",
        submission_id=None,
        patch_hash=None,
        hotkey=None,
        score=0.0,
        report={
            "role": "baseline",
            "metrics": {"output_tokens_per_s": 24.1},
            "cross_rep_variance": {"p99_e2e_ms_rel_range": 0.018},
            "timings": {
                "req-0": {"ttft_s": 0.09, "itl_s": [], "completion_tokens": 44}
            },
            "evidence": "sla_bench/",
        },
    )
    monkeypatch.setattr(server, "get_round_entry_report", lambda _rid, _eid: row)
    body = client.get(f"/v1/rounds/{ROUND_ID}/entries/1/report").json()
    server.RoundEntryReportModel.model_validate(body)
    assert body["role"] == "baseline"
    assert body["score"] == 0.0
    assert body["prompts"] == []
    # The stored replay is the SLA block, reachable under the same key as a
    # candidate's, so one client path reads either shape.
    assert body["sla"]["metrics"]["output_tokens_per_s"] == 24.1
    assert body["sla"]["cross_rep_variance"]["p99_e2e_ms_rel_range"] == 0.018


def test_entry_report_survives_an_empty_or_legacy_report_blob(
    monkeypatch, client: TestClient
):
    from api import server

    row = _score_report_row(status="infra_failed", score=None, report={})
    monkeypatch.setattr(server, "get_round_entry_report", lambda _rid, _eid: row)
    body = client.get(f"/v1/rounds/{ROUND_ID}/entries/2/report").json()
    server.RoundEntryReportModel.model_validate(body)
    assert body["prompts"] == []
    assert body["sla"] is None
    assert body["correctness"] is None
    assert body["engine_crashed"] is False


def test_entry_report_404_and_bad_ids(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(server, "get_round_entry_report", lambda _rid, _eid: None)
    assert client.get(f"/v1/rounds/{ROUND_ID}/entries/2/report").status_code == 404

    def _boom(*_a, **_kw):
        raise AssertionError("store must not be reached")

    monkeypatch.setattr(server, "get_round_entry_report", _boom)
    assert client.get("/v1/rounds/not-a-uuid/entries/2/report").status_code == 422
    assert client.get(f"/v1/rounds/{ROUND_ID}/entries/nope/report").status_code == 422


def test_score_progress_keeps_void_ordinals_and_null_scores(
    monkeypatch, client: TestClient
):
    """Void rounds leave gaps at their ordinal; nothing is renumbered to 0."""
    from api import server

    _open_campaign(monkeypatch)
    points = [
        {
            "round_id": "11111111-1111-1111-1111-111111111111",
            "ordinal": 1,
            "status": "complete",
            "leader_score": 0.31,
            "entries": [
                {
                    "submission_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                    "hotkey": HOTKEY,
                    "role": "challenger",
                    "status": "disqualified",
                    "score": None,
                }
            ],
        },
        {
            "round_id": "22222222-2222-2222-2222-222222222222",
            "ordinal": 2,
            "status": "void",
            "leader_score": None,
            "entries": [],
        },
        {
            "round_id": "33333333-3333-3333-3333-333333333333",
            "ordinal": 3,
            "status": "complete",
            "leader_score": 0.4,
            "entries": [],
        },
    ]
    monkeypatch.setattr(server, "list_score_progress", lambda _cid: points)

    resp = client.get(f"/v1/campaigns/{CAMPAIGN_ID}/score-progress")
    assert resp.status_code == 200
    body = resp.json()
    server.ScoreProgressModel.model_validate(body)
    assert [p["ordinal"] for p in body["points"]] == [1, 2, 3]
    assert [p["leader_score"] for p in body["points"]] == [0.31, None, 0.4]
    # A disqualified entry is null, never 0.0. The client decides how to draw it.
    assert body["points"][0]["entries"][0]["score"] is None
    # List response: the hotkey is truncated.
    assert body["points"][0]["entries"][0]["hotkey"] == HOTKEY[:16]
    assert resp.headers.get("Cache-Control") == V1_CACHE_CONTROL_EXPECTED


def test_score_progress_is_uncacheable_while_a_round_runs(
    monkeypatch, client: TestClient
):
    from api import server

    _open_campaign(monkeypatch)
    monkeypatch.setattr(
        server,
        "list_score_progress",
        lambda _cid: [
            {
                "round_id": "44444444-4444-4444-4444-444444444444",
                "ordinal": 1,
                "status": "running",
                "leader_score": None,
                "entries": [],
            }
        ],
    )
    resp = client.get(f"/v1/campaigns/{CAMPAIGN_ID}/score-progress")
    assert resp.headers.get("Cache-Control") == "no-store"


def _stored_weight_set() -> dict:
    """A `weight_sets` row: 202 dense slots, uid 12 paid, the rest burned."""
    dense = [0.0] * 202
    dense[12] = 0.1
    dense[201] = 0.9
    return {
        "computed_at_block": 6123456,
        "version_key": 2032,
        "burn_uid": 201,
        "weights": dense,
        "breakdown": [
            {
                "campaign_id": CAMPAIGN_ID,
                "hotkey": HOTKEY,
                "uid": 12,
                "blocks_held": 43200,
                "weight": 0.1,
                "note": None,
            },
            {
                "campaign_id": "22222222-2222-2222-2222-222222222222",
                "hotkey": None,
                "uid": None,
                "blocks_held": None,
                "weight": 0.0,
                "note": "vacant",
            },
        ],
    }


def test_weights_serves_the_stored_dense_vector(monkeypatch, client: TestClient):
    from api import server

    monkeypatch.setattr(server, "get_latest_weight_set", _stored_weight_set)

    resp = client.get("/v1/weights")
    assert resp.status_code == 200
    body = resp.json()
    server.WeightsModel.model_validate(body)
    assert "uids" not in body
    # Dense wire form: index is UID. Length is the stored vector, not padded.
    assert body["weights"][12] == pytest.approx(0.1)
    assert body["weights"][201] == pytest.approx(0.9)
    assert len(body["weights"]) == 202
    assert sum(body["weights"]) == pytest.approx(1.0)
    assert body["computed_at_block"] == 6123456
    assert body["version_key"] == 2032
    assert body["burn_uid"] == 201
    # A withheld share stays in the breakdown, so the burn is auditable.
    assert [b["note"] for b in body["breakdown"]] == [None, "vacant"]
    # A cached weight vector is worse than none.
    assert resp.headers.get("Cache-Control") == "no-store"


def test_weights_with_no_stored_row_is_404_not_an_empty_vector(
    monkeypatch, client: TestClient
):
    """An empty vector is a valid on-chain instruction meaning pay nobody."""
    from api import server

    monkeypatch.setattr(server, "get_latest_weight_set", lambda: None)

    resp = client.get("/v1/weights")
    assert resp.status_code == 404
    assert "uids" not in resp.json()
    assert "weights" not in resp.json()
    # 404 is the live path until the first row. A cache holding it would hide
    # the first published vector from other validators.
    assert resp.headers.get("Cache-Control") == "no-store"


def test_weights_all_zero_row_is_404_not_pay_nobody(monkeypatch, client: TestClient):
    """The reader cannot trust the writer across a deploy version skew."""
    from api import server

    def zeros() -> dict:
        row = _stored_weight_set()
        row["weights"] = [0.0] * len(row["weights"])
        return row

    monkeypatch.setattr(server, "get_latest_weight_set", zeros)

    resp = client.get("/v1/weights")
    assert resp.status_code == 404
    assert "uids" not in resp.json()
    assert "weights" not in resp.json()
    assert resp.headers.get("Cache-Control") == "no-store"
