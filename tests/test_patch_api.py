"""Permanent patch privacy: the public API never exposes patch locations."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api import server

CID = "11111111-1111-1111-1111-111111111111"
SID = "22222222-2222-2222-2222-222222222222"
HASH = "sha256:" + "a" * 64
BASE = f"/v1/campaigns/{CID}/submissions/{HASH}"
NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
RAW_URL = (
    "https://pareton-s3.s3.us-east-2.amazonaws.com/stage0/private/campaigns/"
    f"{CID}/patches/hk1/e444781d-171e-4067-b953-15322a5406c9.diff"
)
PUBLIC_URL = RAW_URL.replace("/private/campaigns/", "/campaigns/")
FORBIDDEN = ("retrieval_url", "patch_reveal_at", "patch_download_url")


@pytest.fixture
def scenario(monkeypatch):
    row = {
        "id": SID,
        "campaign_id": CID,
        "patch_hash": HASH,
        "hotkey": "hk1",
        "baseline_commit": "b" * 40,
        "retrieval_url": RAW_URL,
        "commit_block": 10,
        "committed_at": NOW.isoformat(),
        "engine_image_ref": "ghcr.io/pareton-ai/pareton-engine@sha256:abc",
    }
    events = [
        {
            "state": "built",
            "detail": {"build_log_tail": "public compiler output"},
            "created_at": NOW,
        },
        {"state": "committed", "detail": {}, "created_at": NOW},
    ]
    monkeypatch.setattr(
        server,
        "get_submission_for_campaign",
        lambda c, h: row if (c, h) == (CID, HASH) else None,
    )
    monkeypatch.setattr(server, "get_submission", lambda h: row if h == HASH else None)
    monkeypatch.setattr(server, "count_submission_campaigns", lambda h: 1)
    monkeypatch.setattr(server, "list_events", lambda _: events)
    monkeypatch.setattr(server, "list_latest_states", lambda _: {SID: "scored"})
    monkeypatch.setattr(
        server,
        "list_submission_jobs",
        lambda _: [{"status": "done", "last_error": "public diagnostics"}],
    )
    monkeypatch.setattr(
        server,
        "list_submission_round_entries",
        lambda _: {
            SID: {
                "round_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "ordinal": 3,
                "status": "scored",
                "score": 0.31,
                "disqualify_reason": None,
            }
        },
    )
    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
        lambda *a, **k: {
            "total": 1,
            "items": [{**row, "latest_state": "scored", "round": None}],
        },
    )
    return TestClient(server.app), row, events


def _submission(payload: dict) -> dict:
    return (
        payload["submissions"][0] if "submissions" in payload else payload["submission"]
    )


def _assert_private(response, url: str) -> None:
    assert response.status_code == 200
    assert url not in response.text
    submission = _submission(response.json())
    for key in FORBIDDEN:
        assert key not in submission
    assert submission["patch_hash"] == HASH


def test_submission_routes_never_expose_patch_locations(scenario):
    client, _, _ = scenario
    for path in (
        BASE,
        f"/v1/submissions/{HASH}",
        f"/v1/campaigns/{CID}/submissions",
    ):
        _assert_private(client.get(path), RAW_URL)


def test_historically_public_url_is_also_omitted(scenario):
    client, row, _ = scenario
    row["retrieval_url"] = PUBLIC_URL
    for path in (
        BASE,
        f"/v1/submissions/{HASH}",
        f"/v1/campaigns/{CID}/submissions",
    ):
        _assert_private(client.get(path), PUBLIC_URL)


def test_private_locator_is_scrubbed_from_events_and_jobs(scenario, monkeypatch):
    client, row, events = scenario
    events[0]["detail"] = {
        "error": f"fetch failed for {RAW_URL}: timed out",
        "nested": [{"source": RAW_URL, "code": 504}],
    }
    monkeypatch.setattr(
        server,
        "list_submission_jobs",
        lambda _: [{"status": "failed", "last_error": f"GET {RAW_URL} 403"}],
    )
    for path in (BASE, f"/v1/submissions/{HASH}"):
        response = client.get(path)
        assert RAW_URL not in response.text
        body = response.json()
        detail = body["events"][0]["detail"]
        assert detail["error"].endswith(": timed out")
        assert detail["nested"][0]["code"] == 504
        assert "[patch URL withheld]" in body["jobs"][0]["last_error"]
    assert events[0]["detail"]["nested"][0]["source"] == RAW_URL
    assert row["retrieval_url"] == RAW_URL


def test_patch_download_routes_are_gone(scenario):
    client, _, _ = scenario
    for path in (BASE + "/patch", f"/v1/submissions/{HASH}/patch"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 404
        assert RAW_URL not in response.text


def test_openapi_summary_model_has_no_patch_location_fields():
    props = server.app.openapi()["components"]["schemas"]["SubmissionSummaryModel"][
        "properties"
    ]
    for key in FORBIDDEN:
        assert key not in props
    assert "patch_hash" in props
