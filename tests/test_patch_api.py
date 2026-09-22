"""Permanent confidentiality across public patch and diagnostic routes."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import config
from api import server
from storage import s3

CID = "11111111-1111-1111-1111-111111111111"
SID = "22222222-2222-2222-2222-222222222222"
HASH = "sha256:" + "a" * 64
BASE = f"/v1/campaigns/{CID}/submissions/{HASH}"
NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
RAW_URL = (
    f"https://pareton-s3.s3.us-east-2.amazonaws.com/stage0/campaigns/{CID}"
    "/patches/hk1/e444781d-171e-4067-b953-15322a5406c9.diff"
)


@pytest.fixture
def scenario(monkeypatch):
    row = {
        "id": SID,
        "campaign_id": CID,
        "patch_hash": HASH,
        "hotkey": "hk1",
        "baseline_commit": "b" * 40,
        "retrieval_url": RAW_URL,
        "committed_at": NOW.isoformat(),
        "engine_image_ref": "ghcr.io/pareton-ai/pareton-engine@sha256:abc",
    }
    events = [
        {
            "state": "built",
            "detail": {"build_log_tail": "public compiler output"},
            "created_at": NOW.isoformat(),
        }
    ]
    # A present key enrolls a new submission; None means it is not evaluated yet.
    times = {SID: None}
    monkeypatch.setattr(config, "PATCH_REVEAL_DELAY_S", 21600)
    monkeypatch.setattr(
        server,
        "get_submission_for_campaign",
        lambda c, h: row if (c, h) == (CID, HASH) else None,
    )
    monkeypatch.setattr(server, "get_submission", lambda h: row if h == HASH else None)
    monkeypatch.setattr(server, "count_submission_campaigns", lambda h: 1)
    monkeypatch.setattr(
        server,
        "list_events",
        lambda _: (
            events
            + [
                {
                    "state": "committed",
                    "detail": {"patch_reveal_delayed": SID in times},
                    "created_at": NOW.isoformat(),
                }
            ]
        ),
    )
    monkeypatch.setattr(server, "list_latest_states", lambda _: {SID: "scored"})
    monkeypatch.setattr(
        server,
        "list_submission_jobs",
        lambda _: [{"status": "done", "last_error": "public diagnostics"}],
    )
    monkeypatch.setattr(
        server,
        "list_submission_round_entries",
        lambda _: (
            {SID: {"_patch_evaluated_at": times[SID]}}
            if times.get(SID) is not None
            else {}
        ),
    )
    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
        lambda *a, **k: {
            "total": 1,
            "items": [
                {
                    **row,
                    "_patch_reveal_delayed": SID in times,
                    "_patch_evaluated_at": times.get(SID),
                }
            ],
        },
    )
    return TestClient(server.app), times, row, events


@pytest.mark.parametrize("enrolled", [False, True])
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("age", [None, timedelta(days=36500)])
@pytest.mark.parametrize("delay", [0, 315360000])
def test_all_patch_routes_stay_private(
    scenario, monkeypatch, enrolled, private, age, delay
):
    client, times, row, events = scenario
    monkeypatch.setattr(config, "PATCH_REVEAL_DELAY_S", delay)
    if not enrolled:
        times.clear()
    elif age is not None:
        times[SID] = NOW - age
    if private:
        row["retrieval_url"] = RAW_URL.replace("/campaigns/", "/private/campaigns/")
    secret = "private source line from compiler diagnostic"
    events[0]["detail"] = {
        "build_log_tail": secret,
        "nested": {"error": row["retrieval_url"]},
    }
    events[0]["evidence_ref"] = "https://example.test/private-artifact.tar.gz"
    monkeypatch.setattr(
        s3, "_client", lambda **kw: pytest.fail("public API must not access S3")
    )
    for path in (BASE, f"/v1/submissions/{HASH}"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        data = response.json()
        assert data["submission"]["retrieval_url"] == ""
        assert data["submission"]["patch_reveal_at"] is None
        assert data["submission"]["patch_download_url"] is None
        assert data["submission"]["patch_hash"] == HASH
        assert data["latest_state"] == "scored"
        assert data["events"][0]["detail"] == {}
        assert data["events"][0]["evidence_ref"] is None
        assert data["jobs"][0]["last_error"] is None
        assert secret not in response.text
        assert row["retrieval_url"] not in response.text
        for suffix, reason in (
            ("/patch", "patch_private"),
            ("/build-log", "build_log_private"),
        ):
            denied = client.get(path + suffix, follow_redirects=False)
            assert denied.status_code == 403
            assert denied.json()["detail"]["reason"] == reason
            assert denied.headers["cache-control"] == "no-store"
            assert "location" not in denied.headers
    response = client.get(f"/v1/campaigns/{CID}/submissions")
    assert response.headers["cache-control"] == "no-store"
    item = response.json()["submissions"][0]
    assert item["retrieval_url"] == ""
    assert item["patch_reveal_at"] is None
    assert item["patch_download_url"] is None
    assert "_patch_evaluated_at" not in response.text
    assert "_patch_reveal_delayed" not in response.text
    # API redaction never mutates stored audit evidence or internal locators.
    assert events[0]["detail"]["build_log_tail"] == secret
    assert events[0]["evidence_ref"].endswith(".tar.gz")
    assert row["retrieval_url"]


def test_mixed_legacy_listing_keeps_every_patch_private(scenario, monkeypatch):
    client, _, row, _ = scenario
    old = {**row, "id": "33333333-3333-3333-3333-333333333333"}
    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
        lambda *a, **k: {
            "total": 2,
            "items": [old, {**row, "_patch_reveal_delayed": True}],
        },
    )
    response = client.get(f"/v1/campaigns/{CID}/submissions")
    assert response.headers["cache-control"] == "no-store"
    assert [s["retrieval_url"] for s in response.json()["submissions"]] == ["", ""]


@pytest.mark.parametrize("suffix", ["/patch", "/build-log"])
def test_private_routes_preserve_missing_and_ambiguous_lookups(
    scenario, monkeypatch, suffix
):
    client, _, _, _ = scenario
    assert client.get(BASE.replace(CID, "other") + suffix).status_code == 404
    monkeypatch.setattr(server, "count_submission_campaigns", lambda _: 2)
    assert client.get(f"/v1/submissions/{HASH}" + suffix).status_code == 409


@pytest.mark.parametrize("route", ["round", "detail", "legacy_detail", "list"])
@pytest.mark.parametrize(
    "status", ["pending", "running", "disqualified", "infra_failed", "scored"]
)
def test_public_entry_reasons_follow_entry_status_on_every_route(
    scenario, monkeypatch, route, status
):
    from copy import deepcopy

    client, _, row, _ = scenario
    reason = 'Traceback:\n  File "/src/patched.py", line 42\n    raise ValueError("private source")'
    entry = {
        "round_id": CID,
        "ordinal": 3,
        "status": status,
        "score": 0.5 if status == "scored" else None,
        "disqualify_reason": reason,
    }
    stored = deepcopy(entry)
    monkeypatch.setattr(
        server, "get_round", lambda _: {"id": CID, "status": "complete"}
    )
    monkeypatch.setattr(server, "list_round_entries", lambda _: [entry])
    monkeypatch.setattr(server, "list_submission_round_entries", lambda _: {SID: entry})
    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
        lambda *a, **k: {
            "total": 1,
            "items": [{**row, "round": entry, "latest_state": "scored"}],
        },
    )
    paths = {
        "round": f"/v1/rounds/{CID}",
        "detail": BASE,
        "legacy_detail": f"/v1/submissions/{HASH}",
        "list": f"/v1/campaigns/{CID}/submissions",
    }
    response = client.get(paths[route])
    assert response.status_code == 200
    payload = response.json()
    if route == "round":
        public = payload["entries"][0]
    elif route == "list":
        public = payload["submissions"][0]["round"]
    else:
        public = payload["round"]
    assert public["disqualify_reason"] == (reason if status == "scored" else None)
    assert public["status"] == status
    assert public["score"] == entry["score"]

    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    if status != "scored":
        # Inspect decoded JSON; wire escaping must not conceal leaked source.
        assert all(reason not in text for text in strings(payload))
    assert entry == stored
