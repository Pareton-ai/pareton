"""Campaign-controlled disclosure: private by default, timed reveal by opt-in."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api import server
from storage.visibility import patch_is_revealed

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
    monkeypatch.setattr(
        server, "publish_patch", lambda *a: pytest.fail("unexpected publication")
    )
    monkeypatch.setattr(
        server,
        "list_patch_evaluation_times",
        lambda *a: pytest.fail("private policy must not query reveal times"),
    )
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
    assert submission["patch_visibility"] == {"mode": "private"}


def test_submission_routes_never_expose_patch_locations(scenario):
    client, _, _ = scenario
    for path in (
        BASE,
        BASE + "/patch-availability",
        f"/v1/submissions/{HASH}",
        f"/v1/campaigns/{CID}/submissions",
    ):
        _assert_private(client.get(path), RAW_URL)


def test_historically_public_url_is_also_omitted(scenario):
    client, row, _ = scenario
    row["retrieval_url"] = PUBLIC_URL
    for path in (
        BASE,
        BASE + "/patch-availability",
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


def test_private_patch_download_routes_are_denied(scenario):
    client, _, _ = scenario
    for path in (BASE + "/patch", f"/v1/submissions/{HASH}/patch"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 403
        assert response.json()["detail"]["reason"] == "patch_private"
        assert response.headers["cache-control"] == "no-store"
        assert RAW_URL not in response.text


def test_openapi_summary_model_documents_optional_reveal_fields():
    props = server.app.openapi()["components"]["schemas"]["SubmissionSummaryModel"][
        "properties"
    ]
    for key in FORBIDDEN:
        assert key in props
        assert (
            key
            not in server.app.openapi()["components"]["schemas"][
                "SubmissionSummaryModel"
            ]["required"]
        )
    assert "patch_hash" in props


@pytest.fixture
def revealing(scenario, monkeypatch):
    client, row, events = scenario
    row["_patch_visibility"] = {"mode": "public_after_reveal", "reveal_delay_s": 21600}
    row["_patch_evaluated_at"] = None
    publications = []

    def publish(url, patch_hash):
        publications.append((url, patch_hash))
        return PUBLIC_URL

    monkeypatch.setattr(server, "publish_patch", publish)
    monkeypatch.setattr(
        server, "patch_is_revealed", lambda ts, p: patch_is_revealed(ts, p, now=NOW)
    )
    monkeypatch.setattr(
        server,
        "list_patch_evaluation_times",
        lambda _: {SID: row["_patch_evaluated_at"]},
    )
    monkeypatch.setattr(
        server,
        "list_submission_round_entries",
        lambda _: {SID: {"_patch_evaluated_at": row["_patch_evaluated_at"]}},
    )
    return client, row, events, publications


@pytest.mark.parametrize("age", [None, 21599, 21600, 21601, 864000])
def test_all_routes_observe_exact_reveal_boundary(revealing, age):
    client, row, _, publications = revealing
    evaluated_at = NOW - timedelta(seconds=age) if age is not None else None
    row["_patch_evaluated_at"] = evaluated_at
    revealed = age is not None and age >= 21600
    for path in (BASE, f"/v1/submissions/{HASH}", f"/v1/campaigns/{CID}/submissions"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        submission = _submission(response.json())
        assert RAW_URL not in response.text
        assert "_patch_" not in response.text
        assert submission["patch_visibility"] == row["_patch_visibility"]
        assert submission["retrieval_url"] == ""
        assert submission["patch_download_url"] is None
        assert submission["patch_reveal_at"] == (
            (evaluated_at + timedelta(hours=6)).isoformat() if evaluated_at else None
        )
    assert publications == []
    response = client.get(BASE + "/patch-availability")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    submission = response.json()["submission"]
    assert submission["retrieval_url"] == (PUBLIC_URL if revealed else "")
    assert bool(submission["patch_download_url"]) == revealed
    assert submission["patch_reveal_at"] == (
        (evaluated_at + timedelta(hours=6)).isoformat() if evaluated_at else None
    )
    assert RAW_URL not in response.text
    for path in (BASE, f"/v1/submissions/{HASH}"):
        response = client.get(path + "/patch", follow_redirects=False)
        assert response.status_code == (307 if revealed else 403)
        assert response.headers["cache-control"] == "no-store"
        if revealed:
            assert response.headers["location"] == PUBLIC_URL
        else:
            assert response.json()["detail"]["reason"] == "patch_not_revealed"
    assert bool(publications) == revealed
    assert row["retrieval_url"] == RAW_URL


@pytest.mark.parametrize("url", [RAW_URL, PUBLIC_URL])
def test_private_policy_overrides_old_enrollment_and_evaluation(revealing, url):
    client, row, events, publications = revealing
    row["retrieval_url"] = url
    row["_patch_visibility"] = {"mode": "private"}
    row["_patch_evaluated_at"] = NOW - timedelta(days=365)
    events[-1]["detail"] = {"patch_reveal_delayed": True}
    for path in (BASE, f"/v1/submissions/{HASH}", f"/v1/campaigns/{CID}/submissions"):
        _assert_private(client.get(path), url)
    assert client.get(BASE + "/patch").json()["detail"]["reason"] == "patch_private"
    assert publications == []


def test_policy_changes_are_read_on_each_request(revealing):
    client, row, _, publications = revealing
    row["_patch_evaluated_at"] = NOW - timedelta(hours=6)
    assert (
        client.get(BASE + "/patch-availability").json()["submission"]["retrieval_url"]
        == PUBLIC_URL
    )
    row["_patch_visibility"]["reveal_delay_s"] += 1
    assert client.get(BASE + "/patch").status_code == 403
    assert (
        client.get(BASE + "/patch-availability").json()["submission"]["retrieval_url"]
        == ""
    )
    row["_patch_visibility"] = {"mode": "private"}
    _assert_private(client.get(BASE), RAW_URL)
    _assert_private(client.get(BASE + "/patch-availability"), RAW_URL)
    assert client.get(BASE + "/patch").status_code == 403
    assert len(publications) == 1


def test_repeated_full_pages_and_details_never_attempt_publication(
    revealing, monkeypatch
):
    client, row, _, _ = revealing
    row["_patch_evaluated_at"] = NOW - timedelta(days=3)
    monkeypatch.setattr(
        server, "publish_patch", lambda *a: pytest.fail("metadata must not wait on S3")
    )
    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
        lambda *a, **k: {
            "total": 200,
            "items": [{**row, "id": str(i)} for i in range(200)],
        },
    )
    for _ in range(3):
        response = client.get(f"/v1/campaigns/{CID}/submissions?limit=200")
        assert response.status_code == 200
        rows = response.json()["submissions"]
        assert len(rows) == 200
        assert all(r["retrieval_url"] == "" for r in rows)
        assert all(r["patch_download_url"] is None for r in rows)
        for path in (BASE, f"/v1/submissions/{HASH}"):
            assert client.get(path).status_code == 200


def test_publication_failure_is_retryable_and_keeps_json_available(
    revealing, monkeypatch
):
    client, row, _, _ = revealing
    row["_patch_evaluated_at"] = NOW - timedelta(days=3)

    def unavailable(*args):
        raise RuntimeError("sensitive storage failure " + RAW_URL)

    monkeypatch.setattr(server, "publish_patch", unavailable)
    for path in (BASE, f"/v1/submissions/{HASH}", f"/v1/campaigns/{CID}/submissions"):
        response = client.get(path)
        assert response.status_code == 200
        assert _submission(response.json())["retrieval_url"] == ""
        assert _submission(response.json())["patch_download_url"] is None
        assert RAW_URL not in response.text
    response = client.get(BASE + "/patch-availability")
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert RAW_URL not in response.text
    for path in (BASE, f"/v1/submissions/{HASH}"):
        response = client.get(path + "/patch", follow_redirects=False)
        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
        assert RAW_URL not in response.text


def test_download_rejects_ambiguous_hash_and_missing_submission(revealing, monkeypatch):
    client, _, _, publications = revealing
    monkeypatch.setattr(server, "count_submission_campaigns", lambda _: 2)
    assert client.get(f"/v1/submissions/{HASH}/patch").status_code == 409
    assert (
        client.get(BASE.replace(HASH, "sha256:missing") + "/patch").status_code == 404
    )
    response = client.get(BASE.replace(HASH, "sha256:missing") + "/patch-availability")
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert publications == []
