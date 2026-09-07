"""Delayed API disclosure of permanent public patch URLs."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import config
from api import server
from storage.visibility import patch_is_revealed

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
        server, "patch_is_revealed", lambda ts: patch_is_revealed(ts, now=NOW)
    )
    monkeypatch.setattr(
        server,
        "list_patch_evaluation_times",
        lambda ids: {sid: times[sid] for sid in ids if sid in times},
    )
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


@pytest.mark.parametrize("state", ["committed", "building", "scored", "disqualified"])
@pytest.mark.parametrize("age", [None, timedelta(hours=5, minutes=59)])
def test_every_route_withholds_until_evaluated_and_delay_elapsed(
    scenario, monkeypatch, age, state
):
    client, times, original, _ = scenario
    monkeypatch.setattr(server, "list_latest_states", lambda _: {SID: state})
    if age is not None:
        times[SID] = NOW - age
    for path in (BASE, f"/v1/submissions/{HASH}"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        payload = response.json()
        assert payload["submission"]["retrieval_url"] == ""
        assert payload["submission"]["patch_download_url"] is None
        assert payload["submission"]["patch_hash"] == HASH
        assert (
            payload["events"][0]["detail"]["build_log_tail"] == "public compiler output"
        )
        assert payload["jobs"][0]["last_error"] == "public diagnostics"
        download = client.get(path + "/patch", follow_redirects=False)
        assert download.status_code == 403
        assert download.headers["cache-control"] == "no-store"
        assert RAW_URL not in download.text
    listing = client.get(f"/v1/campaigns/{CID}/submissions")
    assert listing.json()["submissions"][0]["retrieval_url"] == ""
    assert listing.headers["cache-control"] == "no-store"
    assert original["retrieval_url"] == RAW_URL


@pytest.mark.parametrize("age", [timedelta(hours=6), timedelta(days=20)])
def test_release_returns_original_permanent_url_and_delay_is_retrospective(
    scenario, monkeypatch, age
):
    client, times, _, _ = scenario
    times[SID] = NOW - age
    for path in (BASE, f"/v1/submissions/{HASH}"):
        submission = client.get(path).json()["submission"]
        assert submission["retrieval_url"] == RAW_URL
        assert (
            submission["patch_reveal_at"]
            == (times[SID] + timedelta(hours=6)).isoformat()
        )
        download = client.get(path + "/patch", follow_redirects=False)
        assert download.status_code == 307
        assert download.headers["location"] == RAW_URL
        assert download.headers["cache-control"] == "no-store"
        assert (
            client.get(
                submission["patch_download_url"], follow_redirects=False
            ).headers["location"]
            == RAW_URL
        )
    listing = client.get(f"/v1/campaigns/{CID}/submissions").json()
    assert listing["submissions"][0]["retrieval_url"] == RAW_URL
    monkeypatch.setattr(config, "PATCH_REVEAL_DELAY_S", int(age.total_seconds()) + 1)
    assert client.get(BASE).json()["submission"]["retrieval_url"] == ""
    assert client.get(BASE + "/patch", follow_redirects=False).status_code == 403
    monkeypatch.setattr(config, "PATCH_REVEAL_DELAY_S", 0)
    assert client.get(BASE).json()["submission"]["retrieval_url"] == RAW_URL


def test_existing_submissions_keep_immediate_visibility(scenario, monkeypatch):
    client, times, _, _ = scenario
    times.clear()
    monkeypatch.setattr(config, "PATCH_REVEAL_DELAY_S", 365 * 86400)
    for path in (BASE, f"/v1/submissions/{HASH}"):
        response = client.get(path)
        assert response.headers["cache-control"] == server.V1_CACHE_CONTROL
        submission = response.json()["submission"]
        assert submission["retrieval_url"] == RAW_URL
        assert submission["patch_reveal_at"] is None
        assert (
            client.get(path + "/patch", follow_redirects=False).headers["location"]
            == RAW_URL
        )
    assert (
        client.get(f"/v1/campaigns/{CID}/submissions").json()["submissions"][0][
            "retrieval_url"
        ]
        == RAW_URL
    )
    assert (
        client.get(f"/v1/campaigns/{CID}/submissions").headers["cache-control"]
        == server.V1_CACHE_CONTROL
    )


def test_mixed_listing_withholds_only_new_submissions(scenario, monkeypatch):
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
    submissions = response.json()["submissions"]
    assert [s["retrieval_url"] for s in submissions] == [RAW_URL, ""]


def test_error_metadata_cannot_reveal_the_withheld_url(scenario):
    client, times, _, events = scenario
    detail = {
        "retrieval_url": RAW_URL,
        "error": f"fetch failed for {RAW_URL}: timed out",
        "nested": [{"source": RAW_URL, "code": 504}],
    }
    events[0]["detail"] = detail
    for path in (BASE, f"/v1/submissions/{HASH}"):
        response = client.get(path)
        assert RAW_URL not in response.text
        assert response.json()["events"][0]["detail"]["error"].endswith(": timed out")
        assert response.json()["events"][0]["detail"]["nested"][0]["code"] == 504
    assert detail["retrieval_url"] == RAW_URL  # Never mutate stored audit events.
    times[SID] = NOW - timedelta(hours=6)
    assert client.get(BASE).json()["events"][0]["detail"] == detail
    times.clear()
    assert client.get(BASE).json()["events"][0]["detail"] == detail


def test_patch_routes_preserve_missing_and_ambiguous_lookup_behavior(
    scenario, monkeypatch
):
    client, _, _, _ = scenario
    assert client.get(BASE.replace(CID, "other") + "/patch").status_code == 404
    monkeypatch.setattr(server, "count_submission_campaigns", lambda _: 2)
    assert client.get(f"/v1/submissions/{HASH}/patch").status_code == 409


@pytest.mark.parametrize("age", [None, timedelta(hours=6)])
def test_json_routes_reuse_loaded_visibility_without_extra_lookup(
    scenario, monkeypatch, age
):
    client, times, _, _ = scenario
    if age is not None:
        times[SID] = NOW - age

    def extra_lookup(_):
        pytest.fail("JSON routes must reuse their existing database reads")

    monkeypatch.setattr(server, "list_patch_evaluation_times", extra_lookup)
    for path in (BASE, f"/v1/submissions/{HASH}", f"/v1/campaigns/{CID}/submissions"):
        response = client.get(path)
        assert response.status_code == 200
        assert "_patch_evaluated_at" not in response.text
        assert "_patch_reveal_delayed" not in response.text
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "route",
    [
        BASE,
        f"/v1/submissions/{HASH}",
        f"/v1/campaigns/{CID}/submissions",
        BASE + "/patch",
        f"/v1/submissions/{HASH}/patch",
    ],
)
def test_private_patch_is_copied_only_after_reveal_and_returns_permanent_url(
    scenario, monkeypatch, route
):
    from types import SimpleNamespace
    from storage import s3

    client, times, row, _ = scenario
    row["retrieval_url"] = RAW_URL.replace("/campaigns/", "/private/campaigns/")
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "")
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "")
    monkeypatch.setattr(config, "S3_BUCKET", "pareton-s3")
    monkeypatch.setattr(config, "S3_REGION", "us-east-2")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    copies = []

    def head(**kwargs):
        from botocore.exceptions import ClientError

        if "/private/" not in kwargs["Key"]:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {
            "ChecksumSHA256": s3._checksum(HASH),
            "ContentLength": 5,
            "ETag": '"etag"',
        }

    monkeypatch.setattr(
        s3,
        "_client",
        lambda **_: SimpleNamespace(
            head_object=head,
            copy_object=lambda **k: copies.append(k),
        ),
    )
    s3.publish_patch.cache_clear()
    try:
        response = client.get(route, follow_redirects=False)
        assert response.status_code == (403 if route.endswith("/patch") else 200)
        assert copies == []
        times[SID] = NOW - timedelta(hours=6)
        response = client.get(route, follow_redirects=False)
        assert len(copies) == 1
        if route.endswith("/patch"):
            assert response.status_code == 307
            assert response.headers["location"] == RAW_URL
        else:
            payload = response.json()
            result = (
                payload["submissions"][0]
                if "submissions" in payload
                else payload["submission"]
            )
            assert result["retrieval_url"] == RAW_URL
        client.get(route, follow_redirects=False)
        assert len(copies) == 1
    finally:
        s3.publish_patch.cache_clear()


def test_private_patch_without_enrollment_fails_closed(scenario, monkeypatch):
    client, times, row, _ = scenario
    times.clear()
    row["retrieval_url"] = RAW_URL.replace("/campaigns/", "/private/campaigns/")
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "")
    monkeypatch.setattr(
        server, "publish_patch", lambda *a: pytest.fail("unrevealed copy")
    )
    evaluated = NOW - timedelta(hours=6)
    monkeypatch.setattr(
        server,
        "list_submission_round_entries",
        lambda _: {SID: {"_patch_evaluated_at": evaluated}},
    )
    monkeypatch.setattr(
        server,
        "list_campaign_submissions",
        lambda *a, **k: {
            "total": 1,
            "items": [
                {
                    **row,
                    "_patch_evaluated_at": evaluated,
                    "_patch_reveal_delayed": False,
                }
            ],
        },
    )
    assert client.get(BASE).json()["submission"]["retrieval_url"] == ""
    assert (
        client.get(f"/v1/submissions/{HASH}").json()["submission"]["retrieval_url"]
        == ""
    )
    assert (
        client.get(f"/v1/campaigns/{CID}/submissions").json()["submissions"][0][
            "retrieval_url"
        ]
        == ""
    )
    assert client.get(BASE + "/patch", follow_redirects=False).status_code == 403


def test_copy_failure_does_not_disclose_a_broken_public_link(scenario, monkeypatch):
    client, times, _, _ = scenario
    times[SID] = NOW - timedelta(hours=6)

    def failure(*args):
        raise RuntimeError("storage failed")

    monkeypatch.setattr(server, "publish_patch", failure)
    for path in (BASE, BASE + "/patch"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
        assert RAW_URL not in response.text
