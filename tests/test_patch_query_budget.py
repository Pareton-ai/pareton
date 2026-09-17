"""Exercise real store calls while counting database reads without a database."""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from api import server
from campaign import store as campaign_store
from round import store as round_store

CID = "11111111-1111-1111-1111-111111111111"
SID = "22222222-2222-2222-2222-222222222222"
HASH = "sha256:" + "a" * 64
URL = "https://pareton-s3.s3.us-east-2.amazonaws.com/stage0/campaigns/c/patches/h/random.diff"


@pytest.mark.parametrize("delayed", [False, True])
@pytest.mark.parametrize("endpoint", ["list", "detail", "legacy_detail"])
def test_json_query_counts_stay_at_pre_reveal_budget(monkeypatch, delayed, endpoint):
    row = {
        "id": SID,
        "campaign_id": CID,
        "patch_hash": HASH,
        "hotkey": "h",
        "baseline_commit": "b" * 40,
        "retrieval_url": URL,
        "committed_at": "2026-09-07T00:00:00+00:00",
        "engine_image_ref": None,
    }
    if endpoint == "list":
        path = f"/v1/campaigns/{CID}/submissions"
        replies = [
            {"ok": True, "n": 1},
            [
                {
                    **row,
                    "latest_state": "committed",
                    "round_id": None,
                    "round_ordinal": None,
                    "round_entry_status": None,
                    "round_score": None,
                    "round_disqualify_reason": None,
                    "_patch_reveal_delayed": delayed,
                    "_patch_evaluated_at": None,
                }
            ],
        ]
        expected_queries, expected_checkouts = 2, 1
    else:
        path = f"/v1/campaigns/{CID}/submissions/{HASH}"
        replies = [
            row,
            [
                {
                    "state": "committed",
                    "detail": {"patch_reveal_delayed": True} if delayed else {},
                    "created_at": row["committed_at"],
                }
            ],
            [{"submission_id": SID, "state": "committed"}],
            [],  # jobs
            [],  # round entries
        ]
        expected_queries = expected_checkouts = 5
        if endpoint == "legacy_detail":
            path = f"/v1/submissions/{HASH}"
            replies.insert(1, {"n": 1})  # Existing ambiguity check.
            expected_queries = expected_checkouts = 6

    queued = iter(replies)
    queries = []
    checkouts = []

    class Cursor:
        def execute(self, query, params):
            queries.append(query)
            self.reply = next(queued, None)
            assert self.reply is not None, "unexpected extra database query"

        def fetchone(self):
            return self.reply

        def fetchall(self):
            return self.reply

    class Connection:
        @contextmanager
        def cursor(self, **kwargs):
            yield Cursor()

    @contextmanager
    def connection(*, readonly=False):
        assert readonly
        checkouts.append(True)
        yield Connection()

    monkeypatch.setattr(campaign_store, "db_connection", connection)
    monkeypatch.setattr(round_store, "db_connection", connection)
    response = TestClient(server.app).get(path)
    assert response.status_code == 200
    payload = response.json()
    public = payload["submissions"][0] if endpoint == "list" else payload["submission"]
    assert public["retrieval_url"] == ("" if delayed else URL)
    assert "_patch_evaluated_at" not in response.text
    assert "_patch_reveal_delayed" not in response.text
    assert len(queries) == expected_queries
    assert len(checkouts) == expected_checkouts
