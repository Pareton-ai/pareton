"""Campaign carryover against Postgres, through the e2e test database only."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytestmark = pytest.mark.e2e

from campaign import carryover
from campaign.manifest import build_manifest
from campaign.models import SLA
from campaign.store import (
    append_event,
    insert_campaign,
    insert_profile,
    insert_submission,
)
from db.connection import db_connection
from e2e_db import cleanup_e2e_rows, require_e2e_database_url
from gate.integrity import hash_patch_bytes, patch_fingerprint_bytes
from storage.s3 import _s3_retrieval_url, object_key_for, public_retrieval_url


@pytest.fixture(autouse=True)
def _bind_e2e_database(monkeypatch: pytest.MonkeyPatch):
    url = require_e2e_database_url()
    monkeypatch.setenv("PARETON_DATABASE_URL", url)
    import db.connection as conn

    monkeypatch.setattr(conn, "DATABASE_URL", url)
    monkeypatch.setattr(conn, "_pool", None)
    yield
    cleanup_e2e_rows()


def _campaign(profile_id, status):
    campaign_id = uuid4()
    insert_campaign(
        build_manifest(
            campaign_id=campaign_id,
            profile_id=profile_id,
            baseline_repo="https://github.com/sgl-project/sglang.git",
            baseline_commit="a" * 40,
            base_image_digest="sha256:" + "d" * 64,
            gpu_skus=["B300"],
            workload_trace_sha256="sha256:" + uuid4().hex * 2,
            workload_trace_url="https://cdn.test/trace.json",
            sla=SLA(),
            scoring_config_sha256=None,
            scoring_config_url=None,
            allowed_paths=["python/sglang/**"],
            denied_paths=[],
            priority_metric="throughput",
            success_threshold=">=10% at SLA",
            submission_fee={"amount_tao": "0", "recipient": "5Test"},
            status=status,
        )
    )
    return str(campaign_id)


def _patch(value, comment=""):
    return (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
        f"+x = {value}{comment}\n"
    ).encode()


def test_carries_latest_patches_in_commit_order_once():
    profile_id = insert_profile("e2e", {"fixture": True})
    old, new = _campaign(profile_id, "open"), _campaign(profile_id, "open")
    objects = {}
    # Five submissions; the last duplicates the second after comment stripping.
    for i, data in enumerate(
        [_patch(0), _patch(1), _patch(2), _patch(3), _patch(1, "  # same")]
    ):
        url = _s3_retrieval_url(object_key_for(old, f"hk{i}", str(uuid4())))
        objects[url] = data
        insert_submission(
            campaign_id=old,
            patch_hash=hash_patch_bytes(data),
            hotkey=f"hk{i}",
            baseline_commit="a" * 40,
            retrieval_url=url,
            commit_block=100 + i,
            payment_block=900 + i,
            payment_tx=0,
        )

    def copier(url, *, campaign_id, hotkey, upload_id, patch_hash):
        target = _s3_retrieval_url(object_key_for(campaign_id, hotkey, upload_id))
        objects[target] = objects[url]
        return target

    with pytest.raises(ValueError, match="close the source campaign"):
        carryover.build_plan(old, new, 4, fetcher=objects.__getitem__)
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE campaigns SET status = 'closed' WHERE id = %s", (old,))

    plan = carryover.build_plan(old, new, 4, fetcher=objects.__getitem__)
    assert [(p["hotkey"], p["skip"]) for p in plan] == [
        ("hk1", None),
        ("hk2", None),
        ("hk3", None),
        ("hk4", f"duplicate of {plan[0]['source_submission_id']}"),
    ]
    assert plan == carryover.build_plan(old, new, 4, fetcher=objects.__getitem__)
    kwargs = dict(operator="ops", reason="workload change", evidence_ref="PR")
    first = carryover.apply_plan(plan, old, new, copier=copier, **kwargs)
    assert [r["outcome"].split()[0] for r in first] == ["entered"] * 3
    again = carryover.apply_plan(plan, old, new, copier=copier, **kwargs)
    assert {r["outcome"] for r in again} == {"skipped: already present"}

    with db_connection(readonly=True) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.hotkey, s.commit_block, s.payment_block, s.retrieval_url,
                   j.status, e.detail
            FROM submissions s
            JOIN submission_jobs j ON j.submission_id = s.id
            JOIN submission_events e
              ON e.submission_id = s.id AND e.state = 'committed'
            WHERE s.campaign_id = %s
            ORDER BY s.commit_block
            """,
            (new,),
        )
        rows = cur.fetchall()
    assert [(r[0], r[1], r[2], r[4]) for r in rows] == [
        ("hk1", 101, None, "pending"),
        ("hk2", 102, None, "pending"),
        ("hk3", 103, None, "pending"),
    ]
    assert all(f"/campaigns/{new}/" in r[3] for r in rows)
    assert rows[0][5]["source_submission_id"] == plan[0]["source_submission_id"]
    assert rows[0][5]["patch_fingerprint"] == patch_fingerprint_bytes(_patch(1))
    assert rows[0][5]["source"] == "operator_carryover"
    assert [
        p["skip"]
        for p in carryover.build_plan(old, new, 4, fetcher=objects.__getitem__)
    ] == ["already in target campaign"] * 3 + ["duplicate already in target campaign"]


def test_skips_a_public_patch_it_cannot_copy():
    profile_id = insert_profile("e2e", {"fixture": True})
    old, new = _campaign(profile_id, "closed"), _campaign(profile_id, "open")
    data = _patch(7)
    key = object_key_for(old, "hk", str(uuid4()))
    url = public_retrieval_url(key.replace("/private/", "/", 1))
    insert_submission(
        campaign_id=old,
        patch_hash=hash_patch_bytes(data),
        hotkey="hk",
        baseline_commit="a" * 40,
        retrieval_url=url,
        commit_block=100,
    )
    [item] = carryover.build_plan(old, new, 1, fetcher={url: data}.__getitem__)
    assert item["skip"] == "not a private patch locator"


def test_rechecks_the_source_before_copying():
    profile_id = insert_profile("e2e", {"fixture": True})
    old, new = _campaign(profile_id, "closed"), _campaign(profile_id, "open")
    objects, ids = {}, []
    for i in range(2):
        data = _patch(10 + i)
        url = _s3_retrieval_url(object_key_for(old, f"hk{i}", str(uuid4())))
        objects[url] = data
        ids.append(
            insert_submission(
                campaign_id=old,
                patch_hash=hash_patch_bytes(data),
                hotkey=f"hk{i}",
                baseline_commit="a" * 40,
                retrieval_url=url,
                commit_block=100 + i,
            )
        )
    plan = carryover.build_plan(old, new, 2, fetcher=objects.__getitem__)
    assert [p["skip"] for p in plan] == [None, None]
    # Both change after the reviewed plan was built.
    append_event(ids[0], "rejected")
    append_event(
        ids[1],
        "disqualified",
        detail={"source": "manual", "scope": "campaign", "action": "disqualified"},
    )
    copied = []

    def copier(url, **kwargs):
        copied.append(url)
        return url

    results = carryover.apply_plan(
        plan, old, new, copier=copier, operator="ops", reason="r", evidence_ref="PR"
    )
    assert [r["outcome"] for r in results] == [
        "skipped: source state rejected",
        "skipped: hotkey disqualified in source campaign",
    ]
    assert copied == []


def test_carries_only_the_named_submissions():
    profile_id = insert_profile("e2e", {"fixture": True})
    old, new = _campaign(profile_id, "closed"), _campaign(profile_id, "open")
    objects, ids = {}, []
    for i in range(4):
        data = _patch(20 + i)
        url = _s3_retrieval_url(object_key_for(old, f"hk{i}", str(uuid4())))
        objects[url] = data
        ids.append(
            insert_submission(
                campaign_id=old,
                patch_hash=hash_patch_bytes(data),
                hotkey=f"hk{i}",
                baseline_commit="a" * 40,
                retrieval_url=url,
                commit_block=100 + i,
            )
        )
    plan = carryover.build_plan(
        old, new, submission_ids=[ids[2], ids[0]], fetcher=objects.__getitem__
    )
    # Oldest first, whatever order the ids were given in.
    assert [p["hotkey"] for p in plan] == ["hk0", "hk2"]
    with pytest.raises(ValueError, match="not in the source campaign"):
        carryover.build_plan(
            old, new, submission_ids=[ids[1], str(uuid4())], fetcher=objects.__getitem__
        )
