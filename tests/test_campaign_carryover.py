"""Offline tests for the campaign carryover CLI."""

from __future__ import annotations

from uuid import UUID, uuid5

import pytest

pytestmark = pytest.mark.unit

from campaign import carryover

OLD = "11111111-1111-4111-8111-111111111111"
NEW = "22222222-2222-4222-8222-222222222222"
ARGS = [
    "--from-campaign",
    OLD,
    "--to-campaign",
    NEW,
    "--last",
    "2",
    "--operator",
    "ops",
    "--reason",
    "workload change",
    "--evidence-ref",
    "PR",
]


def _plan():
    return [
        {
            "source_submission_id": "33333333-3333-4333-8333-333333333333",
            "hotkey": "5Alice",
            "patch_hash": "sha256:" + "a" * 64,
            "baseline_commit": "b" * 40,
            "retrieval_url": "https://bucket/old.diff",
            "commit_block": 42,
            "source_state": "scored",
            "patch_fingerprint": "sha256:" + "c" * 64,
            "skip": None,
        },
        {
            "source_submission_id": "44444444-4444-4444-8444-444444444444",
            "hotkey": "5Bob",
            "patch_hash": "sha256:" + "d" * 64,
            "baseline_commit": "b" * 40,
            "retrieval_url": "https://bucket/dup.diff",
            "commit_block": 43,
            "source_state": "rejected_duplicate",
            "patch_fingerprint": None,
            "skip": "source state rejected_duplicate",
        },
    ]


@pytest.fixture
def calls(monkeypatch):
    seen = {"copies": [], "inserts": []}
    monkeypatch.setattr(carryover, "build_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(carryover, "_recheck", lambda *a: None)

    def copy(url, **kwargs):
        seen["copies"].append((url, kwargs))
        return "https://bucket/new.diff"

    def insert(**kwargs):
        seen["inserts"].append(kwargs)
        return "55555555-5555-4555-8555-555555555555"

    monkeypatch.setattr(carryover, "copy_private_patch", copy)
    monkeypatch.setattr(carryover, "insert_submission", insert)
    return seen


def test_dry_run_writes_nothing_and_prints_plan_hash(calls, capsys):
    assert carryover.main(ARGS) == 0
    out = capsys.readouterr().out
    assert "plan: 1 to carry, 1 skipped" in out
    assert f"plan_sha256: {carryover.plan_sha256(_plan())}" in out
    assert "dry run; nothing written" in out
    assert calls == {"copies": [], "inserts": []}


def test_apply_requires_the_reviewed_plan(calls, capsys):
    with pytest.raises(SystemExit):
        carryover.main([*ARGS, "--apply"])
    assert carryover.main([*ARGS, "--apply", "--plan-sha256", "sha256:0"]) == 1
    assert "plan changed" in capsys.readouterr().err
    assert calls == {"copies": [], "inserts": []}


def test_apply_copies_and_enters_only_kept_rows(calls):
    digest = carryover.plan_sha256(_plan())
    assert carryover.main([*ARGS, "--apply", "--plan-sha256", digest]) == 0
    [(url, copy)] = calls["copies"]
    assert url == "https://bucket/old.diff"
    assert copy["campaign_id"] == NEW
    assert copy["upload_id"] == str(
        uuid5(UUID(NEW), "carryover:33333333-3333-4333-8333-333333333333")
    )
    [insert] = calls["inserts"]
    assert insert["campaign_id"] == NEW
    assert insert["retrieval_url"] == "https://bucket/new.diff"
    # On-chain order is kept; no payment reference is reused.
    assert insert["commit_block"] == 42
    assert "payment_block" not in insert
    assert insert["event_detail"] == {
        "source": "operator_carryover",
        "source_campaign_id": OLD,
        "source_submission_id": "33333333-3333-4333-8333-333333333333",
        "operator": "ops",
        "reason": "workload change",
        "evidence_ref": "PR",
    }


def test_rejects_nonpositive_last():
    with pytest.raises(ValueError, match="positive"):
        carryover.build_plan(OLD, NEW, 0)


def test_needs_exactly_one_selection():
    with pytest.raises(ValueError, match="either --last or --submission-id"):
        carryover.build_plan(OLD, NEW)
    with pytest.raises(ValueError, match="either --last or --submission-id"):
        carryover.build_plan(OLD, NEW, 2, submission_ids=[OLD])
    with pytest.raises(SystemExit):
        carryover.main([a for a in ARGS if a not in ("--last", "2")])


def test_cli_passes_named_submissions(monkeypatch, capsys):
    seen = {}

    def plan(source, target, last, *, submission_ids):
        seen.update(last=last, ids=submission_ids)
        return _plan()

    monkeypatch.setattr(carryover, "build_plan", plan)
    args = [a for a in ARGS if a not in ("--last", "2")]
    ids = [
        "33333333-3333-4333-8333-333333333333",
        "44444444-4444-4444-8444-444444444444",
    ]
    assert (
        carryover.main([*args, "--submission-id", ids[0], "--submission-id", ids[1]])
        == 0
    )
    assert seen == {"last": None, "ids": ids}
