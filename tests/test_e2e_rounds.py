"""Round creation and stale-round reaping against the Neon test branch."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

import config
from campaign.manifest import build_manifest
from campaign.models import SLA
from campaign.store import (
    get_campaign,
    insert_campaign,
    insert_profile,
    insert_submission,
)
from db.connection import db_connection
from e2e_db import cleanup_e2e_rows, require_e2e_database_url
from round.create import create_due_rounds, try_create_round
from round.store import campaigns_with_queue, reap_stale_rounds

pytestmark = pytest.mark.e2e

ENGINE_DIGEST = "sha256:" + "a" * 64
IMAGE_A = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "1" * 64
IMAGE_B = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "2" * 64
IMAGE_C = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "3" * 64


def _image(i: int) -> str:
    """A distinct digest per index, so dedupe keeps every row."""
    return f"ghcr.io/pareton-ai/pareton-engine@sha256:{i:064x}"


@pytest.fixture(autouse=True)
def _bind_e2e_database(monkeypatch: pytest.MonkeyPatch):
    """Point store/connection code at the Neon test branch for this module."""
    url = require_e2e_database_url()
    monkeypatch.setenv("PARETON_DATABASE_URL", url)
    import db.connection as conn

    monkeypatch.setattr(conn, "DATABASE_URL", url)
    monkeypatch.setattr(conn, "_pool", None)
    yield
    cleanup_e2e_rows()


class _FakeSubtensor:
    """Head block plus historical hashes; no chain, no sleeping on the tip."""

    block = 1000

    def get_block_hash(self, number: int) -> str:
        return "0x" + f"{number:064x}"


def _campaign(**over) -> UUID:
    profile_id = insert_profile("e2e", {"fixture": True})
    campaign_id = uuid4()
    fields = {
        "campaign_id": campaign_id,
        "profile_id": profile_id,
        "baseline_repo": "https://example/baseline.git",
        "baseline_commit": "deadbeef",
        "base_image_digest": "sha256:" + "d" * 64,
        "gpu_skus": ["H200", "B200"],
        "workload_trace_sha256": "sha256:" + "e" * 64,
        "workload_trace_url": "https://cdn.test/trace.json",
        "sla": SLA(),
        "scoring_config_sha256": None,
        "scoring_config_url": None,
        "allowed_paths": ["vllm/**"],
        "denied_paths": ["tests/**"],
        "priority_metric": "throughput",
        "success_threshold": ">=10% at SLA",
        "status": "open",
        "bench": {"baseline_engine_image_digest": ENGINE_DIGEST},
        "scoring_rule": {"name": "median_e2e_speedup"},
    }
    fields.update(over)
    insert_campaign(build_manifest(**fields))
    return campaign_id


def _submission(campaign_id: UUID, *, image_ref: str, block: int) -> str:
    """Insert a built submission that is not in the round queue."""
    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash="sha256:" + uuid4().hex * 2,
        hotkey="5FakesHotkeyForE2ETesting000000000000000000000",
        baseline_commit="deadbeef",
        retrieval_url=f"https://cdn.test/stage0/campaigns/{campaign_id}/p/e2e.diff",
        commit_block=block,
    )
    assert sid is not None
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE submissions SET engine_image_ref = %s WHERE id = %s",
                (image_ref, str(sid)),
            )
    return str(sid)


def _queued(campaign_id: UUID, *, image_ref: str, block: int, waited_s: int = 0) -> str:
    """Insert a submission and put it in the round queue, aged by waited_s."""
    sid = _submission(campaign_id, image_ref=image_ref, block=block)
    with db_connection() as conn:
        with conn.cursor() as cur:
            # The gate events came first in real time; age them with the queue
            # event so 'bench_queued' stays the newest state.
            cur.execute(
                """
                UPDATE submission_events
                SET created_at = now() - make_interval(secs => %s)
                WHERE submission_id = %s
                """,
                (int(waited_s) + 60, sid),
            )
            cur.execute(
                """
                INSERT INTO submission_events (submission_id, state, detail, created_at)
                VALUES (%s, 'bench_queued', '{}'::jsonb,
                        now() - make_interval(secs => %s))
                """,
                (sid, int(waited_s)),
            )
    return sid


def _latest_state(submission_id: str) -> tuple[str, dict]:
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT state, detail FROM submission_events
                WHERE submission_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (submission_id,),
            )
            row = cur.fetchone()
    return str(row[0]), dict(row[1])


def _rounds(campaign_id: UUID) -> list[dict]:
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ordinal, status, gpu_sku, seed_block, seed_block_hash,
                       sampled_trace_sha256, scoring_rule, void_reason,
                       incumbent_submission_id
                FROM rounds WHERE campaign_id = %s ORDER BY ordinal
                """,
                (str(campaign_id),),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def _entries(round_id: str) -> list[tuple]:
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, submission_id, engine_image_ref FROM round_entries
                WHERE round_id = %s ORDER BY id
                """,
                (round_id,),
            )
            return [(r[0], str(r[1]) if r[1] else None, r[2]) for r in cur.fetchall()]


def test_aged_cohort_of_three_creates_a_round_of_three():
    campaign_id = _campaign()
    sids = [
        _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000),
        _queued(campaign_id, image_ref=IMAGE_B, block=11, waited_s=39_000),
        _queued(campaign_id, image_ref=IMAGE_C, block=12, waited_s=38_000),
    ]

    create_due_rounds(_FakeSubtensor())

    (rnd,) = _rounds(campaign_id)
    assert rnd["status"] == "pending"
    assert rnd["ordinal"] == 1
    assert rnd["gpu_sku"] == "H200"
    assert rnd["seed_block"] == 1000 - 1
    assert rnd["scoring_rule"] == {"name": "median_e2e_speedup"}
    assert rnd["sampled_trace_sha256"] == "sha256:" + "e" * 64

    entries = _entries(str(rnd["id"]))
    assert entries[0] == (
        "baseline",
        None,
        f"ghcr.io/pareton-ai/pareton-engine@{ENGINE_DIGEST}",
    )
    assert [(role, sid) for role, sid, _ref in entries[1:]] == [
        ("challenger", sid) for sid in sids
    ]
    for sid in sids:
        state, detail = _latest_state(sid)
        assert state == "round_assigned"
        assert detail["ordinal"] == 1

    # The queue is spent: a second pass finds nothing to do.
    assert str(campaign_id) not in {
        str(q["campaign_id"]) for q in campaigns_with_queue()
    }


def test_racing_creators_produce_one_live_round():
    """A leftover queue must not let the loser seat newer submissions.

    The queue is deliberately longer than ROUND_SIZE: without the campaign
    lock the second creator takes the rows the first one left behind, and the
    cohort stops being the oldest by commit_block.
    """
    campaign_id = _campaign()
    assert config.ROUND_SIZE == 5
    sids = [
        _queued(campaign_id, image_ref=_image(i), block=10 + i, waited_s=40_000)
        for i in range(config.ROUND_SIZE + 2)
    ]
    campaign = get_campaign(campaign_id)
    queue = {
        "queued": len(sids),
        "oldest_queued_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
    }
    start = threading.Barrier(2)
    outcomes: list = []

    def _create():
        start.wait()
        outcomes.append(
            try_create_round(
                campaign,
                dict(queue),
                seed_block=999,
                seed_block_hash="0x" + "ab" * 32,
            )
        )

    threads = [threading.Thread(target=_create) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert len([o for o in outcomes if o is not None]) == 1
    (rnd,) = _rounds(campaign_id)
    challengers = [
        sid for role, sid, _ref in _entries(str(rnd["id"])) if role == "challenger"
    ]
    assert challengers == sids[: config.ROUND_SIZE]
    for sid in sids[: config.ROUND_SIZE]:
        assert _latest_state(sid)[0] == "round_assigned"
    for sid in sids[config.ROUND_SIZE :]:
        assert _latest_state(sid)[0] == "bench_queued"


def test_stale_round_voids_and_requeues_challengers_only():
    campaign_id = _campaign()
    sids = [
        _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000),
        _queued(campaign_id, image_ref=IMAGE_B, block=11, waited_s=39_000),
    ]
    dq_sid = _queued(campaign_id, image_ref=_image(9), block=12, waited_s=38_000)
    if_sid = _queued(campaign_id, image_ref=_image(8), block=13, waited_s=37_000)
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    create_due_rounds(_FakeSubtensor())
    (rnd,) = _rounds(campaign_id)
    round_id = str(rnd["id"])

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO round_entries (
                  round_id, submission_id, role, engine_image_ref
                ) VALUES (%s, %s, 'leader', %s)
                """,
                (round_id, leader_sid, IMAGE_C),
            )
            cur.execute(
                """
                UPDATE round_entries SET status = 'disqualified'
                WHERE round_id = %s AND submission_id = %s
                """,
                (round_id, dq_sid),
            )
            cur.execute(
                """
                UPDATE round_entries SET status = 'infra_failed'
                WHERE round_id = %s AND submission_id = %s
                """,
                (round_id, if_sid),
            )
            cur.execute(
                """
                INSERT INTO leaders (
                  campaign_id, submission_id, engine_image_ref, hotkey,
                  won_at_round_id, won_at_ordinal, last_score
                ) VALUES (%s, %s, %s, 'hk', %s, 1, 0.42)
                """,
                (str(campaign_id), leader_sid, IMAGE_C, round_id),
            )
            cur.execute(
                """
                UPDATE rounds
                SET status = 'running',
                    started_at = now() - interval '3 hours',
                    heartbeat_at = now() - interval '2 hours'
                WHERE id = %s
                """,
                (round_id,),
            )
            cur.execute(
                "SELECT to_jsonb(leaders.*) FROM leaders WHERE campaign_id = %s",
                (str(campaign_id),),
            )
            before = cur.fetchone()[0]
    leader_state_before = _latest_state(leader_sid)
    dq_state_before = _latest_state(dq_sid)

    voided = reap_stale_rounds(1800)

    assert [str(r["id"]) for r in voided] == [round_id]
    (after_round,) = _rounds(campaign_id)
    assert after_round["status"] == "void"
    assert after_round["void_reason"] == "heartbeat_stale"
    assert after_round["ordinal"] == 1
    for sid in sids:
        state, detail = _latest_state(sid)
        assert state == "bench_queued"
        assert detail["void_reason"] == "heartbeat_stale"
    # infra_failed is not terminal: the void is its one requeue.
    state, detail = _latest_state(if_sid)
    assert state == "bench_queued"
    assert detail["void_reason"] == "heartbeat_stale"
    assert _latest_state(leader_sid) == leader_state_before
    assert _latest_state(dq_sid) == dq_state_before

    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_jsonb(leaders.*) FROM leaders WHERE campaign_id = %s",
                (str(campaign_id),),
            )
            assert cur.fetchone()[0] == before


def test_leader_is_seated_as_an_entry_and_is_not_requeued():
    """The incumbent runs every round, but is not a queued challenger."""
    campaign_id = _campaign()
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    (first,) = _rounds(campaign_id)

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE rounds SET status = 'complete', completed_at = now() "
                "WHERE id = %s",
                (str(first["id"]),),
            )
            cur.execute(
                """
                INSERT INTO leaders (
                  campaign_id, submission_id, engine_image_ref, hotkey,
                  won_at_round_id, won_at_ordinal, last_score
                ) VALUES (%s, %s, %s, 'hk', %s, 1, 0.42)
                """,
                (str(campaign_id), leader_sid, IMAGE_C, str(first["id"])),
            )
    leader_state_before = _latest_state(leader_sid)

    challenger = _queued(campaign_id, image_ref=IMAGE_B, block=20, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())

    _first, second = _rounds(campaign_id)
    assert second["ordinal"] == 2
    assert str(second["incumbent_submission_id"]) == leader_sid
    assert _entries(str(second["id"])) == [
        ("baseline", None, f"ghcr.io/pareton-ai/pareton-engine@{ENGINE_DIGEST}"),
        ("leader", leader_sid, IMAGE_C),
        ("challenger", challenger, IMAGE_B),
    ]
    # The leader is not queued work, so it gets no round_assigned event.
    assert _latest_state(leader_sid) == leader_state_before


def test_duplicate_image_digest_is_rejected():
    campaign_id = _campaign()
    kept = _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    dupe = _queued(campaign_id, image_ref=IMAGE_A.upper(), block=11, waited_s=39_000)

    create_due_rounds(_FakeSubtensor())

    (rnd,) = _rounds(campaign_id)
    assert [(role, sid) for role, sid, _ref in _entries(str(rnd["id"]))] == [
        ("baseline", None),
        ("challenger", kept),
    ]
    assert _latest_state(kept)[0] == "round_assigned"
    state, detail = _latest_state(dupe)
    assert state == "rejected"
    assert detail["reason"] == "duplicate_image"
    assert detail["kept_submission_id"] == kept
