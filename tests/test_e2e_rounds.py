"""Round creation and stale-round reaping against the Neon test branch."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from e2e_db import cleanup_e2e_rows, require_e2e_database_url

import config
from campaign.manifest import build_manifest
from campaign.models import SLA
from campaign.store import (
    CampaignHotkeyDisqualified,
    get_campaign,
    insert_campaign,
    insert_profile,
    insert_submission,
)
from db.connection import db_connection
from round.create import create_due_rounds, try_create_round
from round.rank import EVENT_OVERTAKEN, EVENT_SEATED, EVENT_VACATED, RankDecision
from round.store import (
    VOID_POD_FAILED,
    campaigns_with_queue,
    claim_pending_round,
    complete_round,
    disqualify_campaign_hotkey,
    get_leader,
    list_idle_seated_leaders,
    reap_stale_rounds,
    set_round_phase,
    touch_round_heartbeat,
    update_round_entry_live_status,
    vacate_leader_if_idle,
    void_round,
    waive_campaign_hotkey,
)

pytestmark = pytest.mark.e2e

ENGINE_DIGEST = "sha256:" + "a" * 64
IMAGE_A = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "1" * 64
IMAGE_B = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "2" * 64
IMAGE_C = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "3" * 64
IMAGE_D = "ghcr.io/pareton-ai/pareton-engine@sha256:" + "4" * 64


def _image(i: int) -> str:
    """A distinct digest per index, so dedupe keeps every row."""
    return f"ghcr.io/pareton-ai/pareton-engine@sha256:{i:064x}"


@pytest.fixture(autouse=True)
def _fake_hf_rows(monkeypatch: pytest.MonkeyPatch):
    """Round creation must not hit HuggingFace from the e2e suite."""

    def _fetch(rule, idx):
        return {"trajectory": [{"role": "user", "content": f"prompt-{idx}"}]}

    monkeypatch.setattr("round.create.fetch_hf_row", _fetch)


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
        "workload_trace_sha256": None,
        "workload_trace_url": None,
        "sampling_rule": {
            "type": "hf_rows",
            "dataset": "d",
            "revision": "r",
            "n_rows": 8,
            "n_prompts": 2,
        },
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


def _submission(
    campaign_id: UUID,
    *,
    image_ref: str,
    block: int,
    hotkey: str = "5FakesHotkeyForE2ETesting000000000000000000000",
) -> str:
    """Insert a built submission that is not in the round queue."""
    sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash="sha256:" + uuid4().hex * 2,
        hotkey=hotkey,
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


def _queued(
    campaign_id: UUID,
    *,
    image_ref: str,
    block: int,
    waited_s: int = 0,
    hotkey: str = "5FakesHotkeyForE2ETesting000000000000000000000",
) -> str:
    """Insert a submission and put it in the round queue, aged by waited_s."""
    sid = _submission(campaign_id, image_ref=image_ref, block=block, hotkey=hotkey)
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
    # A DQ on a still-running round is only a live-streamed pod report, never
    # a settlement verdict: the void requeues it like any unsettled entry.
    state, detail = _latest_state(dq_sid)
    assert state == "bench_queued"
    assert detail["void_reason"] == "heartbeat_stale"

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


def _settle_round(
    round_id: str, *, winner_sid: str | None, status: str, **over
) -> None:
    """Close a round the way the runner (PAR-83) will, so the reads have data."""
    fields = {
        "status": status,
        "winner_submission_id": winner_sid,
        "leader_changed": winner_sid is not None,
        "void_reason": over.get("void_reason"),
    }
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rounds
                SET status = %(status)s,
                    winner_submission_id = %(winner_submission_id)s,
                    leader_changed = %(leader_changed)s,
                    void_reason = %(void_reason)s,
                    completed_at = now()
                WHERE id = %(id)s
                """,
                {**fields, "id": round_id},
            )


def _settle_entry(round_id: str, submission_id: str | None, **fields) -> None:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE round_entries
                SET status = %s, score = %s, disqualify_reason = %s
                WHERE round_id = %s
                  AND submission_id IS NOT DISTINCT FROM %s
                """,
                (
                    fields["status"],
                    fields.get("score"),
                    fields.get("disqualify_reason"),
                    round_id,
                    submission_id,
                ),
            )


def test_public_reads_of_rounds_leader_and_score_progress():
    """One complete round, one void round: the shapes the read API serves."""
    from round.store import (
        get_leader,
        get_round,
        list_round_entries,
        list_rounds,
        list_score_progress,
        list_submission_round_entries,
    )

    campaign_id = _campaign()
    winner = _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    loser = _queued(campaign_id, image_ref=IMAGE_B, block=11, waited_s=39_000)
    create_due_rounds(_FakeSubtensor())
    (first,) = _rounds(campaign_id)
    first_id = str(first["id"])

    _settle_entry(first_id, None, status="scored", score=0)
    _settle_entry(first_id, winner, status="scored", score="0.31")
    _settle_entry(
        first_id, loser, status="disqualified", disqualify_reason="fail_correctness"
    )
    _settle_round(first_id, winner_sid=winner, status="complete")
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO leaders (
                  campaign_id, submission_id, engine_image_ref, hotkey,
                  won_at_round_id, won_at_ordinal, last_score,
                  last_scored_round_id
                ) VALUES (%s, %s, %s, %s, %s, 1, 0.31, %s)
                """,
                (
                    str(campaign_id),
                    winner,
                    IMAGE_A,
                    "5FakesHotkeyForE2ETesting000000000000000000000",
                    first_id,
                    first_id,
                ),
            )

    # A second round that voids: it keeps its ordinal and carries no winner.
    voided = _queued(campaign_id, image_ref=IMAGE_C, block=20, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    _first, second = _rounds(campaign_id)
    second_id = str(second["id"])
    _settle_round(
        second_id, winner_sid=None, status="void", void_reason="baseline_drift"
    )

    # A third round left live, so the leader is re-seated while it runs.
    newcomer = _queued(campaign_id, image_ref=IMAGE_D, block=30, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    third_id = str(_rounds(campaign_id)[2]["id"])

    leader = get_leader(campaign_id)
    assert str(leader["submission_id"]) == winner
    assert leader["hotkey"] == "5FakesHotkeyForE2ETesting000000000000000000000"
    assert float(leader["last_score"]) == 0.31
    assert leader["patch_hash"].startswith("sha256:")
    assert get_leader(uuid4()) is None

    page = list_rounds(campaign_id)
    assert page["total"] == 3
    assert [r["ordinal"] for r in page["items"]] == [3, 2, 1]
    assert page["items"][1]["void_reason"] == "baseline_drift"
    assert page["items"][2]["entry_count"] == 3
    assert page["items"][2]["leader_changed"] is True

    detail = get_round(first_id)
    assert detail["scoring_rule"] == {"name": "median_e2e_speedup"}
    assert str(detail["winner_submission_id"]) == winner
    assert get_round(uuid4()) is None
    entries = list_round_entries(first_id)
    assert [(e["role"], e["status"]) for e in entries] == [
        ("baseline", "scored"),
        ("challenger", "scored"),
        ("challenger", "disqualified"),
    ]
    assert entries[0]["submission_id"] is None and entries[0]["hotkey"] is None
    assert float(entries[0]["score"]) == 0.0
    # A disqualified entry has no score. 0.0 would be a lie: it is baseline speed.
    assert entries[2]["score"] is None
    assert entries[2]["disqualify_reason"] == "fail_correctness"
    assert "evidence_s3_url" not in entries[2]

    points = list_score_progress(campaign_id)
    assert [p["ordinal"] for p in points] == [1, 2, 3]
    assert float(points[0]["leader_score"]) == 0.31
    # The void round keeps ordinal 2 and leaves a gap in the line.
    assert points[1]["ordinal"] == 2 and points[1]["leader_score"] is None
    # The winner is the line, so the scatter holds only the other challenger.
    assert [e["submission_id"] for e in points[0]["entries"]] == [loser]
    assert points[0]["entries"][0]["score"] is None
    assert points[0]["entries"][0]["status"] == "disqualified"

    outcomes = list_submission_round_entries([winner, loser, voided, newcomer])
    # The winner sits live in round 3 and voided out of round 2, but the API
    # reports the scored entry it won round 1 with, not a fresh pending.
    assert outcomes[winner]["round_id"] == first_id
    assert outcomes[winner]["ordinal"] == 1
    assert outcomes[winner]["status"] == "scored"
    assert float(outcomes[winner]["score"]) == 0.31
    assert outcomes[loser]["ordinal"] == 1
    assert outcomes[loser]["disqualify_reason"] == "fail_correctness"
    # A void round changed no submission state, so its only entry surfaces
    # nothing at all.
    assert voided not in outcomes
    # A submission whose only entry is live still reports that live assignment.
    assert outcomes[newcomer]["round_id"] == third_id
    assert outcomes[newcomer]["ordinal"] == 3
    assert outcomes[newcomer]["status"] == "pending"
    assert outcomes[newcomer]["score"] is None
    assert list_submission_round_entries([]) == {}


def _runner_entries(round_id: str) -> list[dict]:
    from psycopg2.extras import RealDictCursor

    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT e.id, e.submission_id, e.role, e.engine_image_ref, s.hotkey
                FROM round_entries e
                LEFT JOIN submissions s ON s.id = e.submission_id
                WHERE e.round_id = %s
                ORDER BY e.id
                """,
                (round_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def _result(
    row: dict, *, status: str, score: float | None, reason: str | None = None
) -> dict:
    return {
        "id": row["id"],
        "submission_id": (
            str(row["submission_id"]) if row["submission_id"] is not None else None
        ),
        "role": row["role"],
        "engine_image_ref": row["engine_image_ref"],
        "hotkey": row["hotkey"],
        "status": status,
        "score": score,
        "disqualify_reason": reason,
        "report": {"status": status, "score": score},
    }


def _seed_incumbent(campaign_id: UUID, sid: str, image_ref: str) -> str:
    """Insert a placeholder complete round so leaders.won_at_round_id can bind."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rounds (
                  campaign_id, ordinal, gpu_sku, seed_block, seed_block_hash,
                  seed_hex, sampled_trace_sha256, sampling_receipt, scoring_rule,
                  status, completed_at
                ) VALUES (
                  %s, 0, 'H200', 1, '0x00', '00', %s, '{}'::jsonb,
                  '{"name":"median_e2e_speedup"}'::jsonb, 'complete', now()
                )
                RETURNING id
                """,
                (str(campaign_id), "sha256:" + "e" * 64),
            )
            rid = str(cur.fetchone()[0])
            cur.execute("SELECT hotkey FROM submissions WHERE id = %s", (sid,))
            hotkey = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO leaders (
                  campaign_id, submission_id, engine_image_ref, hotkey,
                  won_at_round_id, won_at_ordinal, last_score
                ) VALUES (%s, %s, %s, %s, %s, 0, 0.20)
                """,
                (str(campaign_id), sid, image_ref, hotkey, rid),
            )
    return rid


def _history(campaign_id: UUID) -> list[tuple]:
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event, new_submission_id, prev_submission_id
                FROM leader_history WHERE campaign_id = %s ORDER BY id
                """,
                (str(campaign_id),),
            )
            return [
                (r[0], str(r[1]) if r[1] else None, str(r[2]) if r[2] else None)
                for r in cur.fetchall()
            ]


def test_claim_sets_started_at_and_heartbeat_at():
    campaign_id = _campaign()
    _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["started_at"] is not None
    assert claimed["heartbeat_at"] is not None
    assert str(claimed["campaign_id"]) == str(campaign_id)
    assert claim_pending_round() is None


def test_phase_and_heartbeat_land_on_a_running_round():
    campaign_id = _campaign()
    _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    rid = str(claimed["id"])
    assert set_round_phase(round_id=rid, phase="provisioning", progress={"n": 1})
    assert touch_round_heartbeat(round_id=rid)
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, phase, progress, heartbeat_at FROM rounds WHERE id = %s",
                (rid,),
            )
            status, phase, progress, heartbeat_at = cur.fetchone()
    assert status == "running"
    assert phase == "provisioning"
    assert progress == {"n": 1}
    assert heartbeat_at is not None


def test_complete_hold_writes_winner_submission_id():
    campaign_id = _campaign()
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    _seed_incumbent(campaign_id, leader_sid, IMAGE_C)
    challenger = _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    rid = str(claimed["id"])
    rows = _runner_entries(rid)
    by_role = {r["role"]: r for r in rows}
    results = [
        _result(by_role["baseline"], status="scored", score=0.0),
        _result(by_role["leader"], status="scored", score=0.20),
        _result(by_role["challenger"], status="scored", score=0.201),
    ]
    decision = RankDecision(
        leader_submission_id=leader_sid,
        leader_score=0.20,
        overtake_threshold=0.202,
    )
    assert decision.leader_changed is False
    assert complete_round(
        round_id=rid,
        campaign_id=campaign_id,
        ordinal=int(claimed["ordinal"]),
        decision=decision,
        entries=results,
        baseline_drift=0.001,
        epsilon=0.01,
    )
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, winner_submission_id, leader_changed, baseline_drift
                FROM rounds WHERE id = %s
                """,
                (rid,),
            )
            status, winner, changed, drift = cur.fetchone()
    assert status == "complete"
    assert str(winner) == leader_sid
    assert changed is False
    assert float(drift) == pytest.approx(0.001)
    assert _history(campaign_id) == []
    seated = get_leader(campaign_id)
    assert seated is not None
    assert str(seated["submission_id"]) == leader_sid
    assert _latest_state(challenger)[0] == "scored"


def test_complete_overtake_writes_leaders_and_history_together():
    campaign_id = _campaign()
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    _seed_incumbent(campaign_id, leader_sid, IMAGE_C)
    challenger = _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    rid = str(claimed["id"])
    rows = _runner_entries(rid)
    by_role = {r["role"]: r for r in rows}
    results = [
        _result(by_role["baseline"], status="scored", score=0.0),
        _result(by_role["leader"], status="scored", score=0.20),
        _result(by_role["challenger"], status="scored", score=0.30),
    ]
    decision = RankDecision(
        event=EVENT_OVERTAKEN,
        leader_submission_id=challenger,
        leader_score=0.30,
        prev_submission_id=leader_sid,
        prev_score=0.20,
        overtake_threshold=0.202,
    )
    assert complete_round(
        round_id=rid,
        campaign_id=campaign_id,
        ordinal=int(claimed["ordinal"]),
        decision=decision,
        entries=results,
        baseline_drift=0.0,
        epsilon=0.01,
    )
    seated = get_leader(campaign_id)
    assert seated is not None
    assert str(seated["submission_id"]) == challenger
    assert float(seated["last_score"]) == pytest.approx(0.30)
    assert str(seated["won_at_round_id"]) == rid
    assert _history(campaign_id) == [(EVENT_OVERTAKEN, challenger, leader_sid)]
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT winner_submission_id, leader_changed FROM rounds WHERE id = %s",
                (rid,),
            )
            winner, changed = cur.fetchone()
    assert str(winner) == challenger
    assert changed is True


def test_complete_vacate_deletes_leader_and_writes_history():
    campaign_id = _campaign()
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    _seed_incumbent(campaign_id, leader_sid, IMAGE_C)
    challenger = _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    rid = str(claimed["id"])
    rows = _runner_entries(rid)
    by_role = {r["role"]: r for r in rows}
    results = [
        _result(by_role["baseline"], status="scored", score=0.0),
        _result(
            by_role["leader"],
            status="disqualified",
            score=None,
            reason="fail_correctness",
        ),
        _result(
            by_role["challenger"], status="disqualified", score=None, reason="fail"
        ),
    ]
    decision = RankDecision(
        event=EVENT_VACATED,
        prev_submission_id=leader_sid,
        prev_score=0.20,
        overtake_threshold=0.0,
    )
    assert complete_round(
        round_id=rid,
        campaign_id=campaign_id,
        ordinal=int(claimed["ordinal"]),
        decision=decision,
        entries=results,
        baseline_drift=0.0,
        epsilon=0.01,
    )
    assert get_leader(campaign_id) is None
    assert _history(campaign_id) == [(EVENT_VACATED, None, leader_sid)]
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT winner_submission_id, leader_changed FROM rounds WHERE id = %s",
                (rid,),
            )
            winner, changed = cur.fetchone()
    assert winner is None
    assert changed is True
    assert _latest_state(challenger)[0] == "disqualified"


def test_void_leaves_leader_untouched_and_requeues_challengers():
    campaign_id = _campaign()
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    _seed_incumbent(campaign_id, leader_sid, IMAGE_C)
    challenger = _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    rid = str(claimed["id"])
    assert void_round(rid, VOID_POD_FAILED)
    seated = get_leader(campaign_id)
    assert seated is not None
    assert str(seated["submission_id"]) == leader_sid
    assert _history(campaign_id) == []
    state, detail = _latest_state(challenger)
    assert state == "bench_queued"
    assert detail["void_reason"] == VOID_POD_FAILED
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, void_reason, winner_submission_id FROM rounds WHERE id = %s",
                (rid,),
            )
            status, reason, winner = cur.fetchone()
    assert status == "void"
    assert reason == VOID_POD_FAILED
    assert winner is None


def test_void_reverts_live_streamed_disqualification():
    """A live-streamed DQ is provisional: settlement never ran on a voided
    round, so the entry goes back to pending and the challenger requeues."""
    campaign_id = _campaign()
    challenger = _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    rid = str(claimed["id"])
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE round_entries
                SET status = 'disqualified', disqualify_reason = 'live: crashed'
                WHERE round_id = %s AND role = 'challenger'
                RETURNING id
                """,
                (rid,),
            )
            (entry_id,) = cur.fetchone()
    assert void_round(rid, VOID_POD_FAILED)
    state, _detail = _latest_state(challenger)
    assert state == "bench_queued"
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, disqualify_reason FROM round_entries "
                "WHERE round_id = %s AND role = 'challenger'",
                (rid,),
            )
            row = cur.fetchone()
    assert row == ("pending", None)
    # A poll thread can outlive its join: a late beacon for a voided round
    # must not land.
    assert not update_round_entry_live_status(
        entry_id=entry_id, status="disqualified", reason="late"
    )
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM round_entries WHERE id = %s", (entry_id,))
            assert cur.fetchone()[0] == "pending"


def test_complete_seats_first_leader():
    campaign_id = _campaign()
    challenger = _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    rid = str(claimed["id"])
    rows = _runner_entries(rid)
    by_role = {r["role"]: r for r in rows}
    results = [
        _result(by_role["baseline"], status="scored", score=0.0),
        _result(by_role["challenger"], status="scored", score=0.31),
    ]
    decision = RankDecision(
        event=EVENT_SEATED,
        leader_submission_id=challenger,
        leader_score=0.31,
        overtake_threshold=0.0,
    )
    assert complete_round(
        round_id=rid,
        campaign_id=campaign_id,
        ordinal=int(claimed["ordinal"]),
        decision=decision,
        entries=results,
        baseline_drift=0.0,
        epsilon=0.01,
    )
    seated = get_leader(campaign_id)
    assert seated is not None
    assert str(seated["submission_id"]) == challenger
    assert _history(campaign_id) == [(EVENT_SEATED, challenger, None)]


def test_vacate_if_idle_writes_history_when_no_live_round():
    campaign_id = _campaign()
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    _seed_incumbent(campaign_id, leader_sid, IMAGE_C)
    idle = {str(r["campaign_id"]) for r in list_idle_seated_leaders()}
    assert str(campaign_id) in idle
    assert vacate_leader_if_idle(campaign_id, epsilon=0.01) is True
    assert get_leader(campaign_id) is None
    assert _history(campaign_id) == [(EVENT_VACATED, None, leader_sid)]


def test_vacate_if_idle_skips_a_running_round():
    campaign_id = _campaign()
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    _seed_incumbent(campaign_id, leader_sid, IMAGE_C)
    _queued(campaign_id, image_ref=IMAGE_A, block=10, waited_s=40_000)
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    idle = {str(r["campaign_id"]) for r in list_idle_seated_leaders()}
    assert str(campaign_id) not in idle
    assert vacate_leader_if_idle(campaign_id, epsilon=0.01) is False
    seated = get_leader(campaign_id)
    assert seated is not None
    assert str(seated["submission_id"]) == leader_sid
    assert _history(campaign_id) == []


def test_campaign_hotkey_disqualification_vacates_and_blocks_reentry():
    campaign_id = _campaign()
    hotkey = "5FakesHotkeyForE2ETesting000000000000000000000"
    leader_sid = _submission(campaign_id, image_ref=IMAGE_C, block=1)
    queued_sid = _queued(campaign_id, image_ref=IMAGE_A, block=2)
    _seed_incumbent(campaign_id, leader_sid, IMAGE_C)
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT patch_hash FROM submissions WHERE id = %s", (leader_sid,)
            )
            disqualified_patch_hash = cur.fetchone()[0]

    result = disqualify_campaign_hotkey(
        campaign_id,
        hotkey,
        reason="manual policy violation",
        evidence_ref="PAR-123",
        disqualified_by="test-operator",
        epsilon=0.01,
    )

    assert result.created is True
    assert result.submissions_disqualified == 2
    assert result.pending_jobs_stopped == 2
    assert result.leader_vacated is True
    assert get_leader(campaign_id) is None
    assert _history(campaign_id) == [(EVENT_VACATED, None, leader_sid)]
    assert _latest_state(leader_sid)[0] == "disqualified"
    assert _latest_state(queued_sid)[0] == "disqualified"
    assert campaigns_with_queue() == []

    with pytest.raises(CampaignHotkeyDisqualified):
        insert_submission(
            campaign_id=campaign_id,
            patch_hash="sha256:" + "9" * 64,
            hotkey=hotkey,
            baseline_commit="deadbeef",
            retrieval_url="https://cdn.test/reentry.diff",
            commit_block=3,
        )

    assert (
        insert_submission(
            campaign_id=campaign_id,
            patch_hash=disqualified_patch_hash.upper(),
            hotkey="5DifferentHotkeyForE2ETesting0000000000000000000",
            baseline_commit="deadbeef",
            retrieval_url="https://cdn.test/copied-solution.diff",
            commit_block=4,
        )
        is None
    )


def test_campaign_hotkey_waiver_allows_only_future_submissions():
    campaign_id = _campaign()
    hotkey = "5WaivedHotkeyForE2ETesting00000000000000000000"
    old_sid = _submission(
        campaign_id,
        image_ref=IMAGE_C,
        block=1,
        hotkey=hotkey,
    )
    disqualify_campaign_hotkey(
        campaign_id,
        hotkey,
        reason="temporary test",
        evidence_ref="PAR-123",
        disqualified_by="test-operator",
        epsilon=0.01,
    )

    waiver = waive_campaign_hotkey(
        campaign_id,
        hotkey,
        reason="temporary test ended",
        evidence_ref="PAR-124",
        waived_by="test-operator",
    )
    assert waiver.created is True
    assert _latest_state(old_sid)[0] == "disqualified"

    new_sid = insert_submission(
        campaign_id=campaign_id,
        patch_hash="sha256:" + "8" * 64,
        hotkey=hotkey,
        baseline_commit="deadbeef",
        retrieval_url="https://cdn.test/waived-reentry.diff",
        commit_block=2,
    )
    assert new_sid is not None
    assert _latest_state(str(new_sid))[0] == "committed"

    duplicate = waive_campaign_hotkey(
        campaign_id,
        hotkey,
        reason="duplicate operator request",
        evidence_ref="PAR-124",
        waived_by="test-operator",
    )
    assert duplicate.created is False

    reapplied = disqualify_campaign_hotkey(
        campaign_id,
        hotkey,
        reason="new policy violation",
        evidence_ref="PAR-125",
        disqualified_by="test-operator",
        epsilon=0.01,
    )
    assert reapplied.created is True
    assert _latest_state(str(new_sid))[0] == "disqualified"


def test_campaign_hotkey_disqualification_seats_best_eligible_runner_up():
    campaign_id = _campaign()
    excluded_hotkey = "5ExcludedLeaderForE2ETesting00000000000000000000"
    fallback_hotkey = "5FallbackLeaderForE2ETesting00000000000000000000"
    leader_sid = _submission(
        campaign_id,
        image_ref=IMAGE_C,
        block=1,
        hotkey=excluded_hotkey,
    )
    _seed_incumbent(campaign_id, leader_sid, IMAGE_C)
    fallback_sid = _queued(
        campaign_id,
        image_ref=IMAGE_A,
        block=2,
        waited_s=40_000,
        hotkey=fallback_hotkey,
    )
    create_due_rounds(_FakeSubtensor())
    claimed = claim_pending_round()
    assert claimed is not None
    round_id = str(claimed["id"])
    by_role = {row["role"]: row for row in _runner_entries(round_id)}
    assert complete_round(
        round_id=round_id,
        campaign_id=campaign_id,
        ordinal=int(claimed["ordinal"]),
        decision=RankDecision(
            leader_submission_id=leader_sid,
            leader_score=0.30,
            overtake_threshold=0.303,
        ),
        entries=[
            _result(by_role["baseline"], status="scored", score=0.0),
            _result(by_role["leader"], status="scored", score=0.30),
            _result(by_role["challenger"], status="scored", score=0.20),
        ],
        baseline_drift=0.0,
        epsilon=0.01,
    )

    result = disqualify_campaign_hotkey(
        campaign_id,
        excluded_hotkey,
        reason="manual policy violation",
        evidence_ref="PAR-123",
        disqualified_by="test-operator",
        epsilon=0.01,
    )

    assert result.leader_vacated is True
    assert result.replacement_submission_id == fallback_sid
    assert result.replacement_hotkey == fallback_hotkey
    replacement = get_leader(campaign_id)
    assert replacement is not None
    assert str(replacement["submission_id"]) == fallback_sid
    assert replacement["hotkey"] == fallback_hotkey
    assert float(replacement["last_score"]) == pytest.approx(0.20)
    assert str(replacement["won_at_round_id"]) == round_id
    assert _history(campaign_id) == [
        (EVENT_VACATED, None, leader_sid),
        (EVENT_SEATED, fallback_sid, None),
    ]
