"""The emission rule in Postgres: round-trip plus the cross-campaign sum trigger.

Binds to the Neon test branch via ``e2e_db``, never the partner/main database.
The trigger is the only place the ``sum(start_weight) <= 1.0`` invariant lives,
so it is exercised through the same insert path ops and the seeder use.
"""

from __future__ import annotations

from uuid import uuid4

import psycopg2
import pytest
from psycopg2.extras import Json

pytestmark = pytest.mark.e2e

from campaign.manifest import build_manifest
from campaign.models import SLA
from campaign.store import get_campaign, insert_campaign, insert_profile
from db.connection import db_connection
from e2e_db import cleanup_e2e_rows, require_e2e_database_url

RULE = {
    "name": "linear_decay",
    "start_weight": 0.10,
    "floor_weight": 0.02,
    "decay_blocks": 201600,
}


@pytest.fixture(autouse=True)
def _bind_e2e_database(monkeypatch: pytest.MonkeyPatch):
    """Point store/connection code at the Neon test branch for this module."""
    url = require_e2e_database_url()
    monkeypatch.setenv("PARETON_DATABASE_URL", url)
    import db.connection as conn

    monkeypatch.setattr(conn, "DATABASE_URL", url)
    monkeypatch.setattr(conn, "_pool", None)
    if _open_start_weight_sum() != 0:
        pytest.skip("the test branch already has an open campaign that pays")
    yield
    cleanup_e2e_rows()


def _open_start_weight_sum() -> float:
    """What the trigger sees today. The sum tests need the whole budget free."""
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM((emission_rule ->> 'start_weight')::NUMERIC), 0)
                FROM campaigns
                WHERE status = 'open' AND emission_rule IS NOT NULL
                """
            )
            return float(cur.fetchone()[0])


def _insert(*, status: str, emission_rule: dict | None):
    campaign_id = uuid4()
    profile_id = insert_profile("e2e-emission", {"fixture": True})
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo="https://github.com/vllm-project/vllm.git",
        baseline_commit="f" * 40,
        base_image_digest="sha256:" + "d" * 64,
        gpu_skus=["H200"],
        workload_trace_sha256="sha256:" + "e" * 64,
        workload_trace_url="https://cdn.test/trace.json",
        sla=SLA(),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        submission_fee={"amount_tao": "0", "recipient": "5Test"},
        status=status,
        emission_rule=emission_rule,
    )
    insert_campaign(manifest)
    return manifest


def _weighted(share: float) -> dict:
    return {**RULE, "start_weight": share, "floor_weight": 0.0}


def test_emission_rule_round_trips_through_postgres():
    written = _insert(status="draft", emission_rule=RULE)
    read_back = get_campaign(written.campaign_id)
    assert read_back is not None
    assert read_back.emission_rule == RULE
    # The stored rule must reproduce the hash it was pinned with, and that is what
    # makes the pay schedule signed rather than loose operational config.
    assert read_back.manifest_hash == written.manifest_hash


def test_absent_emission_rule_round_trips_as_null():
    written = _insert(status="draft", emission_rule=None)
    read_back = get_campaign(written.campaign_id)
    assert read_back is not None
    assert read_back.emission_rule is None
    assert read_back.manifest_hash == written.manifest_hash


def test_the_open_campaigns_may_not_promise_more_than_the_subnet_has():
    _insert(status="open", emission_rule=_weighted(0.7))
    with pytest.raises(psycopg2.errors.RaiseException, match="over 1.0"):
        _insert(status="open", emission_rule=_weighted(0.4))


def test_exactly_one_whole_subnet_is_allowed():
    _insert(status="open", emission_rule=_weighted(0.7))
    _insert(status="open", emission_rule=_weighted(0.3))
    assert _open_start_weight_sum() == 1.0


def test_draft_and_closed_campaigns_do_not_spend_the_budget():
    """Only open campaigns pay, so only open campaigns count against the cap."""
    _insert(status="draft", emission_rule=_weighted(0.9))
    _insert(status="closed", emission_rule=_weighted(0.9))
    _insert(status="open", emission_rule=_weighted(1.0))
    assert _open_start_weight_sum() == 1.0


@pytest.mark.parametrize(
    "broken, expected",
    [
        ({"name": "linear_decay"}, "must be a number, got missing"),
        ({"name": "linear_decay", "start_weight": None}, "must be a number, got null"),
        (
            {"name": "linear_decay", "start_weight": "0.4"},
            "must be a number, got string",
        ),
        ({"name": "linear_decay", "start_weight": -0.5}, r"must be in \[0, 1\]"),
    ],
)
def test_a_rule_whose_start_weight_cannot_be_read_is_rejected(broken, expected):
    """A missing or unreadable start_weight must not slip past the cap.

    `->>` on a missing key yields SQL NULL, and `NULL + others > 1.0` is
    unknown, so the guard used to pass exactly the row it exists to stop.
    Reaching this needs raw SQL, which is the threat model the trigger is for:
    validate_emission_rule already blocks these through the Python path.
    """
    _insert(status="open", emission_rule=_weighted(1.0))
    spare = _insert(status="draft", emission_rule=_weighted(0.0))
    with pytest.raises(psycopg2.errors.RaiseException, match=expected):
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE campaigns SET status = 'open', emission_rule = %s "
                    "WHERE id = %s",
                    (Json(broken), str(spare.campaign_id)),
                )


def test_the_trigger_catches_a_manual_update_too():
    """An application gate cannot see this write; the trigger is why it exists."""
    _insert(status="open", emission_rule=_weighted(0.7))
    drafted = _insert(status="draft", emission_rule=_weighted(0.4))
    with pytest.raises(psycopg2.errors.RaiseException, match="over 1.0"):
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE campaigns SET status = 'open' WHERE id = %s",
                    (str(drafted.campaign_id),),
                )
