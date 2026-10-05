"""DB-backed migration and constraint regressions on an isolated test schema."""

from pathlib import Path

import psycopg2
import pytest
import test_fee_schema_upgrade
from psycopg2.extras import Json

from campaign.store import _row_to_manifest

schema_db = test_fee_schema_upgrade.schema_db
SCHEMA_SQL = test_fee_schema_upgrade.SCHEMA_SQL

pytestmark = pytest.mark.e2e
MIGRATION = (
    Path(__file__).parents[1] / "db/migrations/20261001_campaign_patch_visibility.sql"
).read_text()


def test_existing_campaigns_default_private_without_changing_pins(schema_db):
    schema_db.execute("ALTER TABLE campaigns DROP COLUMN patch_visibility")
    schema_db.execute("""
        INSERT INTO campaigns (baseline_repo, baseline_commit, base_image_digest,
          priority_metric, success_threshold, manifest_hash, customer_signoff,
          workload_trace_sha256, workload_trace_url, scoring_rule, submission_fee_history)
        VALUES ('repo', 'commit', 'digest', 'throughput', 'threshold', 'original-hash',
          '{"approved_manifest_hash":"original-hash","approver":"test","timestamp":"2026-10-01T00:00:00Z"}',
          'trace', 'https://example.com/trace', '{"name":"median_e2e_speedup"}',
          '[{"amount_tao":"0","recipient":"5Test","effective_from_block":0}]')
    """)
    schema_db.execute("SELECT manifest_hash, customer_signoff FROM campaigns")
    pins = schema_db.fetchone()
    with pytest.raises(psycopg2.Error, match="20261001_campaign_patch_visibility"):
        schema_db.execute(SCHEMA_SQL)
    schema_db.execute("ROLLBACK")
    schema_db.execute(MIGRATION)
    schema_db.execute("SELECT * FROM campaigns")
    row = schema_db.fetchone()
    assert row["patch_visibility"] == {"mode": "private"}
    assert _row_to_manifest(dict(row)).patch_visibility == {"mode": "private"}
    public = {"mode": "public_after_reveal", "reveal_delay_s": 3600}
    schema_db.execute("UPDATE campaigns SET patch_visibility = %s", (Json(public),))
    schema_db.execute(MIGRATION)
    schema_db.execute(SCHEMA_SQL)
    schema_db.execute("SELECT * FROM campaigns")
    row = schema_db.fetchone()
    assert row["patch_visibility"] == public
    assert {key: row[key] for key in pins} == pins
    assert _row_to_manifest(dict(row)).patch_visibility == public


@pytest.mark.parametrize(
    "policy,valid",
    [
        ({"mode": "private"}, True),
        ({"mode": "public_after_reveal", "reveal_delay_s": 0}, True),
        ({"mode": "public_after_reveal", "reveal_delay_s": 2147483647}, True),
        (None, False),
        ({}, False),
        ([], False),
        ("private", False),
        ({"mode": "public"}, False),
        ({"mode": "public_after_reveal"}, False),
        ({"mode": "private", "reveal_delay_s": 1}, False),
        *[
            ({"mode": "public_after_reveal", "reveal_delay_s": v}, False)
            for v in (-1, True, 1.5, "0", None, 2147483648)
        ],
    ],
)
def test_database_policy_validation_matches_canonical_python(schema_db, policy, valid):
    schema_db.execute(
        "SELECT valid_campaign_patch_visibility(%s) AS valid", (Json(policy),)
    )
    assert schema_db.fetchone()["valid"] is valid
