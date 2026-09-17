"""Require migration before reapplying the schema to pre-fee campaigns."""

from pathlib import Path
from uuid import uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import pytest

from campaign.store import _row_to_manifest
from e2e_db import require_e2e_database_url

pytestmark = pytest.mark.e2e

# gen_random_uuid is built in; do not move/install a shared extension in a test schema.
SCHEMA_SQL = (
    (Path(__file__).parents[1] / "db/schema.sql")
    .read_text()
    .replace("CREATE EXTENSION IF NOT EXISTS pgcrypto;", "")
)


@pytest.fixture
def schema_db():
    conn = psycopg2.connect(require_e2e_database_url())
    conn.autocommit = True
    name = "fee_schema_test_" + uuid4().hex
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
            cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(name)))
            cur.execute(SCHEMA_SQL)
            yield cur
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name))
            )
        conn.close()


def _legacy_campaign(cur, status, manifest_hash):
    cur.execute(
        """INSERT INTO campaigns (
        baseline_repo, baseline_commit, base_image_digest, priority_metric,
        success_threshold, manifest_hash, customer_signoff, scoring_rule,
        status, bench, workload_trace_sha256, workload_trace_url
    ) VALUES (
        'repo', 'commit', 'digest', 'throughput', 'threshold', %s,
        jsonb_build_object('approved_manifest_hash', %s::text,
                          'approver', 'test', 'timestamp', '2026-09-17T00:00:00Z'),
        '{"name":"median_e2e_speedup"}', %s,
        '{"model":{"hf_repo":"RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"}}',
        'trace', 'https://example.com/trace'
    )""",
        (manifest_hash, manifest_hash, status),
    )


def test_migration_then_schema_keeps_populated_campaigns_readable(schema_db):
    schema_db.execute(
        "ALTER TABLE campaigns DROP COLUMN submission_fee_history CASCADE"
    )
    _legacy_campaign(schema_db, "closed", "closed-original")
    _legacy_campaign(schema_db, "open", "open-original")
    with pytest.raises(psycopg2.Error, match="run db/migrations"):
        schema_db.execute(SCHEMA_SQL)
    schema_db.execute("ROLLBACK")
    schema_db.execute(
        (
            Path(__file__).parents[1]
            / "db/migrations/20260917_campaign_fee_history.sql"
        ).read_text()
    )
    schema_db.execute(SCHEMA_SQL)
    schema_db.execute("SELECT * FROM campaigns ORDER BY manifest_hash")
    rows = schema_db.fetchall()
    assert [r["submission_fee_history"][0]["amount_tao"] for r in rows] == [
        "0.1",
        "0.15",
    ]
    for row in rows:
        manifest = _row_to_manifest(dict(row))
        assert manifest.submission_fee_history == row["submission_fee_history"]
        assert (
            manifest.manifest_hash == row["customer_signoff"]["approved_manifest_hash"]
        )
    schema_db.execute(SCHEMA_SQL)
    schema_db.execute("SELECT * FROM campaigns ORDER BY manifest_hash")
    assert schema_db.fetchall() == rows
    with pytest.raises(psycopg2.Error):
        schema_db.execute("UPDATE campaigns SET submission_fee_history = NULL")
    with pytest.raises(psycopg2.Error, match="append-only"):
        schema_db.execute("""UPDATE campaigns SET submission_fee_history =
            '[{"amount_tao":"0.2","recipient":"5Recipient","effective_from_block":0}]'""")


def test_fresh_schema_reapplication_succeeds(schema_db):
    schema_db.execute(SCHEMA_SQL)
    schema_db.execute("""SELECT is_nullable FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'campaigns'
        AND column_name = 'submission_fee_history'""")
    assert schema_db.fetchone()["is_nullable"] == "NO"


def test_unmigrated_campaign_rejects_schema_reapplication(schema_db):
    schema_db.execute(
        "ALTER TABLE campaigns DROP COLUMN submission_fee_history CASCADE"
    )
    _legacy_campaign(schema_db, "draft", "draft-original")
    with pytest.raises(psycopg2.Error, match="run db/migrations"):
        schema_db.execute(SCHEMA_SQL)
    schema_db.execute("ROLLBACK")
    schema_db.execute("""SELECT column_name FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'campaigns'
        AND column_name = 'submission_fee_history'""")
    assert schema_db.fetchone() is None
