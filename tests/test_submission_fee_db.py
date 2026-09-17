"""Actual hand-run migration checks in an isolated schema on the test DB only."""

from pathlib import Path
from uuid import uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
import pytest

from e2e_db import require_e2e_database_url

pytestmark = pytest.mark.e2e


@pytest.fixture
def fee_db():
    conn = psycopg2.connect(require_e2e_database_url())
    conn.autocommit = True
    schema = "fee_test_" + uuid4().hex
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        try:
            cur.execute("""CREATE TABLE campaigns (
                id uuid DEFAULT gen_random_uuid(), status text DEFAULT 'open',
                manifest_hash text, customer_signoff jsonb, bench jsonb,
                submission_fee_history jsonb,
                created_at timestamptz DEFAULT now())""")
            cur.execute("""INSERT INTO campaigns (manifest_hash, customer_signoff, bench)
                VALUES ('signed-original', '{"approved_manifest_hash":"signed-original"}',
                        '{"model":{"hf_repo":"RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"}}')""")
            cur.execute("""INSERT INTO campaigns (status, manifest_hash)
                VALUES ('closed', 'closed-original')""")
            cur.execute("""INSERT INTO campaigns (status, manifest_hash, submission_fee_history)
                VALUES ('draft', 'existing-history',
                        '[{"amount_tao":"0.27","recipient":"5Recipient","effective_from_block":0}]')""")
            migration = (
                Path(__file__).parents[1]
                / "db/migrations/20260917_campaign_fee_history.sql"
            ).read_text()
            cur.execute(migration)
            cur.execute(migration)
            yield cur
        finally:
            conn.rollback()
            cur.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )
            conn.close()


def test_migration_preserves_signed_terms_and_history_is_append_only(fee_db):
    fee_db.execute(
        "SELECT manifest_hash, customer_signoff, submission_fee_history FROM campaigns WHERE manifest_hash = 'signed-original'"
    )
    manifest_hash, signoff, history = fee_db.fetchone()
    assert manifest_hash == signoff["approved_manifest_hash"] == "signed-original"
    assert history[0]["amount_tao"] == "0.15"
    updated = [
        *history,
        {**history[0], "amount_tao": "0.2", "effective_from_block": 1000},
    ]
    fee_db.execute(
        "UPDATE campaigns SET submission_fee_history = %s WHERE manifest_hash = 'signed-original'",
        (Json(updated),),
    )
    with pytest.raises(psycopg2.Error, match="append-only"):
        fee_db.execute(
            "UPDATE campaigns SET submission_fee_history = %s WHERE manifest_hash = 'signed-original'",
            (Json(history),),
        )
    with pytest.raises(psycopg2.Error, match="append-only"):
        fee_db.execute(
            "UPDATE campaigns SET submission_fee_history = %s WHERE manifest_hash = 'signed-original'",
            (Json([{**history[0], "amount_tao": "0.1"}, updated[1]]),),
        )


@pytest.mark.parametrize("amount", [0.15, "0.0000000001", "NaN", "-1", None])
def test_database_rejects_invalid_amounts(fee_db, amount):
    history = [
        {"amount_tao": amount, "recipient": "5Recipient", "effective_from_block": 0}
    ]
    fee_db.execute("SELECT valid_campaign_fee_history(%s)", (Json(history),))
    assert fee_db.fetchone()[0] is False


def test_initial_backfill_assigns_closed_fee_and_preserves_existing_history(fee_db):
    fee_db.execute("SELECT manifest_hash, submission_fee_history FROM campaigns")
    fees = {name: history for name, history in fee_db.fetchall()}
    assert fees["closed-original"][0]["amount_tao"] == "0.1"
    assert fees["signed-original"][0]["amount_tao"] == "0.15"
    assert fees["existing-history"] == [
        {
            "amount_tao": "0.27",
            "recipient": "5Recipient",
            "effective_from_block": 0,
        }
    ]


@pytest.mark.parametrize(
    "status,bench",
    [
        ("draft", None),
        ("open", None),
        ("open", {"model": {"hf_repo": "Other/model"}}),
    ],
)
def test_initial_backfill_refuses_unmapped_campaigns(fee_db, status, bench):
    fee_db.execute(
        "ALTER TABLE campaigns ALTER COLUMN submission_fee_history DROP NOT NULL"
    )
    fee_db.execute(
        "ALTER TABLE campaigns DROP CONSTRAINT campaigns_submission_fee_history_check"
    )
    fee_db.execute(
        "INSERT INTO campaigns (status, bench) VALUES (%s, %s)", (status, Json(bench))
    )
    migration = (
        Path(__file__).parents[1] / "db/migrations/20260917_campaign_fee_history.sql"
    ).read_text()
    with pytest.raises(psycopg2.Error, match="initial fee backfill only covers"):
        fee_db.execute(migration)
    fee_db.connection.rollback()
