"""Neon test-branch binding for DB-backed e2e tests.

E2E must never write to the partner/main database. Gate on
``PARETON_TEST_DATABASE_URL`` (not ``PARETON_DATABASE_URL``), refuse when both
URLs share a hostname, then rebind the process connection URL for the test.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest

_E2E_PROFILE_NAMES = (
    "e2e",
    "e2e-bench",
    "e2e-xenv",
    "e2e-engine",
    "e2e-calib",
    "e2e-emission",
)


def _hostname(url: str) -> str | None:
    host = urlparse(url).hostname
    if host is None:
        return None
    # Neon serves one branch at both ep-<id>.<region>... and
    # ep-<id>-pooler.<region>...; the pair must compare equal (fail closed).
    first, dot, rest = host.partition(".")
    return first.removesuffix("-pooler") + dot + rest


def require_e2e_database_url() -> str:
    """Return the e2e DB URL, or skip/fail per policy."""
    test_url = (os.environ.get("PARETON_TEST_DATABASE_URL") or "").strip()
    if not test_url:
        pytest.skip("PARETON_TEST_DATABASE_URL not set")

    main_url = (os.environ.get("PARETON_DATABASE_URL") or "").strip()
    test_host = _hostname(test_url)
    main_host = _hostname(main_url) if main_url else None
    if not test_host:
        pytest.fail("PARETON_TEST_DATABASE_URL has no hostname")
    if main_host and test_host == main_host:
        pytest.fail(
            "PARETON_TEST_DATABASE_URL must not use the same host as "
            "PARETON_DATABASE_URL (refusing to run e2e against main)"
        )
    return test_url


def cleanup_e2e_rows() -> None:
    """Delete e2e profiles and dependent rows (test branch hygiene only)."""
    from db.connection import db_connection

    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM profiles
                WHERE name = ANY(%s)
                """,
                (list(_E2E_PROFILE_NAMES),),
            )
            profile_ids = tuple(row[0] for row in cur.fetchall())
            if not profile_ids:
                return
            # Rounds and leaders reference submissions, so they go first.
            # round_entries follows its round by ON DELETE CASCADE.
            for table in ("leader_history", "leaders", "rounds"):
                cur.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE campaign_id IN (
                      SELECT id FROM campaigns WHERE profile_id IN %s
                    )
                    """,
                    (profile_ids,),
                )
            cur.execute(
                """
                DELETE FROM submissions
                WHERE campaign_id IN (
                  SELECT id FROM campaigns WHERE profile_id IN %s
                )
                """,
                (profile_ids,),
            )
            cur.execute(
                "DELETE FROM campaigns WHERE profile_id IN %s",
                (profile_ids,),
            )
            cur.execute(
                "DELETE FROM profiles WHERE id IN %s",
                (profile_ids,),
            )
