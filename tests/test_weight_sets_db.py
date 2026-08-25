"""`get_latest_weight_set` against Postgres: which row is in force, and none.

Binds to the Neon test branch via ``e2e_db``, never the partner/main database.
Ordering is the whole point of the reader, and it lives in SQL, so it can only
be proved with real rows.
"""

from __future__ import annotations

import pytest
from psycopg2.extras import Json

pytestmark = pytest.mark.e2e

from db.connection import db_connection
from e2e_db import require_e2e_database_url
from round.store import get_latest_weight_set

_INSERTED: list[int] = []


@pytest.fixture(autouse=True)
def _bind_e2e_database(monkeypatch: pytest.MonkeyPatch):
    """Point store/connection code at the Neon test branch for this module."""
    url = require_e2e_database_url()
    monkeypatch.setenv("PARETON_DATABASE_URL", url)
    import db.connection as conn

    monkeypatch.setattr(conn, "DATABASE_URL", url)
    monkeypatch.setattr(conn, "_pool", None)
    _INSERTED.clear()
    yield
    _delete_inserted()


def _count() -> int:
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM weight_sets")
            return int(cur.fetchone()[0])


def _delete_inserted() -> None:
    """Remove only the rows this module wrote. `weight_sets` is append-only."""
    if not _INSERTED:
        return
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM weight_sets WHERE id = ANY(%s)", (_INSERTED,))
    _INSERTED.clear()


def _insert(block: int, uid: int, weight: float) -> None:
    dense = [0.0] * 202
    dense[uid] = weight
    dense[201] = round(1.0 - weight, 10)
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO weight_sets (computed_at_block, version_key, burn_uid,
                                         weights, breakdown)
                VALUES (%s, 2032, 201, %s, %s)
                RETURNING id
                """,
                (block, Json(dense), Json([])),
            )
            _INSERTED.append(int(cur.fetchone()[0]))


def test_an_empty_table_reads_as_none_not_an_empty_vector():
    """Before the first cycle there is nothing to serve, and that is not zero."""
    if _count() != 0:
        pytest.skip("the test branch already holds weight_sets rows")
    assert get_latest_weight_set() is None


def test_the_newest_row_is_the_one_in_force():
    _insert(6_000_000, 12, 0.1)
    _insert(6_000_360, 12, 0.2)
    _insert(6_000_720, 31, 0.3)

    row = get_latest_weight_set()
    assert row is not None
    assert row["computed_at_block"] == 6_000_720
    assert row["weights"][31] == pytest.approx(0.3)
    # Only the vector is served. How the chain call went is not part of it.
    assert "set_ok" not in row
    assert "set_error" not in row
