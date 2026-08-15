"""Offline tests for the pooled db_connection helper."""

from __future__ import annotations

import pytest
from psycopg2 import OperationalError
from psycopg2.pool import PoolError

from db.exceptions import DatabaseUnavailable

pytestmark = pytest.mark.unit


class FakeConn:
    def __init__(self, *, closed: int = 0):
        self.closed = closed
        self.autocommit = False
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = 1


class FakePool:
    def __init__(self, conn: FakeConn):
        self.conn = conn
        self.puts: list[tuple[FakeConn, bool]] = []

    def getconn(self) -> FakeConn:
        return self.conn

    def putconn(self, conn: FakeConn, key=None, close: bool = False) -> None:
        self.puts.append((conn, close))


class ExhaustedPool:
    def getconn(self) -> FakeConn:
        raise PoolError("connection pool exhausted")


@pytest.fixture
def conn_mod():
    import db.connection as conn_mod

    prev = conn_mod._pool
    conn_mod._pool = None
    yield conn_mod
    conn_mod._pool = prev


def test_write_commits_and_returns_to_pool(conn_mod):
    conn = FakeConn()
    pool = FakePool(conn)
    conn_mod._pool = pool

    with conn_mod.db_connection() as yielded:
        assert yielded is conn
        assert conn.autocommit is False

    assert conn.commits == 1
    assert conn.rollbacks == 0
    assert pool.puts == [(conn, False)]


def test_readonly_skips_commit(conn_mod):
    conn = FakeConn()
    pool = FakePool(conn)
    conn_mod._pool = pool

    with conn_mod.db_connection(readonly=True) as yielded:
        assert yielded.autocommit is True

    assert conn.commits == 0
    assert conn.rollbacks == 0
    assert conn.autocommit is False
    assert pool.puts == [(conn, False)]


def test_write_exception_rolls_back(conn_mod):
    conn = FakeConn()
    pool = FakePool(conn)
    conn_mod._pool = pool

    with pytest.raises(RuntimeError, match="boom"):
        with conn_mod.db_connection():
            raise RuntimeError("boom")

    assert conn.commits == 0
    assert conn.rollbacks == 1
    assert pool.puts == [(conn, False)]


def test_pool_error_is_unavailable(conn_mod):
    conn_mod._pool = ExhaustedPool()

    with pytest.raises(DatabaseUnavailable, match="database connection failed"):
        with conn_mod.db_connection():
            pass


def test_closed_connection_is_discarded(conn_mod):
    conn = FakeConn()
    pool = FakePool(conn)
    conn_mod._pool = pool

    with conn_mod.db_connection() as yielded:
        yielded.closed = 1

    assert pool.puts == [(conn, True)]


def test_dead_pooled_socket_is_unavailable_not_raw_error(conn_mod):
    """Server-dropped socket: 503 like a failed connect, and never reused."""
    conn = FakeConn()
    pool = FakePool(conn)
    conn_mod._pool = pool

    with pytest.raises(DatabaseUnavailable, match="database connection failed"):
        with conn_mod.db_connection(readonly=True) as yielded:
            # psycopg2 only notices on first use, then marks the conn closed.
            yielded.closed = 2
            raise OperationalError("SSL connection has been closed unexpectedly")

    assert conn.rollbacks == 0
    assert pool.puts == [(conn, True)]


def test_query_error_keeps_its_type(conn_mod):
    """A live connection failing a statement must not be masked as a 503."""
    conn = FakeConn()
    pool = FakePool(conn)
    conn_mod._pool = pool

    with pytest.raises(OperationalError, match="deadlock detected"):
        with conn_mod.db_connection():
            raise OperationalError("deadlock detected")

    assert conn.rollbacks == 1
    assert pool.puts == [(conn, False)]


def test_failed_pool_init_does_not_poison(conn_mod, monkeypatch):
    monkeypatch.setenv("PARETON_DATABASE_URL", "postgresql://u:p@example/db")

    def boom(*_args, **_kwargs):
        raise OSError("neon unreachable")

    monkeypatch.setattr("psycopg2.pool.ThreadedConnectionPool", boom)

    with pytest.raises(DatabaseUnavailable):
        with conn_mod.db_connection():
            pass

    assert conn_mod._pool is None
