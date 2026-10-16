"""Offline tests for the pooled db_connection helper."""

from __future__ import annotations

import threading
import time

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


class BoundedPool:
    """Mimics psycopg2's pool: PoolError as soon as `size` conns are out."""

    def __init__(self, size: int):
        self.size = size
        self.out = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.puts = 0

    def getconn(self) -> FakeConn:
        with self.lock:
            if self.out >= self.size:
                raise PoolError("connection pool exhausted")
            self.out += 1
            self.peak = max(self.peak, self.out)
        return FakeConn()

    def putconn(self, conn: FakeConn, key=None, close: bool = False) -> None:
        with self.lock:
            self.out -= 1
            self.puts += 1


@pytest.fixture
def conn_mod(monkeypatch):
    import db.connection as conn_mod

    prev = conn_mod._pool
    conn_mod._pool = None
    monkeypatch.setattr(conn_mod, "_slots", threading.BoundedSemaphore(4))
    monkeypatch.setattr(conn_mod, "_POOL_WAIT_S", 2.0)
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


def test_burst_above_pool_size_waits_instead_of_503(conn_mod):
    """More concurrent requests than connections: they queue, none fail."""
    pool = BoundedPool(size=4)
    conn_mod._pool = pool
    errors: list[BaseException] = []

    def handle_request() -> None:
        try:
            with conn_mod.db_connection(readonly=True):
                time.sleep(0.05)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=handle_request) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert pool.peak <= 4
    assert pool.puts == 20
    assert pool.out == 0


def test_saturated_pool_is_unavailable_after_wait(conn_mod):
    conn_mod._pool = BoundedPool(size=4)
    conn_mod._POOL_WAIT_S = 0.05
    for _ in range(4):
        assert conn_mod._slots.acquire(blocking=False)

    started = time.monotonic()
    with pytest.raises(DatabaseUnavailable, match="database connection failed"):
        with conn_mod.db_connection(readonly=True):
            pass
    assert time.monotonic() - started >= 0.05


def test_failed_checkout_frees_its_slot(conn_mod):
    """A getconn error must not leak a slot, or the pool shrinks for good."""
    conn_mod._pool = ExhaustedPool()
    for _ in range(10):
        with pytest.raises(DatabaseUnavailable):
            with conn_mod.db_connection():
                pass

    conn = FakeConn()
    conn_mod._pool = FakePool(conn)
    with conn_mod.db_connection() as yielded:
        assert yielded is conn
