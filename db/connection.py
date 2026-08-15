"""Postgres connection helper for Pareton (Neon)."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Iterator
from typing import Any

from .exceptions import DatabaseNotConfigured, DatabaseUnavailable

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.environ.get("PARETON_DATABASE_URL", "")

_POOL_MIN = 1
_POOL_MAX = 10
_KEEPALIVES = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
}

_pool: Any = None
_pool_lock = threading.Lock()


def require_database_url() -> str:
    """Return PARETON_DATABASE_URL or raise."""
    url = os.environ.get("PARETON_DATABASE_URL", "") or DATABASE_URL
    if not url:
        raise DatabaseNotConfigured("PARETON_DATABASE_URL is not configured")
    return url


def _get_pool() -> Any:
    """Return the process pool. Failed init leaves ``_pool`` as None."""
    global _pool
    if _pool is not None:
        return _pool
    url = require_database_url()
    from psycopg2.pool import ThreadedConnectionPool

    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            created = ThreadedConnectionPool(_POOL_MIN, _POOL_MAX, url, **_KEEPALIVES)
        except Exception as exc:
            raise DatabaseUnavailable("database connection failed") from exc
        _pool = created
        return _pool


def _checkout() -> Any:
    from psycopg2.pool import PoolError

    pool = _get_pool()
    try:
        return pool.getconn()
    except PoolError as exc:
        raise DatabaseUnavailable("database connection failed") from exc
    except Exception as exc:
        raise DatabaseUnavailable("database connection failed") from exc


def _putconn(conn: Any, *, close: bool) -> None:
    pool = _pool
    if pool is None:
        if not conn.closed:
            conn.close()
        return
    try:
        pool.putconn(conn, close=close)
    except Exception:
        if not conn.closed:
            try:
                conn.close()
            except Exception:
                pass


def _release(conn: Any) -> None:
    if conn.closed:
        _putconn(conn, close=True)
        return
    try:
        conn.autocommit = False
    except Exception:
        _putconn(conn, close=True)
        return
    _putconn(conn, close=False)


@contextlib.contextmanager
def db_connection(*, readonly: bool = False) -> Iterator[Any]:
    """Yield a pooled connection. Writes stay transactional; reads use autocommit."""
    conn = None
    try:
        conn = _checkout()
        conn.autocommit = readonly
        yield conn
        if not readonly:
            conn.commit()
    except Exception as exc:
        if conn is not None and not conn.closed and not readonly:
            try:
                conn.rollback()
            except Exception:
                pass
        # A pooled socket the server dropped (Neon maintenance/compute release)
        # only surfaces on first use: psycopg2 marks the connection closed and
        # raises OperationalError. Keep the pre-pool contract of 503, not 500.
        if conn is not None and conn.closed:
            raise DatabaseUnavailable("database connection failed") from exc
        raise
    finally:
        if conn is not None:
            _release(conn)
