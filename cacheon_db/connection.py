"""Postgres connection helper."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Any

from .config import DATABASE_URL, enabled, require_database_url
from .exceptions import DatabaseUnavailable

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def db_connection() -> Iterator[Any | None]:
    """Yield a psycopg2 connection, or ``None`` when DB mirroring is disabled."""
    if not enabled():
        yield None
        return

    import psycopg2

    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextlib.contextmanager
def read_db_connection() -> Iterator[Any]:
    """Yield a psycopg2 connection for required API reads."""
    import psycopg2

    url = require_database_url()
    try:
        conn = psycopg2.connect(url)
    except Exception as exc:
        raise DatabaseUnavailable("database connection failed") from exc
    try:
        yield conn
    finally:
        conn.close()


def run_best_effort(label: str, fn: Callable[[Any], None]) -> None:
    """Run *fn(conn)* inside a transaction; log and swallow failures."""
    if not enabled():
        return
    try:
        with db_connection() as conn:
            if conn is None:
                return
            fn(conn)
    except Exception as exc:
        logger.error("Postgres mirror %s failed: %s", label, exc)
