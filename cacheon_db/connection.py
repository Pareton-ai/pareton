"""Postgres connection helper."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Any

from .config import require_database_url
from .exceptions import DatabaseUnavailable

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def db_connection() -> Iterator[Any]:
    """Yield a psycopg2 connection. Raises when URL is missing or connect fails."""
    import psycopg2

    url = require_database_url()
    try:
        conn = psycopg2.connect(url)
    except Exception as exc:
        raise DatabaseUnavailable("database connection failed") from exc
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
    with db_connection() as conn:
        yield conn


def run_best_effort(label: str, fn: Callable[[Any], None]) -> None:
    """Run *fn(conn)* inside a transaction; log and swallow failures."""
    try:
        with db_connection() as conn:
            fn(conn)
    except DatabaseUnavailable as exc:
        logger.error("Postgres mirror %s failed: %s", label, exc)
    except Exception as exc:
        logger.error("Postgres mirror %s failed: %s", label, exc)
