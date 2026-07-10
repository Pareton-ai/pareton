"""Postgres connection helper for Pareton (Neon)."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

from .exceptions import DatabaseNotConfigured, DatabaseUnavailable

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.environ.get("PARETON_DATABASE_URL", "")


def require_database_url() -> str:
    """Return PARETON_DATABASE_URL or raise."""
    url = os.environ.get("PARETON_DATABASE_URL", "") or DATABASE_URL
    if not url:
        raise DatabaseNotConfigured("PARETON_DATABASE_URL is not configured")
    return url


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
