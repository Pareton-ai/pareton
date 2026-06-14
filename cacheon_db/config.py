"""Environment-driven Postgres config."""

from __future__ import annotations

import os

from .exceptions import DatabaseNotConfigured

DATABASE_URL: str = os.environ.get("CACHEON_DATABASE_URL", "")
SKIP_DB: bool = os.environ.get("CACHEON_SKIP_DB", "0") == "1"


def enabled() -> bool:
    return bool(DATABASE_URL) and not SKIP_DB


def require_database_url() -> str:
    """Return DATABASE_URL or raise for API read paths."""
    if not DATABASE_URL:
        raise DatabaseNotConfigured("CACHEON_DATABASE_URL is not configured")
    return DATABASE_URL
