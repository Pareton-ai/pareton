"""Environment-driven Postgres config."""

from __future__ import annotations

import os

from .exceptions import DatabaseNotConfigured

DATABASE_URL: str = os.environ.get("CACHEON_DATABASE_URL", "")


def require_database_url() -> str:
    """Return DATABASE_URL or raise."""
    if not DATABASE_URL:
        raise DatabaseNotConfigured("CACHEON_DATABASE_URL is not configured")
    return DATABASE_URL
