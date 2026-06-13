"""Environment-driven Postgres config."""

from __future__ import annotations

import os

DATABASE_URL: str = os.environ.get("CACHEON_DATABASE_URL", "")
SKIP_DB: bool = os.environ.get("CACHEON_SKIP_DB", "0") == "1"


def enabled() -> bool:
    return bool(DATABASE_URL) and not SKIP_DB
