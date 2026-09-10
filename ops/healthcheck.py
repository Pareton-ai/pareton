"""API readiness probe: HTTP responds and Postgres accepts a query."""

import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from db.connection import db_connection


def main() -> None:
    if sys.argv[1:] == ["worker"]:
        age = time.time() - Path(os.environ["PARETON_HEALTH_FILE"]).stat().st_mtime
        if age > 660:
            raise RuntimeError("worker heartbeat is stale")
        return
    with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("API is not ready")
    with db_connection(readonly=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        if cur.fetchone()[0] != 1:
            raise RuntimeError("database is not ready")


if __name__ == "__main__":
    main()
