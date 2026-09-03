"""Fail-closed database activity probe for the pull deploy."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from db.connection import db_connection

EXIT_BUSY = 0
EXIT_ERROR = 2
# Python itself commonly uses 1 for import, syntax, and interpreter failures.
# A successful idle probe needs a value those pre-main failures cannot mimic.
EXIT_IDLE = 10


@dataclass(frozen=True)
class ActiveWork:
    submission_job: bool
    round: bool

    @property
    def busy(self) -> bool:
        return self.submission_job or self.round

    def labels(self) -> str:
        labels = []
        if self.submission_job:
            labels.append("submission_job")
        if self.round:
            labels.append("round")
        return ",".join(labels)


def probe_active_work() -> ActiveWork:
    """Return durable work state that makes a worker restart unsafe."""
    with db_connection(readonly=True) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              EXISTS (
                SELECT 1 FROM submission_jobs WHERE status = 'running'
              ),
              EXISTS (
                SELECT 1 FROM rounds WHERE status = 'running'
              )
            """
        )
        row = cur.fetchone()
    return ActiveWork(submission_job=bool(row[0]), round=bool(row[1]))


def main() -> int:
    """Return the deploy probe protocol code; only EXIT_IDLE permits deploy."""
    try:
        active = probe_active_work()
    except Exception:  # noqa: BLE001 - uncertainty must defer the deploy
        print("deploy: database activity probe failed", file=sys.stderr)
        return EXIT_ERROR
    if active.busy:
        print(f"deploy: active work: {active.labels()}")
        return EXIT_BUSY
    return EXIT_IDLE


if __name__ == "__main__":
    raise SystemExit(main())
