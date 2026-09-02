"""Fail-closed database activity probe for the pull deploy."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from db.connection import db_connection


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
    """Exit 0 when busy, 1 when idle, and 2 when the probe cannot decide."""
    try:
        active = probe_active_work()
    except Exception:  # noqa: BLE001 - uncertainty must defer the deploy
        print("deploy: database activity probe failed", file=sys.stderr)
        return 2
    if active.busy:
        print(f"deploy: active work: {active.labels()}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
