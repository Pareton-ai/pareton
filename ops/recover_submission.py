#!/usr/bin/env python3
"""Residual submission recovery CLI (stage-2 spec section 5.1).

For submission_jobs left in 'running' by an OOM-killed or crashed worker:
read-only ``inspect`` and an explicit ``recover`` with two evidence-gated
outcomes — ``requeue`` (infra-interrupted, nothing durable happened) and
``settle`` (reliable durable evidence already exists). This is the human
recovery entry the deploy tick points at; deploy ticks never call recover
automatically.

Run with the application venv:

    /opt/pareton/.venv/bin/python ops/recover_submission.py inspect --job 42
    /opt/pareton/.venv/bin/python ops/recover_submission.py recover --job 42 \
        --attempt 3 --outcome requeue --operator NAME --reason "..."

recover only runs under release coordination: deploy mutex held, claim gate
closed (draining/quiescing), worker activity lock exclusive — the same
ordering as the deploy tick (spec 4.1). Exit codes: 0 recovered, 1 refused
or already changed, 2 error.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from db.connection import db_connection

TERMINAL_REJECT_STATES = ("rejected", "rejected_duplicate", "disqualified")
PROGRESS_STATES = ("bench_queued", "scored")


def deploy_lock_path() -> Path:
    return Path(os.environ.get("PARETON_DEPLOY_LOCK", "/run/pareton-deploy.lock"))


def activity_lock_path() -> Path:
    return Path(os.environ.get("PARETON_ACTIVITY_LOCK", "/run/pareton-activity.lock"))


def state_path() -> Path:
    return Path(
        os.environ.get(
            "PARETON_RELEASE_STATE", "/var/lib/pareton-deploy/release-state.json"
        )
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_gate_phase() -> str | None:
    try:
        state = json.loads(state_path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    phase = state.get("phase")
    return phase if phase in ("draining", "quiescing") else None


@contextmanager
def coordination():
    """Deploy mutex -> gate closed -> worker activity exclusive (spec 4.1)."""
    deploy_lock_path().parent.mkdir(parents=True, exist_ok=True)
    deploy_fd = os.open(str(deploy_lock_path()), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(deploy_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(deploy_fd)
        raise SystemExit("recover: a deploy holds the mutex; retry after it finishes")
    try:
        phase = read_gate_phase()
        if phase is None:
            raise SystemExit(
                "recover: claim gate must be closed (draining/quiescing) and "
                "state readable; run the deploy tick first so it reports the "
                "residual record"
            )
        activity_lock_path().parent.mkdir(parents=True, exist_ok=True)
        activity_fd = os.open(str(activity_lock_path()), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(activity_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(activity_fd)
            raise SystemExit(
                "recover: a worker still holds the activity lock; an executor "
                "may be live — refusing (spec 5.1)"
            )
        try:
            yield phase
        finally:
            fcntl.flock(activity_fd, fcntl.LOCK_UN)
            os.close(activity_fd)
    finally:
        fcntl.flock(deploy_fd, fcntl.LOCK_UN)
        os.close(deploy_fd)


def load_job(cur, job_id: int) -> dict | None:
    cur.execute(
        """
        SELECT id, submission_id, status, attempts, phase, heartbeat_at,
               progress, last_error, created_at, updated_at
        FROM submission_jobs WHERE id = %s
        """,
        (job_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    keys = (
        "id",
        "submission_id",
        "status",
        "attempts",
        "phase",
        "heartbeat_at",
        "progress",
        "last_error",
        "created_at",
        "updated_at",
    )
    return dict(zip(keys, row))


def load_states(cur, submission_id) -> list[str]:
    cur.execute(
        """
        SELECT state FROM submission_events
        WHERE submission_id = %s
          AND state IN ('rejected', 'rejected_duplicate', 'disqualified',
                        'bench_queued', 'scored')
        ORDER BY created_at
        """,
        (str(submission_id),),
    )
    return [row[0] for row in cur.fetchall()]


def evidence_class(states: list[str]) -> str:
    rejected = [s for s in states if s in TERMINAL_REJECT_STATES]
    progressed = [s for s in states if s in PROGRESS_STATES]
    if rejected and progressed:
        return "contradictory"
    if rejected:
        return "terminal-reject"
    if progressed:
        return "progressed"
    return "none"


def cmd_inspect(args: argparse.Namespace) -> int:
    with db_connection(readonly=True) as conn, conn.cursor() as cur:
        job = load_job(cur, args.job)
        if job is None:
            print(f"inspect: job {args.job} not found")
            return 1
        states = load_states(cur, job["submission_id"])
    payload = {
        **{k: str(v) for k, v in job.items()},
        "evidence_states": states,
        "evidence_class": evidence_class(states),
        "note": (
            "lock release does not prove Docker builds or remote resources "
            "ended; confirm externally before recover (spec 5.1)"
        ),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    if args.outcome not in ("requeue", "settle"):
        print("recover: --outcome must be requeue or settle", file=sys.stderr)
        return 2
    with coordination() as phase, db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, submission_id, status, attempts FROM "
            "submission_jobs WHERE id = %s FOR UPDATE",
            (args.job,),
        )
        row = cur.fetchone()
        if row is None:
            print(f"recover: job {args.job} not found")
            return 1
        job_id, submission_id, status, attempts = row
        if status != "running":
            print(
                f"recover: job {job_id} status is {status!r}, "
                "not running; nothing to recover"
            )
            return 1
        if attempts != args.attempt:
            print(
                f"recover: attempt moved to {attempts} (given "
                f"{args.attempt}); re-inspect and retry"
            )
            return 1
        states = load_states(cur, submission_id)
        klass = evidence_class(states)
        target = _decide(args.outcome, klass, states)
        if target is None:
            print(
                f"recover: outcome {args.outcome} refused for evidence "
                f"class {klass} ({states}); spec 5.1 rules",
                file=sys.stderr,
            )
            return 1
        cur.execute(
            """
                    UPDATE submission_jobs
                    SET status = %s,
                        last_error = %s,
                        phase = NULL,
                        phase_started_at = NULL,
                        heartbeat_at = NULL,
                        progress = NULL,
                        updated_at = now()
                    WHERE id = %s AND attempts = %s AND status = 'running'
                    """,
            (target, args.reason, args.job, args.attempt),
        )
        if cur.rowcount != 1:
            print(
                f"recover: job {job_id} changed concurrently "
                f"(rowcount {cur.rowcount}); not recovered"
            )
            return 1
        cur.execute(
            """
                    INSERT INTO submission_events (submission_id, state, detail)
                    VALUES (%s, 'recovered', %s)
                    """,
            (
                str(submission_id),
                json.dumps(
                    {
                        "job_id": job_id,
                        "attempt": args.attempt,
                        "outcome": args.outcome,
                        "job_status": target,
                        "operator": args.operator,
                        "reason": args.reason,
                        "gate_phase": phase,
                        "at": now_iso(),
                    }
                ),
            ),
        )
    print(
        f"recover: job {args.job} attempt {args.attempt} -> {target} "
        f"({args.outcome}); repeat runs will not re-enqueue"
    )
    return 0


def _decide(outcome: str, klass: str, states: list[str]) -> str | None:
    """Map (requested outcome, evidence) to a job status, or refuse.

    requeue: only with no durable evidence at all — a rerun may build again,
    which the operator explicitly accepts; it never emits miner-visible
    rejection events.
    settle: terminal-reject evidence -> failed; progressed past gates
    (bench_queued/scored) -> done. Contradictory evidence refuses.
    """
    if klass == "contradictory":
        return None
    if outcome == "requeue":
        return "pending" if klass == "none" else None
    if outcome == "settle":
        if klass == "terminal-reject":
            return "failed"
        if klass == "progressed":
            return "done"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--job", type=int, required=True)
    recover = sub.add_parser("recover")
    recover.add_argument("--job", type=int, required=True)
    recover.add_argument("--attempt", type=int, required=True)
    recover.add_argument("--outcome", required=True, choices=("requeue", "settle"))
    recover.add_argument("--operator", required=True)
    recover.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    if args.mode == "inspect":
        return cmd_inspect(args)
    return cmd_recover(args)


if __name__ == "__main__":
    sys.exit(main())
