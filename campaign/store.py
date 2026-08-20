"""Postgres loaders/writers for profiles, campaigns, submissions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from psycopg2.extras import Json, RealDictCursor

from db.connection import db_connection
from gate.types import SUBMISSION_STATES

from .manifest import build_manifest
from .models import CampaignManifest, CustomerSignoff, SLA, validate_scoring_rule


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _parse_json_obj(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_manifest(row: dict[str, Any]) -> CampaignManifest:
    bench = _parse_json_obj(row.get("bench"))
    engine = _parse_json_obj(row.get("engine"))
    workload_pool = _parse_json_obj(row.get("workload_pool"))
    sampling_rule = _parse_json_obj(row.get("sampling_rule"))
    scoring_rule = _parse_json_obj(row.get("scoring_rule"))
    return build_manifest(
        campaign_id=row["id"],
        profile_id=row.get("profile_id"),
        baseline_repo=row["baseline_repo"],
        baseline_commit=row["baseline_commit"],
        base_image_digest=row["base_image_digest"],
        gpu_skus=list(row.get("gpu_skus") or []),
        workload_trace_sha256=row["workload_trace_sha256"],
        workload_trace_url=row["workload_trace_url"],
        sla=SLA.from_dict(row.get("sla") or {}),
        scoring_config_sha256=row.get("scoring_config_sha256"),
        scoring_config_url=row.get("scoring_config_url"),
        allowed_paths=list(row.get("allowed_paths") or []),
        denied_paths=list(row.get("denied_paths") or []),
        status=row["status"],
        priority_metric=row["priority_metric"],
        success_threshold=row["success_threshold"],
        customer_signoff=CustomerSignoff.from_dict(row.get("customer_signoff")),
        manifest_hash=row["manifest_hash"],
        bench=bench if isinstance(bench, dict) else None,
        engine=engine if isinstance(engine, dict) else None,
        workload_pool=list(workload_pool) if isinstance(workload_pool, list) else None,
        sampling_rule=dict(sampling_rule) if isinstance(sampling_rule, dict) else None,
        scoring_rule=dict(scoring_rule) if isinstance(scoring_rule, dict) else None,
        created_at=(
            _parse_ts(row["created_at"]) if row.get("created_at") is not None else None
        ),
    )


def insert_profile(name: str, data: dict[str, Any]) -> UUID:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO profiles (name, data)
                VALUES (%s, %s)
                RETURNING id
                """,
                (name, Json(data)),
            )
            return cur.fetchone()[0]


def insert_campaign(manifest: CampaignManifest) -> UUID:
    signoff = (
        Json(manifest.customer_signoff.to_dict()) if manifest.customer_signoff else None
    )
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO campaigns (
                  id, profile_id, baseline_repo, baseline_commit, base_image_digest,
                  gpu_skus, workload_trace_sha256, workload_trace_url, sla,
                  scoring_config_sha256, scoring_config_url,
                  allowed_paths, denied_paths,
                  manifest_hash, customer_signoff, status, bench, engine,
                  priority_metric, success_threshold,
                  workload_pool, sampling_rule, scoring_rule
                ) VALUES (
                  COALESCE(%s, gen_random_uuid()), %s, %s, %s, %s,
                  %s, %s, %s, %s,
                  %s, %s,
                  %s, %s,
                  %s, %s, %s, %s, %s,
                  %s, %s,
                  %s, %s, %s
                )
                RETURNING id
                """,
                (
                    str(manifest.campaign_id) if manifest.campaign_id else None,
                    str(manifest.profile_id) if manifest.profile_id else None,
                    manifest.baseline_repo,
                    manifest.baseline_commit,
                    manifest.base_image_digest,
                    Json(manifest.gpu_skus),
                    manifest.workload_trace_sha256,
                    manifest.workload_trace_url,
                    Json(manifest.sla.to_dict()),
                    manifest.scoring_config_sha256,
                    manifest.scoring_config_url,
                    Json(manifest.allowed_paths),
                    Json(manifest.denied_paths),
                    manifest.manifest_hash,
                    signoff,
                    manifest.status,
                    Json(manifest.bench) if manifest.bench is not None else None,
                    Json(manifest.engine) if manifest.engine is not None else None,
                    manifest.priority_metric,
                    manifest.success_threshold,
                    (
                        Json(manifest.workload_pool)
                        if manifest.workload_pool is not None
                        else None
                    ),
                    (
                        Json(manifest.sampling_rule)
                        if manifest.sampling_rule is not None
                        else None
                    ),
                    Json(manifest.scoring_rule),
                ),
            )
            return cur.fetchone()[0]


def get_campaign(campaign_id: UUID | str) -> CampaignManifest | None:
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM campaigns WHERE id = %s", (str(campaign_id),))
            row = cur.fetchone()
    if row is None:
        return None
    return _row_to_manifest(dict(row))


def list_campaigns(*, status: str | None = None) -> list[CampaignManifest]:
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if status:
                cur.execute(
                    "SELECT * FROM campaigns WHERE status = %s ORDER BY created_at DESC",
                    (status,),
                )
            else:
                cur.execute("SELECT * FROM campaigns ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [_row_to_manifest(dict(r)) for r in rows]


def assert_scoring_rule_mutable(status: str, *, campaign_id: Any = None) -> None:
    """Refuse a scoring_rule write on a campaign that left draft.

    The rule decides how every round ranks, and it is pinned in
    manifest_hash. Changing it under a live campaign would re-rank work that
    miners already submitted against the old formula. The schema carries no
    trigger for this, by design: this is the guard.
    """
    if status != "draft":
        raise ValueError(
            f"campaign {campaign_id} is {status!r}: scoring_rule is fixed once a "
            "campaign leaves draft"
        )


def set_campaign_scoring_rule(
    campaign_id: UUID | str, rule: dict[str, Any] | None
) -> CampaignManifest:
    """Replace scoring_rule on a draft campaign. The only write path for it.

    manifest_hash covers scoring_rule, so it is recomputed in the same
    transaction; leaving the stored hash behind would describe a formula the
    campaign no longer uses.
    """
    normalized = validate_scoring_rule(rule)
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM campaigns WHERE id = %s FOR UPDATE",
                (str(campaign_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"campaign {campaign_id} not found")
            row = dict(row)
            assert_scoring_rule_mutable(str(row["status"]), campaign_id=campaign_id)
            row["scoring_rule"] = normalized
            # Force a recompute instead of carrying the stored hash forward.
            row["manifest_hash"] = None
            manifest = _row_to_manifest(row)
            cur.execute(
                """
                UPDATE campaigns
                SET scoring_rule = %s, manifest_hash = %s
                WHERE id = %s
                """,
                (Json(normalized), manifest.manifest_hash, str(campaign_id)),
            )
    return manifest


def insert_submission(
    *,
    campaign_id: UUID | str,
    patch_hash: str,
    hotkey: str,
    baseline_commit: str,
    retrieval_url: str,
    commit_block: int | None = None,
    payment_block: int | None = None,
    payment_tx: int | None = None,
) -> UUID | None:
    """Insert submission. Returns id, or None if (campaign_id, patch_hash) exists."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO submissions (
                  campaign_id, patch_hash, hotkey, baseline_commit,
                  retrieval_url, commit_block, payment_block, payment_tx
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (campaign_id, patch_hash) DO NOTHING
                RETURNING id
                """,
                (
                    str(campaign_id),
                    patch_hash,
                    hotkey,
                    baseline_commit.lower(),
                    retrieval_url,
                    commit_block,
                    payment_block,
                    payment_tx,
                ),
            )
            row = cur.fetchone()
            if row is None:
                return None
            submission_id = row[0]
            cur.execute(
                """
                INSERT INTO submission_jobs (submission_id, status)
                VALUES (%s, 'pending')
                ON CONFLICT (submission_id) DO NOTHING
                """,
                (str(submission_id),),
            )
            detail: dict[str, Any] = {"commit_block": commit_block, "hotkey": hotkey}
            if payment_block is not None:
                detail["payment_block"] = payment_block
                detail["payment_tx"] = payment_tx
            cur.execute(
                """
                INSERT INTO submission_events (submission_id, state, detail)
                VALUES (%s, 'committed', %s)
                """,
                (str(submission_id), Json(detail)),
            )
            return submission_id


def payment_ref_consumed(payment_block: int, payment_tx: int) -> bool:
    """Whether a fee payment already backs a submission."""
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM submissions
                WHERE payment_block = %s AND payment_tx = %s
                LIMIT 1
                """,
                (payment_block, payment_tx),
            )
            return cur.fetchone() is not None


def append_event(
    submission_id: UUID | str,
    state: str,
    *,
    evidence_ref: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO submission_events (submission_id, state, evidence_ref, detail)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    str(submission_id),
                    state,
                    evidence_ref,
                    Json(detail or {}),
                ),
            )


def set_job_status(
    submission_id: UUID | str,
    status: str,
    *,
    job_id: int | None = None,
    last_error: str | None = None,
    bump_attempts: bool = False,
) -> None:
    """Update one job row. Prefer job_id; else submission_id.

    Also clears live activity so a settled job cannot look like it is still running.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            if job_id is not None:
                where = "WHERE id = %s"
                where_args: tuple[Any, ...] = (job_id,)
            else:
                where = "WHERE submission_id = %s"
                where_args = (str(submission_id),)
            attempts_set = "attempts = attempts + 1," if bump_attempts else ""
            cur.execute(
                f"""
                UPDATE submission_jobs
                SET status = %s,
                    last_error = %s,
                    {attempts_set}
                    phase = NULL,
                    phase_started_at = NULL,
                    heartbeat_at = NULL,
                    progress = NULL,
                    updated_at = now()
                {where}
                """,
                (status, last_error, *where_args),
            )


def set_job_phase(
    *,
    job_id: int,
    attempt: int,
    phase: str,
    progress: dict[str, Any] | None = None,
) -> bool:
    """Record what a running attempt is doing now. Returns whether it landed.

    Matches (job_id, attempt, status=running) so a superseded attempt updates zero rows.
    `phase_started_at` moves only when the phase actually changes.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE submission_jobs
                SET phase = %s,
                    phase_started_at = CASE
                      WHEN phase IS DISTINCT FROM %s THEN now()
                      ELSE phase_started_at
                    END,
                    heartbeat_at = now(),
                    progress = %s,
                    updated_at = now()
                WHERE id = %s AND attempts = %s AND status = 'running'
                """,
                (
                    phase,
                    phase,
                    Json(progress) if progress else None,
                    job_id,
                    attempt,
                ),
            )
            return cur.rowcount > 0


def touch_job_heartbeat(*, job_id: int, attempt: int) -> bool:
    """Confirm the attempt is still alive. Returns whether it landed."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE submission_jobs
                SET heartbeat_at = now(), updated_at = now()
                WHERE id = %s AND attempts = %s AND status = 'running'
                """,
                (job_id, attempt),
            )
            return cur.rowcount > 0


def set_engine_image(submission_id: UUID | str, image_ref: str) -> None:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE submissions
                SET engine_image_ref = %s
                WHERE id = %s
                """,
                (image_ref, str(submission_id)),
            )


def submission_has_terminal_event(submission_id: UUID | str) -> bool:
    """True if a terminal state was already recorded (blocks requeue)."""
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM submission_events
                WHERE submission_id = %s AND state IN ('rejected', 'scored', 'disqualified')
                LIMIT 1
                """,
                (str(submission_id),),
            )
            return cur.fetchone() is not None


def complete_gates_job(
    submission_id: UUID | str,
    *,
    job_id: int | None = None,
    enqueue_round: bool = False,
) -> None:
    """Mark the gates job done, and queue the submission for a round.

    Bench used to be a second job row enqueued here. Rounds replaced it: a
    submission that reaches ``bench_queued`` waits for the round creator to
    pick it up; ``round.create`` selects each cohort from that state.

    The ``bench_queued`` event IS that queue, so it is written in the same
    transaction as the completion. Split over two transactions, a crash
    between them settles the job while leaving the submission in no queue at
    all: only 'pending' jobs are reclaimed, and the round creator only reads
    'bench_queued'.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            if job_id is not None:
                cur.execute(
                    """
                    UPDATE submission_jobs
                    SET status = 'done', last_error = NULL, updated_at = now()
                    WHERE id = %s
                    """,
                    (job_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE submission_jobs
                    SET status = 'done', last_error = NULL, updated_at = now()
                    WHERE submission_id = %s
                    """,
                    (str(submission_id),),
                )
            if enqueue_round:
                cur.execute(
                    """
                    INSERT INTO submission_events (submission_id, state, detail)
                    VALUES (%s, 'bench_queued', %s)
                    """,
                    (str(submission_id), Json({})),
                )


def count_pending_jobs() -> int:
    """How many gate jobs are waiting to be claimed. Read-only, for observability."""
    with db_connection(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM submission_jobs WHERE status = 'pending'")
            return int(cur.fetchone()[0])


def claim_next_job() -> dict[str, Any] | None:
    """Atomically claim the oldest pending gates job."""
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT j.id AS job_id, j.submission_id
                FROM submission_jobs j
                WHERE j.status = 'pending'
                ORDER BY j.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            )
            job = cur.fetchone()
            if job is None:
                return None
            # New attempt: drop the previous phase so it cannot look current.
            cur.execute(
                """
                UPDATE submission_jobs
                SET status = 'running',
                    attempts = attempts + 1,
                    phase = NULL,
                    phase_started_at = NULL,
                    heartbeat_at = now(),
                    progress = NULL,
                    updated_at = now()
                WHERE id = %s
                RETURNING attempts
                """,
                (job["job_id"],),
            )
            attempt = int(cur.fetchone()["attempts"])
            cur.execute(
                """
                SELECT s.*, c.baseline_commit AS campaign_baseline_commit,
                       c.baseline_repo, c.base_image_digest,
                       c.allowed_paths, c.denied_paths, c.status AS campaign_status,
                       c.bench, c.sla, c.workload_trace_url, c.workload_trace_sha256,
                       c.gpu_skus, c.manifest_hash,
                       c.workload_pool, c.sampling_rule, c.scoring_rule,
                       c.engine,
                       %s::bigint AS job_id,
                       %s::int AS job_attempt
                FROM submissions s
                JOIN campaigns c ON c.id = s.campaign_id
                WHERE s.id = %s
                """,
                (job["job_id"], attempt, str(job["submission_id"])),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def get_submission(patch_hash: str) -> dict[str, Any] | None:
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM submissions
                WHERE patch_hash = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (patch_hash,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def get_submission_for_campaign(
    campaign_id: UUID | str, patch_hash: str
) -> dict[str, Any] | None:
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM submissions
                WHERE campaign_id = %s AND patch_hash = %s
                """,
                (str(campaign_id), patch_hash),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def count_submission_campaigns(patch_hash: str) -> int:
    """Campaign count holding patch_hash; >1 means bare-hash lookup is ambiguous."""
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM submissions WHERE patch_hash = %s",
                (patch_hash,),
            )
            return int(cur.fetchone()["n"])


KNOWN_CAMPAIGN_STATUSES: tuple[str, ...] = ("draft", "open", "closed")
# Derived, not copied. Add states in gate/types.py only (PAR-46).
KNOWN_SUBMISSION_STATES: tuple[str, ...] = SUBMISSION_STATES


def list_submissions(
    campaign_id: UUID | str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated submissions for a campaign.

    Ordered by ``committed_at DESC, id DESC``. Returns
    ``{"total": int, "items": [row, ...]}``.
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM submissions WHERE campaign_id = %s",
                (str(campaign_id),),
            )
            total = int(cur.fetchone()["n"])
            cur.execute(
                """
                SELECT id, campaign_id, patch_hash, hotkey, baseline_commit,
                       retrieval_url, commit_block, committed_at, engine_image_ref
                FROM submissions
                WHERE campaign_id = %s
                ORDER BY committed_at DESC, id DESC
                LIMIT %s OFFSET %s
                """,
                (str(campaign_id), int(limit), int(offset)),
            )
            rows = cur.fetchall()
    return {"total": total, "items": [dict(r) for r in rows]}


def list_latest_states(
    submission_ids: list[UUID | str],
) -> dict[str, str | None]:
    """Map submission_id -> latest event state (created_at DESC, id DESC)."""
    if not submission_ids:
        return {}
    ids = [str(s) for s in submission_ids]
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (submission_id) submission_id, state
                FROM submission_events
                WHERE submission_id = ANY(%s::uuid[])
                ORDER BY submission_id, created_at DESC, id DESC
                """,
                (ids,),
            )
            rows = cur.fetchall()
    out: dict[str, str | None] = {sid: None for sid in ids}
    for r in rows:
        out[str(r["submission_id"])] = str(r["state"])
    return out


# TODO(PAR-80): live bench phase now lives on rounds.phase, not on a
# per-submission job row. The round read API replaces list_live_bench_phases.


def list_events(submission_id: UUID | str) -> list[dict[str, Any]]:
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT state, evidence_ref, detail, created_at
                FROM submission_events
                WHERE submission_id = %s
                ORDER BY created_at ASC
                """,
                (str(submission_id),),
            )
            rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("detail"), str):
            d["detail"] = json.loads(d["detail"])
        out.append(d)
    return out


def list_submission_jobs(submission_id: UUID | str) -> list[dict[str, Any]]:
    """Gate job rows for one submission."""
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT status, last_error,
                       phase, phase_started_at, heartbeat_at, progress
                FROM submission_jobs
                WHERE submission_id = %s
                ORDER BY id ASC
                """,
                (str(submission_id),),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


# TODO(PAR-80): round_entries replaces bench_reports. Entry reads live in
# round/store.py.


BENCH_REJECT_REASONS = frozenset(
    {
        "fail_correctness",
        "fail_sla",
        "fail_engine_candidate",
        "duplicate_image",
    }
)


# Sampling is per round: the realized trace is snapshotted onto rounds, not
# onto submissions. See round/create.py.


def derive_bench_verdict_from_events(events: list[dict[str, Any]]) -> str | None:
    """Terminal round verdict from submission_events.

    ``scored`` -> ``pass``; ``disqualified`` or a bench ``rejected`` -> its
    reason; anything earlier -> ``None`` (in progress).
    """
    terminal: str | None = None
    for e in events:
        state = str(e.get("state") or "")
        detail = e.get("detail") or {}
        if not isinstance(detail, dict):
            detail = {}
        if state == "scored":
            terminal = "pass"
        elif state == "disqualified":
            terminal = str(detail.get("reason") or "disqualified")
        elif state == "rejected":
            reason = detail.get("reason")
            if reason in BENCH_REJECT_REASONS:
                terminal = str(reason)
    return terminal


def list_bench_summaries(
    campaign_id: UUID | str,
    submission_ids: list[UUID | str] | None = None,
) -> dict[str, str | None]:
    """Map submission_id -> event-sourced bench_verdict for campaign list API.

    When ``submission_ids`` is set, only those rows are loaded (page-scoped).
    """
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if submission_ids is not None:
                ids = [str(s) for s in submission_ids]
                if not ids:
                    return {}
                cur.execute(
                    """
                    SELECT s.id AS submission_id, e.state, e.detail, e.created_at
                    FROM submissions s
                    LEFT JOIN submission_events e ON e.submission_id = s.id
                    WHERE s.campaign_id = %s
                      AND s.id = ANY(%s::uuid[])
                    ORDER BY s.id, e.created_at ASC NULLS LAST
                    """,
                    (str(campaign_id), ids),
                )
            else:
                cur.execute(
                    """
                    SELECT s.id AS submission_id, e.state, e.detail, e.created_at
                    FROM submissions s
                    LEFT JOIN submission_events e ON e.submission_id = s.id
                    WHERE s.campaign_id = %s
                    ORDER BY s.id, e.created_at ASC NULLS LAST
                    """,
                    (str(campaign_id),),
                )
            rows = cur.fetchall()
    by_sub: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        sid = str(r["submission_id"])
        by_sub.setdefault(sid, [])
        if r.get("state") is None:
            continue
        detail = r.get("detail")
        if isinstance(detail, str):
            detail = json.loads(detail)
        by_sub[sid].append({"state": r["state"], "detail": detail or {}})
    return {sid: derive_bench_verdict_from_events(evts) for sid, evts in by_sub.items()}


def get_public_stats() -> dict[str, Any]:
    """Campaign status counts + submission counts by latest event state."""
    by_status = {s: 0 for s in KNOWN_CAMPAIGN_STATUSES}
    by_latest_state = {s: 0 for s in KNOWN_SUBMISSION_STATES}
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM campaigns
                GROUP BY status
                """
            )
            for r in cur.fetchall():
                status = str(r["status"])
                if status in by_status:
                    by_status[status] = int(r["n"])
            cur.execute("SELECT COUNT(*) AS n FROM campaigns")
            campaigns_total = int(cur.fetchone()["n"])
            cur.execute("SELECT COUNT(*) AS n FROM submissions")
            submissions_total = int(cur.fetchone()["n"])
            cur.execute(
                """
                SELECT latest.state AS latest_state, COUNT(*) AS n
                FROM submissions s
                LEFT JOIN LATERAL (
                    SELECT state
                    FROM submission_events e
                    WHERE e.submission_id = s.id
                    ORDER BY e.created_at DESC, e.id DESC
                    LIMIT 1
                ) latest ON true
                WHERE latest.state IS NOT NULL
                GROUP BY latest.state
                """
            )
            for r in cur.fetchall():
                state = str(r["latest_state"])
                if state in by_latest_state:
                    by_latest_state[state] = int(r["n"])
    return {
        "campaigns": {"total": campaigns_total, "by_status": by_status},
        "submissions": {
            "total": submissions_total,
            "by_latest_state": by_latest_state,
        },
    }
