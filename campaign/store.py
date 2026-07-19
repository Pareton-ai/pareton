"""Postgres loaders/writers for profiles, campaigns, submissions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from psycopg2.extras import Json, RealDictCursor

from db.connection import db_connection

from .manifest import build_manifest
from .models import CampaignManifest, CustomerSignoff, SLA


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _row_to_manifest(row: dict[str, Any]) -> CampaignManifest:
    bench = row.get("bench")
    if isinstance(bench, str):
        bench = json.loads(bench)
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
        window_opens_at=_parse_ts(row["window_opens_at"]),
        window_closes_at=_parse_ts(row["window_closes_at"]),
        status=row["status"],
        customer_signoff=CustomerSignoff.from_dict(row.get("customer_signoff")),
        manifest_hash=row["manifest_hash"],
        bench=bench if isinstance(bench, dict) else None,
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
                  allowed_paths, denied_paths, window_opens_at, window_closes_at,
                  manifest_hash, customer_signoff, status, bench
                ) VALUES (
                  COALESCE(%s, gen_random_uuid()), %s, %s, %s, %s,
                  %s, %s, %s, %s,
                  %s, %s,
                  %s, %s, %s, %s,
                  %s, %s, %s, %s
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
                    manifest.window_opens_at,
                    manifest.window_closes_at,
                    manifest.manifest_hash,
                    signoff,
                    manifest.status,
                    Json(manifest.bench) if manifest.bench is not None else None,
                ),
            )
            return cur.fetchone()[0]


def get_campaign(campaign_id: UUID | str) -> CampaignManifest | None:
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM campaigns WHERE id = %s", (str(campaign_id),))
            row = cur.fetchone()
    if row is None:
        return None
    return _row_to_manifest(dict(row))


def list_campaigns(*, status: str | None = None) -> list[CampaignManifest]:
    with db_connection() as conn:
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


def insert_submission(
    *,
    campaign_id: UUID | str,
    patch_hash: str,
    hotkey: str,
    baseline_commit: str,
    retrieval_url: str,
    commit_block: int | None = None,
) -> UUID | None:
    """Insert submission. Returns id, or None if (campaign_id, patch_hash) exists."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO submissions (
                  campaign_id, patch_hash, hotkey, baseline_commit,
                  retrieval_url, commit_block
                ) VALUES (%s, %s, %s, %s, %s, %s)
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
                ),
            )
            row = cur.fetchone()
            if row is None:
                return None
            submission_id = row[0]
            cur.execute(
                """
                INSERT INTO submission_jobs (submission_id, kind, status)
                VALUES (%s, 'gates', 'pending')
                ON CONFLICT (submission_id, kind) DO NOTHING
                """,
                (str(submission_id),),
            )
            cur.execute(
                """
                INSERT INTO submission_events (submission_id, state, detail)
                VALUES (%s, 'committed', %s)
                """,
                (
                    str(submission_id),
                    Json({"commit_block": commit_block, "hotkey": hotkey}),
                ),
            )
            return submission_id


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
    kind: str = "gates",
    job_id: int | None = None,
    last_error: str | None = None,
    bump_attempts: bool = False,
) -> None:
    """Update one job row. Prefer job_id; else (submission_id, kind)."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            if job_id is not None:
                where = "WHERE id = %s"
                where_args: tuple[Any, ...] = (job_id,)
            else:
                where = "WHERE submission_id = %s AND kind = %s"
                where_args = (str(submission_id), kind)
            if bump_attempts:
                cur.execute(
                    f"""
                    UPDATE submission_jobs
                    SET status = %s,
                        last_error = %s,
                        attempts = attempts + 1,
                        updated_at = now()
                    {where}
                    """,
                    (status, last_error, *where_args),
                )
            else:
                cur.execute(
                    f"""
                    UPDATE submission_jobs
                    SET status = %s, last_error = %s, updated_at = now()
                    {where}
                    """,
                    (status, last_error, *where_args),
                )


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
    """True if rejected or benched already recorded (blocks rebench)."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM submission_events
                WHERE submission_id = %s AND state IN ('rejected', 'benched')
                LIMIT 1
                """,
                (str(submission_id),),
            )
            return cur.fetchone() is not None


def enqueue_bench_job(submission_id: UUID | str) -> bool:
    """Enqueue kind=bench pending. Returns False if terminal or already exists."""
    if submission_has_terminal_event(submission_id):
        return False
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO submission_jobs (submission_id, kind, status)
                VALUES (%s, 'bench', 'pending')
                ON CONFLICT (submission_id, kind) DO NOTHING
                RETURNING id
                """,
                (str(submission_id),),
            )
            return cur.fetchone() is not None


def complete_gates_job(
    submission_id: UUID | str,
    *,
    job_id: int | None = None,
    enqueue_bench: bool = False,
) -> bool:
    """Mark gates job done and optionally enqueue bench in one transaction.

    Avoids the crash window where gates is ``done`` but no bench row exists.
    Returns whether a new bench job row was inserted.
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
                    WHERE submission_id = %s AND kind = 'gates'
                    """,
                    (str(submission_id),),
                )
            if not enqueue_bench:
                return False
            cur.execute(
                """
                SELECT 1 FROM submission_events
                WHERE submission_id = %s AND state IN ('rejected', 'benched')
                LIMIT 1
                """,
                (str(submission_id),),
            )
            if cur.fetchone() is not None:
                return False
            cur.execute(
                """
                INSERT INTO submission_jobs (submission_id, kind, status)
                VALUES (%s, 'bench', 'pending')
                ON CONFLICT (submission_id, kind) DO NOTHING
                RETURNING id
                """,
                (str(submission_id),),
            )
            return cur.fetchone() is not None


def claim_next_job(*, kind: str = "gates") -> dict[str, Any] | None:
    """Atomically claim the oldest pending job of ``kind``."""
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT j.id AS job_id, j.submission_id, j.kind
                FROM submission_jobs j
                WHERE j.status = 'pending' AND j.kind = %s
                ORDER BY j.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (kind,),
            )
            job = cur.fetchone()
            if job is None:
                return None
            if kind == "bench":
                cur.execute(
                    """
                    SELECT 1 FROM submission_events
                    WHERE submission_id = %s AND state IN ('rejected', 'benched')
                    LIMIT 1
                    """,
                    (str(job["submission_id"]),),
                )
                if cur.fetchone() is not None:
                    cur.execute(
                        """
                        UPDATE submission_jobs
                        SET status = 'failed',
                            last_error = 'submission_terminal',
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (job["job_id"],),
                    )
                    return None
            cur.execute(
                """
                UPDATE submission_jobs
                SET status = 'running', attempts = attempts + 1, updated_at = now()
                WHERE id = %s
                """,
                (job["job_id"],),
            )
            cur.execute(
                """
                SELECT s.*, c.baseline_commit AS campaign_baseline_commit,
                       c.baseline_repo, c.base_image_digest,
                       c.allowed_paths, c.denied_paths, c.status AS campaign_status,
                       c.window_opens_at, c.window_closes_at,
                       c.bench, c.sla, c.workload_trace_url, c.workload_trace_sha256,
                       c.gpu_skus, c.manifest_hash,
                       %s::bigint AS job_id, %s::text AS job_kind
                FROM submissions s
                JOIN campaigns c ON c.id = s.campaign_id
                WHERE s.id = %s
                """,
                (job["job_id"], job["kind"], str(job["submission_id"])),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def get_submission(patch_hash: str) -> dict[str, Any] | None:
    with db_connection() as conn:
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


def list_submissions(campaign_id: UUID | str) -> list[dict[str, Any]]:
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, campaign_id, patch_hash, hotkey, baseline_commit,
                       retrieval_url, commit_block, committed_at, engine_image_ref
                FROM submissions
                WHERE campaign_id = %s
                ORDER BY committed_at DESC
                """,
                (str(campaign_id),),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def list_events(submission_id: UUID | str) -> list[dict[str, Any]]:
    with db_connection() as conn:
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


def insert_bench_report(
    *,
    submission_id: UUID | str,
    task_id: str,
    stage: str,
    verdict: str,
    report: dict[str, Any],
    evidence_s3_url: str | None = None,
    gpu_sku: str | None = None,
    mock: bool = False,
) -> None:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bench_reports (
                  submission_id, task_id, stage, verdict, report,
                  evidence_s3_url, gpu_sku, mock
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(submission_id),
                    task_id,
                    stage,
                    verdict,
                    Json(report),
                    evidence_s3_url,
                    gpu_sku,
                    mock,
                ),
            )


def list_bench_reports(submission_id: UUID | str) -> list[dict[str, Any]]:
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT task_id, stage, verdict, report, evidence_s3_url,
                       gpu_sku, mock, created_at
                FROM bench_reports
                WHERE submission_id = %s
                ORDER BY created_at ASC
                """,
                (str(submission_id),),
            )
            rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("report"), str):
            d["report"] = json.loads(d["report"])
        out.append(d)
    return out


def list_bench_summaries(campaign_id: UUID | str) -> dict[str, str | None]:
    """Map submission_id -> derived bench_verdict for campaign list API."""
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT s.id AS submission_id, br.stage, br.verdict
                FROM submissions s
                LEFT JOIN bench_reports br ON br.submission_id = s.id
                WHERE s.campaign_id = %s
                ORDER BY s.id, br.created_at ASC
                """,
                (str(campaign_id),),
            )
            rows = cur.fetchall()
    by_sub: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        sid = str(r["submission_id"])
        by_sub.setdefault(sid, [])
        if r.get("stage") is not None:
            by_sub[sid].append({"stage": r["stage"], "verdict": r["verdict"]})
    return {sid: derive_bench_verdict(reps) for sid, reps in by_sub.items()}


def derive_bench_verdict(reports: list[dict[str, Any]]) -> str | None:
    """pass iff all present stage verdicts pass and sla_bench exists; else first fail."""
    if not reports:
        return None
    stage_order = ("correctness", "perf_screen", "sla_bench")
    by_stage = {r["stage"]: r["verdict"] for r in reports if r.get("stage")}
    for stage in stage_order:
        v = by_stage.get(stage)
        if v is None:
            continue
        if v != "pass":
            return str(v)
    if "sla_bench" in by_stage and all(by_stage.get(s) == "pass" for s in by_stage):
        return "pass"
    return None


def finalize_bench_job(
    *,
    submission_id: UUID | str,
    job_id: int,
    task_id: str,
    report_rows: list[dict[str, Any]],
    events: list[tuple[str, dict[str, Any]]],
    job_status: str,
    last_error: str | None = None,
) -> None:
    """One transaction: insert bench_reports + append events + set job status."""
    from psycopg2 import errors as pg_errors

    with db_connection() as conn:
        with conn.cursor() as cur:
            try:
                for row in report_rows:
                    cur.execute(
                        """
                        INSERT INTO bench_reports (
                          submission_id, task_id, stage, verdict, report,
                          evidence_s3_url, gpu_sku, mock
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(submission_id),
                            task_id,
                            row["stage"],
                            row["verdict"],
                            Json(row["report"]),
                            row.get("evidence_s3_url"),
                            row.get("gpu_sku"),
                            bool(row.get("mock", False)),
                        ),
                    )
            except pg_errors.UniqueViolation as exc:
                raise RuntimeError(
                    "bench_reports unique (submission_id, stage) conflict; "
                    "delete stale rows before requeue"
                ) from exc
            for state, detail in events:
                cur.execute(
                    """
                    INSERT INTO submission_events
                      (submission_id, state, evidence_ref, detail)
                    VALUES (%s, %s, NULL, %s)
                    """,
                    (str(submission_id), state, Json(detail)),
                )
            cur.execute(
                """
                UPDATE submission_jobs
                SET status = %s, last_error = %s, updated_at = now()
                WHERE id = %s
                """,
                (job_status, last_error, job_id),
            )
