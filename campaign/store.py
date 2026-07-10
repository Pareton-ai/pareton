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
        Json(manifest.customer_signoff.to_dict())
        if manifest.customer_signoff
        else None
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
                  manifest_hash, customer_signoff, status
                ) VALUES (
                  COALESCE(%s, gen_random_uuid()), %s, %s, %s, %s,
                  %s, %s, %s, %s,
                  %s, %s,
                  %s, %s, %s, %s,
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
                    manifest.window_opens_at,
                    manifest.window_closes_at,
                    manifest.manifest_hash,
                    signoff,
                    manifest.status,
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
                INSERT INTO submission_jobs (submission_id, status)
                VALUES (%s, 'pending')
                ON CONFLICT (submission_id) DO NOTHING
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
    last_error: str | None = None,
    bump_attempts: bool = False,
) -> None:
    with db_connection() as conn:
        with conn.cursor() as cur:
            if bump_attempts:
                cur.execute(
                    """
                    UPDATE submission_jobs
                    SET status = %s,
                        last_error = %s,
                        attempts = attempts + 1,
                        updated_at = now()
                    WHERE submission_id = %s
                    """,
                    (status, last_error, str(submission_id)),
                )
            else:
                cur.execute(
                    """
                    UPDATE submission_jobs
                    SET status = %s, last_error = %s, updated_at = now()
                    WHERE submission_id = %s
                    """,
                    (status, last_error, str(submission_id)),
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


def claim_next_job() -> dict[str, Any] | None:
    """Atomically claim the oldest pending job. Returns submission+campaign fields."""
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
                       c.window_opens_at, c.window_closes_at
                FROM submissions s
                JOIN campaigns c ON c.id = s.campaign_id
                WHERE s.id = %s
                """,
                (str(job["submission_id"]),),
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
