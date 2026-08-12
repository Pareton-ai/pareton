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


def _parse_json_obj(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_manifest(row: dict[str, Any]) -> CampaignManifest:
    bench = _parse_json_obj(row.get("bench"))
    engine = _parse_json_obj(row.get("engine"))
    workload_pool = _parse_json_obj(row.get("workload_pool"))
    sampling_rule = _parse_json_obj(row.get("sampling_rule"))
    z_raw = row.get("z_threshold")
    z_threshold = float(z_raw) if z_raw is not None else None
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
        priority_metric=row["priority_metric"],
        success_threshold=row["success_threshold"],
        customer_signoff=CustomerSignoff.from_dict(row.get("customer_signoff")),
        manifest_hash=row["manifest_hash"],
        bench=bench if isinstance(bench, dict) else None,
        engine=engine if isinstance(engine, dict) else None,
        workload_pool=list(workload_pool) if isinstance(workload_pool, list) else None,
        sampling_rule=dict(sampling_rule) if isinstance(sampling_rule, dict) else None,
        z_threshold=z_threshold,
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


def apply_campaign_correctness_calibration(
    campaign_id: UUID | str,
    correctness_dict: dict[str, Any],
    *,
    approver: str = "pareton-admin",
) -> str:
    """Write calibrated correctness into a draft campaign with zero submissions.

    Updates ``bench.correctness``, recomputes ``manifest_hash``, and refreshes
    ``customer_signoff`` in one transaction. Returns the new manifest_hash.
    """
    from .manifest import compute_manifest_hash, freeze_manifest_fields

    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM campaigns WHERE id = %s FOR UPDATE",
                (str(campaign_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"campaign not found: {campaign_id}")
            if str(row["status"]) != "draft":
                raise ValueError(
                    f"campaign status must be draft, got {row['status']!r}"
                )
            cur.execute(
                "SELECT COUNT(*) AS n FROM submissions WHERE campaign_id = %s",
                (str(campaign_id),),
            )
            n_subs = int(cur.fetchone()["n"])
            if n_subs != 0:
                raise ValueError(
                    f"campaign has {n_subs} submissions; calibration apply requires zero"
                )

            bench = row.get("bench")
            if isinstance(bench, str):
                bench = json.loads(bench)
            if not isinstance(bench, dict):
                raise ValueError("campaign.bench missing")
            bench = dict(bench)
            bench["correctness"] = correctness_dict

            engine = row.get("engine")
            if isinstance(engine, str):
                engine = json.loads(engine)
            if engine is not None and not isinstance(engine, dict):
                engine = None

            workload_pool = _parse_json_obj(row.get("workload_pool"))
            sampling_rule = _parse_json_obj(row.get("sampling_rule"))
            z_raw = row.get("z_threshold")
            z_threshold = float(z_raw) if z_raw is not None else None
            fields = freeze_manifest_fields(
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
                priority_metric=row["priority_metric"],
                success_threshold=row["success_threshold"],
                bench=bench,
                engine=engine if isinstance(engine, dict) else None,
                workload_pool=(
                    list(workload_pool) if isinstance(workload_pool, list) else None
                ),
                sampling_rule=(
                    dict(sampling_rule) if isinstance(sampling_rule, dict) else None
                ),
                z_threshold=z_threshold,
            )
            new_hash = compute_manifest_hash(fields)
            now = datetime.now(timezone.utc)
            signoff = CustomerSignoff(
                approved_manifest_hash=new_hash,
                approver=approver,
                timestamp=now,
            )
            cur.execute(
                """
                UPDATE campaigns
                SET bench = %s,
                    manifest_hash = %s,
                    customer_signoff = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    Json(bench),
                    new_hash,
                    Json(signoff.to_dict()),
                    str(campaign_id),
                ),
            )
            return new_hash


def apply_campaign_z_calibration(
    campaign_id: UUID | str,
    calibration: dict[str, Any],
    *,
    approver: str = "pareton-admin",
) -> str:
    """Write z-score distribution into campaigns.calibration (draft, zero subs).

    Does not change manifest_hash pin set (calibration is measured, not pinned).
    Refreshes customer_signoff timestamp only. Returns current manifest_hash.
    """
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM campaigns WHERE id = %s FOR UPDATE",
                (str(campaign_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"campaign not found: {campaign_id}")
            if str(row["status"]) != "draft":
                raise ValueError(
                    f"campaign status must be draft, got {row['status']!r}"
                )
            cur.execute(
                "SELECT COUNT(*) AS n FROM submissions WHERE campaign_id = %s",
                (str(campaign_id),),
            )
            n_subs = int(cur.fetchone()["n"])
            if n_subs != 0:
                raise ValueError(
                    f"campaign has {n_subs} submissions; calibration apply requires zero"
                )
            if not isinstance(calibration, dict) or not calibration.get("metrics"):
                raise ValueError("calibration must include metrics object")
            now = datetime.now(timezone.utc)
            signoff = CustomerSignoff(
                approved_manifest_hash=str(row["manifest_hash"]),
                approver=approver,
                timestamp=now,
            )
            cur.execute(
                """
                UPDATE campaigns
                SET calibration = %s,
                    customer_signoff = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    Json(calibration),
                    Json(signoff.to_dict()),
                    str(campaign_id),
                ),
            )
            return str(row["manifest_hash"])


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
                  manifest_hash, customer_signoff, status, bench, engine,
                  priority_metric, success_threshold,
                  workload_pool, sampling_rule, z_threshold
                ) VALUES (
                  COALESCE(%s, gen_random_uuid()), %s, %s, %s, %s,
                  %s, %s, %s, %s,
                  %s, %s,
                  %s, %s, %s, %s,
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
                    manifest.window_opens_at,
                    manifest.window_closes_at,
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
                    manifest.z_threshold,
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
                INSERT INTO submission_jobs (submission_id, kind, status)
                VALUES (%s, 'gates', 'pending')
                ON CONFLICT (submission_id, kind) DO NOTHING
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
    with db_connection() as conn:
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
                       c.workload_pool, c.sampling_rule, c.calibration, c.z_threshold,
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


def get_submission_for_campaign(
    campaign_id: UUID | str, patch_hash: str
) -> dict[str, Any] | None:
    with db_connection() as conn:
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
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM submissions WHERE patch_hash = %s",
                (patch_hash,),
            )
            return int(cur.fetchone()["n"])


KNOWN_CAMPAIGN_STATUSES: tuple[str, ...] = ("draft", "open", "closed")
KNOWN_SUBMISSION_STATES: tuple[str, ...] = (
    "committed",
    "picked_up",
    "fetched",
    "verified",
    "applied",
    "surface_ok",
    "building",
    "image_pushed",
    "built",
    "bench_queued",
    "sampled",
    "correct",
    "screened",
    "benched",
    "rejected",
)


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
    with db_connection() as conn:
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
    with db_connection() as conn:
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


def list_submission_jobs(submission_id: UUID | str) -> list[dict[str, Any]]:
    """Job rows (kind, status, last_error) for one submission, ordered by kind."""
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT kind, status, last_error
                FROM submission_jobs
                WHERE submission_id = %s
                ORDER BY kind ASC
                """,
                (str(submission_id),),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


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


BENCH_REJECT_REASONS = frozenset(
    {
        "fail_correctness",
        "fail_perf_screen",
        "fail_sla",
        "fail_engine_candidate",
        "fail_cross_env_speedup",
        "fail_promotion",
    }
)


def record_submission_sample(
    *,
    submission_id: UUID | str,
    sample_seed_block: int,
    sample_seed_block_hash: str,
    sampled_trace_sha256: str,
    sampling_receipt: dict[str, Any],
) -> None:
    """Persist realized sample columns and append a sampled event."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE submissions
                SET sample_seed_block = %s,
                    sample_seed_block_hash = %s,
                    sampled_trace_sha256 = %s,
                    sampling_receipt = %s
                WHERE id = %s
                """,
                (
                    int(sample_seed_block),
                    str(sample_seed_block_hash),
                    str(sampled_trace_sha256).lower(),
                    Json(sampling_receipt),
                    str(submission_id),
                ),
            )
            cur.execute(
                """
                INSERT INTO submission_events (submission_id, state, detail)
                VALUES (%s, 'sampled', %s)
                """,
                (str(submission_id), Json(sampling_receipt)),
            )


def derive_bench_verdict_from_events(events: list[dict[str, Any]]) -> str | None:
    """Terminal bench verdict from submission_events (WS-E event-sourced).

    ``benched`` -> ``pass``; bench ``rejected`` -> its reason; only
    ``correct``/``screened`` (or no bench events) -> ``None`` (in progress).
    """
    terminal: str | None = None
    for e in events:
        state = str(e.get("state") or "")
        detail = e.get("detail") or {}
        if not isinstance(detail, dict):
            detail = {}
        if state == "benched":
            terminal = "pass"
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
    with db_connection() as conn:
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
    with db_connection() as conn:
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


def derive_bench_verdict(reports: list[dict[str, Any]]) -> str | None:
    """Legacy report-collapse helper (pre-WS-E). Prefer event-sourced API."""
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
    """One transaction: insert bench_reports + append events + set job status.

    Each report row may carry its own ``task_id`` (WS-E per-SKU); otherwise the
    top-level ``task_id`` is used (single-SKU / legacy callers).
    """
    from psycopg2 import errors as pg_errors

    with db_connection() as conn:
        with conn.cursor() as cur:
            try:
                for row in report_rows:
                    cur.execute(
                        """
                        INSERT INTO bench_reports (
                          submission_id, task_id, stage, verdict, report,
                          evidence_s3_url, gpu_sku, mock,
                          z_scores, aggregate_z, promoted
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(submission_id),
                            str(row.get("task_id") or task_id),
                            row["stage"],
                            row["verdict"],
                            Json(row["report"]),
                            row.get("evidence_s3_url"),
                            row.get("gpu_sku") or "unknown",
                            bool(row.get("mock", False)),
                            (
                                Json(row["z_scores"])
                                if row.get("z_scores") is not None
                                else None
                            ),
                            row.get("aggregate_z"),
                            row.get("promoted"),
                        ),
                    )
            except pg_errors.UniqueViolation as exc:
                raise RuntimeError(
                    "bench_reports unique (submission_id, stage, gpu_sku) conflict; "
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
