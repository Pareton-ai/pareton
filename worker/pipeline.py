"""Run gates a–e for one claimed submission job."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import config
from builder.hermetic import build_engine_image, build_engine_image_local_mock
from campaign.store import (
    append_event,
    enqueue_bench_job,
    get_campaign,
    set_engine_image,
    set_job_status,
)
from gate.base_apply import check_base_apply, check_base_apply_local
from gate.identity import check_identity
from gate.integrity import check_integrity
from gate.surface import check_surface
from gate.types import GateResult, SubmissionState

logger = logging.getLogger(__name__)


def process_submission(
    row: dict[str, Any],
    *,
    registered_hotkeys: Iterable[str],
    fetcher: Callable[[str], bytes] | None = None,
    mock_build: bool = False,
    local_repo: Path | None = None,
    work_root: Path | None = None,
) -> GateResult:
    """Execute Stage 0 gates fail-fast; persist events along the way."""
    submission_id = str(row["id"])
    campaign_id = str(row["campaign_id"])
    hotkey = row["hotkey"]
    patch_hash = row["patch_hash"]
    retrieval_url = row["retrieval_url"]
    baseline_commit = row["baseline_commit"]
    job_id = row.get("job_id")

    campaign = get_campaign(campaign_id)
    if campaign is None:
        result = GateResult.reject("campaign_missing", campaign_id=campaign_id)
        _fail(submission_id, result, job_id=job_id)
        return result

    # a. Identity (fail-fast; success has no dedicated state in the machine)
    id_res = check_identity(
        hotkey=hotkey,
        registered_hotkeys=registered_hotkeys,
        campaign=campaign,
        baseline_commit=baseline_commit,
        now=datetime.now(timezone.utc),
    )
    if not id_res.ok:
        _fail(submission_id, id_res, job_id=job_id)
        return id_res

    # b. Integrity → fetched → verified
    integrity_kwargs: dict[str, Any] = {
        "retrieval_url": retrieval_url,
        "expected_patch_hash": patch_hash,
    }
    if fetcher is not None:
        integrity_kwargs["fetcher"] = fetcher
    int_res = check_integrity(**integrity_kwargs)
    if not int_res.ok:
        _fail(submission_id, int_res, job_id=job_id)
        return int_res
    patch_bytes: bytes = int_res.evidence["patch_bytes"]
    append_event(
        submission_id,
        SubmissionState.FETCHED,
        detail={
            "patch_hash": int_res.evidence.get("patch_hash"),
            "size": int_res.evidence.get("size"),
        },
    )
    append_event(
        submission_id,
        SubmissionState.VERIFIED,
        detail={"patch_hash": int_res.evidence.get("patch_hash"), "identity": "ok"},
    )

    # c. Base apply
    if local_repo is not None:
        apply_res = check_base_apply_local(repo_dir=local_repo, patch_bytes=patch_bytes)
    else:
        apply_res = check_base_apply(
            baseline_repo=row["baseline_repo"],
            baseline_commit=campaign.baseline_commit,
            patch_bytes=patch_bytes,
            work_root=work_root,
        )
    if not apply_res.ok:
        _fail(submission_id, apply_res, job_id=job_id)
        return apply_res
    append_event(submission_id, SubmissionState.APPLIED, detail={"ok": True})

    # d. Surface
    surface_res = check_surface(
        patch_bytes=patch_bytes,
        allowed_paths=list(campaign.allowed_paths),
        denied_paths=list(campaign.denied_paths),
    )
    if not surface_res.ok:
        _fail(submission_id, surface_res, job_id=job_id)
        return surface_res
    append_event(
        submission_id,
        SubmissionState.SURFACE_OK,
        detail={"files": surface_res.evidence.get("files")},
    )

    # e. Hermetic build
    if mock_build:
        build_res = build_engine_image_local_mock(
            patch_hash=patch_hash,
            patch_bytes=patch_bytes,
            work_root=work_root,
        )
    else:
        build_res = build_engine_image(
            baseline_repo=row["baseline_repo"],
            baseline_commit=campaign.baseline_commit,
            base_image=config.BASE_IMAGE,
            patch_bytes=patch_bytes,
            patch_hash=patch_hash,
            work_root=work_root,
            push=True,
        )
    if not build_res.ok:
        _fail(submission_id, build_res, job_id=job_id)
        return build_res

    image_ref = str(build_res.evidence.get("image_ref") or "")
    image_tag = build_res.evidence.get("image_tag")
    set_engine_image(submission_id, image_ref)
    append_event(
        submission_id,
        SubmissionState.BUILT,
        detail={
            "image_ref": image_ref,
            "image_tag": image_tag,
            "mock": bool(build_res.evidence.get("mock")),
        },
    )
    set_job_status(
        submission_id,
        "done",
        kind="gates",
        job_id=int(job_id) if job_id is not None else None,
    )
    if campaign.bench is not None:
        enqueued = enqueue_bench_job(submission_id)
        if enqueued:
            logger.info("enqueued bench job for submission %s", submission_id)
    return build_res


def _fail(
    submission_id: str,
    result: GateResult,
    *,
    job_id: Any = None,
) -> None:
    append_event(
        submission_id,
        SubmissionState.REJECTED,
        detail={"reason": result.reason, **result.evidence},
    )
    set_job_status(
        submission_id,
        "failed",
        kind="gates",
        job_id=int(job_id) if job_id is not None else None,
        last_error=result.reason,
    )
