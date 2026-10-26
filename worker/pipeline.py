"""Run gates a–e (plus the optional d1/d2 reviews) for one claimed submission job."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import config
from builder.cleanup import evict_candidate_image
from builder.hermetic import build_engine_image, build_engine_image_local_mock
from builder.registry import baseline_build_image_ref, baseline_engine_image_ref
from campaign.store import (
    append_event,
    complete_gates_job,
    defer_job,
    get_campaign,
    set_engine_image,
    set_job_status,
)
from gate.base_apply import check_base_apply, check_base_apply_local
from gate.identity import check_identity
from gate.integrity import check_integrity
from gate.llm_review import check_llm_review
from gate.obfuscation import check_obfuscation
from gate.surface import check_surface
from gate.types import GateResult, SubmissionState
from observability import events as obs
from observability.events import Timer

logger = logging.getLogger(__name__)


def _build_base_image(campaign: Any) -> str:
    """Resolve the image a miner build starts FROM.

    Prefer the campaign's pinned baseline engine image: only it carries
    /src/.deps (cmake FetchContent stamps), which --network=none builds need.
    A bare base image fails at cmake configure (cutlass clone has no network).
    """
    bench = getattr(campaign, "bench", None) or {}
    engine_digest = bench.get("baseline_engine_image_digest")
    if engine_digest:
        return baseline_engine_image_ref(engine_digest)
    logger.warning(
        "campaign %s: no baseline engine pin; falling back to bare base image",
        getattr(campaign, "campaign_id", "?"),
    )
    return baseline_build_image_ref(campaign.base_image_digest)


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

    append_event(
        submission_id,
        SubmissionState.PICKED_UP,
        detail={"job_id": int(job_id) if job_id is not None else None},
    )

    # a. Identity (fail-fast; success has no dedicated state in the machine)
    id_res = check_identity(
        hotkey=hotkey,
        registered_hotkeys=registered_hotkeys,
        campaign=campaign,
        baseline_commit=baseline_commit,
    )
    if not id_res.ok:
        obs.gate_failed(
            submission_id=submission_id,
            gate="identity",
            error=id_res.reason or "",
            patch_sha256=patch_hash,
        )
        _fail(submission_id, id_res, job_id=job_id)
        return id_res
    obs.gate_passed(
        submission_id=submission_id, gate="identity", patch_sha256=patch_hash
    )

    # b. Integrity → fetched → verified
    integrity_kwargs: dict[str, Any] = {
        "retrieval_url": retrieval_url,
        "expected_patch_hash": patch_hash,
        "hotkey": hotkey,
        "campaign_id": campaign_id,
    }
    if fetcher is not None:
        integrity_kwargs["fetcher"] = fetcher
    int_res = check_integrity(**integrity_kwargs)
    if not int_res.ok:
        obs.gate_failed(
            submission_id=submission_id,
            gate="integrity",
            error=int_res.reason or "",
            patch_sha256=patch_hash,
        )
        _fail(submission_id, int_res, job_id=job_id)
        return int_res
    obs.gate_passed(
        submission_id=submission_id, gate="integrity", patch_sha256=patch_hash
    )
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
        obs.gate_failed(
            submission_id=submission_id,
            gate="base_apply",
            error=apply_res.reason or "",
            patch_sha256=patch_hash,
        )
        _fail(submission_id, apply_res, job_id=job_id)
        return apply_res
    obs.gate_passed(
        submission_id=submission_id, gate="base_apply", patch_sha256=patch_hash
    )
    append_event(submission_id, SubmissionState.APPLIED, detail={"ok": True})

    # d. Surface
    patch_limits = (campaign.bench or {}).get("patch_limits") or {}
    surface_res = check_surface(
        patch_bytes=patch_bytes,
        allowed_paths=list(campaign.allowed_paths),
        denied_paths=list(campaign.denied_paths),
        max_patch_bytes=patch_limits.get("max_patch_bytes"),
        max_added_line_chars=patch_limits.get("max_added_line_chars"),
        reject_renames=bool(patch_limits.get("reject_renames")),
        reject_mode_changes=bool(patch_limits.get("reject_mode_changes")),
    )
    if not surface_res.ok:
        obs.gate_failed(
            submission_id=submission_id,
            gate="surface",
            error=surface_res.reason or "",
            patch_sha256=patch_hash,
        )
        _fail(submission_id, surface_res, job_id=job_id)
        return surface_res
    obs.gate_passed(
        submission_id=submission_id, gate="surface", patch_sha256=patch_hash
    )
    append_event(
        submission_id,
        SubmissionState.SURFACE_OK,
        detail={"files": surface_res.evidence.get("files")},
    )

    # d1. Obfuscation scan (optional, deterministic)
    if patch_limits.get("reject_obfuscation"):
        obf_res = check_obfuscation(patch_bytes=patch_bytes)
        if not obf_res.ok:
            obs.gate_failed(
                submission_id=submission_id,
                gate="obfuscation",
                error=obf_res.reason or "",
                patch_sha256=patch_hash,
            )
            _fail(submission_id, obf_res, job_id=job_id)
            return obf_res
        obs.gate_passed(
            submission_id=submission_id, gate="obfuscation", patch_sha256=patch_hash
        )

    # d2. LLM review (optional, reject-only; never rewrites the patch). A
    # provider outage defers the job; it never skips the review.
    llm_review = (campaign.bench or {}).get("llm_review")
    if llm_review:
        review_res = check_llm_review(
            patch_bytes=patch_bytes,
            patch_hash=patch_hash,
            spec=llm_review,
        )
        if not review_res.ok and review_res.evidence.get("retryable"):
            deferred = _defer_llm_review(submission_id, row, review_res)
            if deferred is not None:
                return deferred
        if not review_res.ok:
            obs.gate_failed(
                submission_id=submission_id,
                gate="llm_review",
                error=review_res.reason or "",
                patch_sha256=patch_hash,
            )
            _fail(submission_id, review_res, job_id=job_id)
            return review_res
        obs.gate_passed(
            submission_id=submission_id, gate="llm_review", patch_sha256=patch_hash
        )

    # e. Hermetic build
    append_event(submission_id, SubmissionState.BUILDING, detail={})
    obs.build_started(submission_id=submission_id, patch_sha256=patch_hash)
    build_timer = Timer()
    with build_timer:
        if mock_build:
            build_res = build_engine_image_local_mock(
                patch_hash=patch_hash,
                patch_bytes=patch_bytes,
                work_root=work_root,
            )
        else:
            try:
                base_image = _build_base_image(campaign)
            except ValueError as exc:
                result = GateResult.reject(
                    "base_image_digest_invalid",
                    error=str(exc),
                    base_image_digest=campaign.base_image_digest,
                )
                obs.build_failed(
                    submission_id=submission_id,
                    patch_sha256=patch_hash,
                    error="base_image_digest_invalid",
                    duration_s=build_timer.elapsed_s,
                )
                _fail(submission_id, result, job_id=job_id)
                return result
            build_res = build_engine_image(
                baseline_repo=row["baseline_repo"],
                baseline_commit=campaign.baseline_commit,
                base_image=base_image,
                patch_bytes=patch_bytes,
                patch_hash=patch_hash,
                work_root=work_root,
                log_dir=config.BUILD_LOG_DIR / submission_id,
                push=True,
                engine=campaign.engine,
            )
    if not build_res.ok:
        obs.build_failed(
            submission_id=submission_id,
            patch_sha256=patch_hash,
            error=build_res.reason or "",
            duration_s=build_timer.elapsed_s,
        )
        _fail(submission_id, build_res, job_id=job_id)
        return build_res
    obs.build_succeeded(
        submission_id=submission_id,
        patch_sha256=patch_hash,
        image_digest=str(build_res.evidence.get("image_ref", "")),
        duration_s=build_timer.elapsed_s,
    )

    image_ref = str(build_res.evidence.get("image_ref") or "")
    image_tag = build_res.evidence.get("image_tag")
    set_engine_image(submission_id, image_ref)
    if build_res.evidence.get("pushed"):
        append_event(
            submission_id,
            SubmissionState.IMAGE_PUSHED,
            detail={
                "image_ref": image_ref,
                "image_digest": build_res.evidence.get("image_digest"),
            },
        )
    append_event(
        submission_id,
        SubmissionState.BUILT,
        detail={
            "image_ref": image_ref,
            "image_tag": image_tag,
            "mock": bool(build_res.evidence.get("mock")),
            "build_log": build_res.evidence.get("build_log"),
        },
    )
    # bench_queued is the round creator's input queue, so it is appended in the
    # same transaction that settles the job: the watcher selects each cohort
    # from submissions in this state.
    enqueue_round = campaign.bench is not None
    complete_gates_job(
        submission_id,
        job_id=int(job_id) if job_id is not None else None,
        enqueue_round=enqueue_round,
    )
    if build_res.evidence.get("pushed") and image_tag:
        try:
            evict_candidate_image(str(image_tag))
        except (RuntimeError, ValueError) as exc:
            # GHCR and the digest-pinned DB ref are already durable. A local
            # cleanup failure must not reject a successfully built submission.
            logger.warning(
                "candidate image cleanup failed for submission %s: %s",
                submission_id,
                exc,
            )
    if enqueue_round:
        logger.info("queued submission %s for the next round", submission_id)
    return build_res


def _defer_llm_review(
    submission_id: str, row: dict[str, Any], result: GateResult
) -> GateResult | None:
    """Requeue the job after a provider outage. None means attempts ran out.

    The submission gets no event: it stays at surface_ok until the retry.
    """
    job_id = row.get("job_id")
    attempt = int(row.get("job_attempt") or 1)
    if job_id is None or attempt >= config.LLM_REVIEW_MAX_ATTEMPTS:
        return None
    delay_s = min(
        config.LLM_REVIEW_RETRY_BASE_S * 2 ** (attempt - 1),
        config.LLM_REVIEW_RETRY_MAX_S,
    )
    if not defer_job(
        job_id=int(job_id),
        attempt=attempt,
        delay_s=delay_s,
        last_error=f"{result.reason}: {result.evidence.get('detail', '')}",
    ):
        # Superseded attempt; whoever holds the job now settles it.
        logger.warning("submission %s: deferral did not land", submission_id)
    logger.warning(
        "submission %s: llm review unavailable (%s); retry %d/%d in %.0fs",
        submission_id,
        result.evidence.get("detail"),
        attempt + 1,
        config.LLM_REVIEW_MAX_ATTEMPTS,
        delay_s,
    )
    return GateResult(
        ok=False,
        state=SubmissionState.SURFACE_OK,
        reason="llm_review_deferred",
        evidence={**result.evidence, "retry_in_s": delay_s, "attempt": attempt},
    )


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
        job_id=int(job_id) if job_id is not None else None,
        last_error=result.reason,
    )
