"""Pareton Stage 0 API: campaigns, submissions, presigned patch uploads."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

import config
from bench.phases import BenchPhase, coerce_phase, coerce_progress
from bench.score import summarize_prompt_scores
from builder.hermetic import _ANSI_SEQ, _CONTROL_CHARS
from campaign.store import (
    campaign_hotkey_is_disqualified,
    count_submission_campaigns,
    get_campaign,
    get_public_stats,
    get_submission,
    get_submission_for_campaign,
    list_campaigns,
    list_campaign_submissions,
    list_events,
    list_latest_states,
    list_submission_jobs,
)
from db.exceptions import DatabaseNotConfigured, DatabaseUnavailable
from gate.types import SubmissionState
from round.store import (
    get_latest_weight_set,
    get_leader,
    get_round,
    get_round_entry_report,
    list_round_entries,
    list_rounds,
    list_score_progress,
    list_submission_round_entries,
)
from storage.s3 import create_presigned_patch_upload

V1_CACHE_CONTROL = "public, max-age=30, stale-while-revalidate=300"
# Live pipeline endpoints use no-store until the submission reaches a terminal
# state (PAR-44). Lists / campaigns keep the shared short TTL above.
# ``built`` is terminal for no-bench campaigns; bench campaigns continue via
# ``bench_queued`` (and later) in the same worker turn after enqueue.
_TERMINAL_SUBMISSION_STATES = frozenset(
    {"built", "scored", "disqualified", "rejected", "rejected_duplicate"}
)
# rounds.status: a pending or running round is live and moves without warning.
_TERMINAL_ROUND_STATUSES = frozenset({"complete", "void"})
_NO_STORE = "no-store"

# Identity is the hotkey. Lists carry a prefix; detail pages carry it in full.
_HOTKEY_PREFIX_LEN = 16


def _is_terminal_submission_state(state: str | None) -> bool:
    return state in _TERMINAL_SUBMISSION_STATES


def _set_live_submission_cache_control(
    response: Response, latest_state: str | None
) -> None:
    """Detail + build-log stay fresh while the pipeline is still moving."""
    if not _is_terminal_submission_state(latest_state):
        response.headers["Cache-Control"] = _NO_STORE


def _set_live_round_cache_control(response: Response, statuses: list[str]) -> None:
    """Same rule for rounds: one live round makes the whole payload uncacheable."""
    if any(s not in _TERMINAL_ROUND_STATUSES for s in statuses):
        response.headers["Cache-Control"] = _NO_STORE


def _short_hotkey(hotkey: str | None) -> str | None:
    return hotkey[:_HOTKEY_PREFIX_LEN] if hotkey else hotkey


class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        path = request.url.path
        if path == "/health":
            response.headers["Cache-Control"] = _NO_STORE
            return response
        if (
            request.method == "GET"
            and path.startswith("/v1/")
            and response.status_code == 200
            # Handlers may already have set no-store for live submissions.
            and "cache-control" not in response.headers
        ):
            response.headers["Cache-Control"] = V1_CACHE_CONTROL
        return response


app = FastAPI(
    title="Pareton API",
    description="Stage 0 campaign + submission surface (SN10).",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(CacheControlMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(DatabaseNotConfigured)
async def _db_not_configured(_request, exc: DatabaseNotConfigured):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(DatabaseUnavailable)
async def _db_unavailable(_request, exc: DatabaseUnavailable):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": str(exc)})


class PresignRequest(BaseModel):
    campaign_id: str
    hotkey: str = Field(min_length=8, max_length=128)


class PresignResponse(BaseModel):
    upload_url: str
    retrieval_url: str
    object_key: str
    expires_in: int


# `str` fallback is deliberate: submission_events.state is unconstrained TEXT,
# so a legacy or hand-inserted row outside the enum must not break the
# documented contract. The named `SubmissionState` component is still emitted,
# and that is what the frontend union derives from (PAR-46).
SubmissionStateName = SubmissionState | str
BenchPhaseName = BenchPhase | str


class SubmissionRoundModel(BaseModel):
    """A submission's newest round entry: where it ran and how it did.

    `status` is the entry verdict (`round_entries.status`). A submission that
    never reached a round has no model at all, and `rejected` is reported by
    `latest_state`: a rejected submission never got an entry.
    """

    round_id: str
    ordinal: int
    status: str
    score: float | None = None
    disqualify_reason: str | None = None


class SubmissionSummaryModel(BaseModel):
    """One row of `GET /v1/campaigns/{campaign_id}/submissions`."""

    id: str
    campaign_id: str
    patch_hash: str
    hotkey: str
    baseline_commit: str
    retrieval_url: str
    commit_block: int | None = None
    committed_at: str
    engine_image_ref: str | None = None
    latest_state: SubmissionStateName | None = None
    round: SubmissionRoundModel | None = None


class SubmissionsPageModel(BaseModel):
    campaign_id: str
    total: int
    limit: int
    offset: int
    submissions: list[SubmissionSummaryModel]


class SubmissionEventModel(BaseModel):
    state: SubmissionStateName
    evidence_ref: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class SubmissionJobModel(BaseModel):
    status: str
    last_error: str | None = None
    phase: BenchPhaseName | None = None
    phase_started_at: str | None = None
    heartbeat_at: str | None = None  # stale after ~60s means the worker is gone
    progress: dict[str, Any] | None = None


class SubmissionDetailModel(BaseModel):
    """`submission` stays loose: no state field, no payoff."""

    submission: dict[str, Any]
    latest_state: SubmissionStateName | None = None
    jobs: list[SubmissionJobModel]
    events: list[SubmissionEventModel]
    round: SubmissionRoundModel | None = None


class LeaderModel(BaseModel):
    """`GET /v1/campaigns/{campaign_id}/leader`. Detail page: full hotkey."""

    campaign_id: str
    submission_id: str
    patch_hash: str
    hotkey: str
    engine_image_ref: str
    won_at_round_id: str
    won_at_ordinal: int
    last_score: float
    last_scored_round_id: str | None = None
    updated_at: str


class RoundSummaryModel(BaseModel):
    """One row of `GET /v1/campaigns/{campaign_id}/rounds`."""

    id: str
    ordinal: int
    status: str
    void_reason: str | None = None
    void_detail: str | None = None
    gpu_sku: str
    seed_block: int
    seed_block_hash: str
    entry_count: int
    leader_changed: bool | None = None
    created_at: str
    completed_at: str | None = None


class RoundsPageModel(BaseModel):
    campaign_id: str
    total: int
    limit: int
    offset: int
    rounds: list[RoundSummaryModel]


class RoundEntryModel(BaseModel):
    """One image run inside a round. Evidence URLs stay behind their gate.

    `role` and `status` are `round_entries` values; the vocabularies are
    `round.rank.ENTRY_ROLES` and `round.rank.ENTRY_STATUSES`. `score` is null
    for a disqualified or infra-failed entry; 0.0 means baseline speed.
    """

    id: int
    submission_id: str | None = None
    patch_hash: str | None = None
    hotkey: str | None = None
    role: str
    engine_image_ref: str
    status: str
    score: float | None = None
    disqualify_reason: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class RoundDetailModel(BaseModel):
    """`GET /v1/rounds/{round_id}`. `phase` is live while the round runs."""

    id: str
    campaign_id: str
    ordinal: int
    status: str
    void_reason: str | None = None
    void_detail: str | None = None
    gpu_sku: str
    seed_block: int
    seed_block_hash: str
    seed_hex: str
    sampled_trace_sha256: str
    scoring_rule: dict[str, Any]
    incumbent_submission_id: str | None = None
    winner_submission_id: str | None = None
    leader_changed: bool | None = None
    baseline_drift: float | None = None
    phase: BenchPhaseName | None = None
    phase_started_at: str | None = None
    heartbeat_at: str | None = None
    progress: dict[str, Any] | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    entries: list[RoundEntryModel]


class PromptScoreModel(BaseModel):
    """One prompt's contribution to an entry's score.

    `speedup` is the fraction faster than baseline at the same output token
    count: 0.35 is 35 percent faster, and a negative value is slower. A
    non-null `reason` means the prompt was forced to 0.0 and did not measure
    anything; a 0.0 with no reason is a real result meaning baseline speed.
    """

    request_id: str
    speedup: float
    aligned_tokens: int
    baseline_e2e_s: float | None = None
    candidate_e2e_s: float | None = None
    reason: str | None = None


class PromptSummaryModel(BaseModel):
    """Counts over `prompts`, so the headline number needs no client math."""

    total: int
    scored: int
    zeroed: int
    below_tolerance: int
    zeroed_by_reason: dict[str, int]


class RoundEntryReportModel(BaseModel):
    """`GET /v1/rounds/{round_id}/entries/{entry_id}/report`.

    The arithmetic behind one entry's score. `prompts` is empty for an entry
    that never reached scoring, which is every disqualified and infra-failed
    entry: read `status` and `reason` for why.
    """

    round_id: str
    round_ordinal: int
    entry_id: int
    submission_id: str | None = None
    patch_hash: str | None = None
    hotkey: str | None = None
    role: str
    status: str
    engine_image_ref: str
    image_digest: str | None = None
    score: float | None = None
    reason: str | None = None
    engine_crashed: bool = False
    scoring_rule: dict[str, Any]
    prompt_summary: PromptSummaryModel
    prompts: list[PromptScoreModel]
    sla: dict[str, Any] | None = None
    correctness: dict[str, Any] | None = None
    started_at: str | None = None
    completed_at: str | None = None


class ScorePointEntryModel(BaseModel):
    """One scatter dot. List response, so the hotkey is truncated."""

    submission_id: str
    hotkey: str | None = None
    role: str
    status: str
    score: float | None = None


class ScorePointModel(BaseModel):
    """One round ordinal on the chart. A void round leaves a gap, not a zero."""

    round_id: str
    ordinal: int
    status: str
    leader_score: float | None = None
    entries: list[ScorePointEntryModel]


class ScoreProgressModel(BaseModel):
    campaign_id: str
    points: list[ScorePointModel]


class WeightBreakdownModel(BaseModel):
    """One campaign's contribution, so the vector is auditable, not magic.

    `uid` is what the metagraph reported at compute time and is NOT
    authoritative afterwards: a UID is a lease that deregistration reassigns,
    so nobody may cache it. Resolve a hotkey against the live metagraph.

    `note` says why a share was withheld (`vacant`, `closed`, `deregistered`)
    and is null for a share that paid. A withheld share burns.
    """

    campaign_id: str
    hotkey: str | None = None
    uid: int | None = None
    blocks_held: int | None = None
    weight: float
    note: str | None = None


class WeightsModel(BaseModel):
    """`GET /v1/weights`. `weights[i]` is the weight for UID `i`."""

    computed_at_block: int
    version_key: int
    burn_uid: int
    weights: list[float]
    breakdown: list[WeightBreakdownModel]


@app.get("/health")
def health():
    return {"ok": True, "service": "pareton", "stage": 0}


@app.get("/v1/campaigns")
def campaigns(status: str | None = Query(default=None)):
    items = list_campaigns(status=status)
    return {"campaigns": [c.to_public_dict() for c in items]}


@app.get("/v1/campaigns/{campaign_id}")
def campaign_detail(campaign_id: str):
    c = get_campaign(campaign_id)
    if c is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return c.to_public_dict()


@app.get(
    "/v1/campaigns/{campaign_id}/submissions",
    responses={200: {"model": SubmissionsPageModel}},
)
def campaign_submissions(
    campaign_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    page = list_campaign_submissions(campaign_id, limit=limit, offset=offset)
    if page is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return {
        "campaign_id": campaign_id,
        "total": page["total"],
        "limit": limit,
        "offset": offset,
        "submissions": [
            {
                **{
                    k: (str(v) if k in ("id", "campaign_id") else v)
                    for k, v in r.items()
                    if k not in ("latest_state", "round")
                },
                "latest_state": r.get("latest_state"),
                "round": r.get("round"),
            }
            for r in page["items"]
        ],
    }


@app.get("/v1/stats")
def stats():
    return get_public_stats()


def _require_campaign(campaign_id: str) -> None:
    if get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")


@app.get(
    "/v1/campaigns/{campaign_id}/leader",
    responses={200: {"model": LeaderModel}},
)
def campaign_leader(campaign_id: UUID):
    """The crown holder. A vacant crown has no row, so it is a 404."""
    _require_campaign(str(campaign_id))
    row = get_leader(str(campaign_id))
    if row is None:
        raise HTTPException(status_code=404, detail="leader is vacant")
    return {"campaign_id": str(campaign_id), **row}


@app.get(
    "/v1/campaigns/{campaign_id}/rounds",
    responses={200: {"model": RoundsPageModel}},
)
def campaign_rounds(
    campaign_id: UUID,
    response: Response,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    page = list_rounds(str(campaign_id), limit=limit, offset=offset)
    if page is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    rows = page["items"]
    _set_live_round_cache_control(response, [r["status"] for r in rows])
    return {
        "campaign_id": str(campaign_id),
        "total": page["total"],
        "limit": limit,
        "offset": offset,
        "rounds": rows,
    }


@app.get(
    "/v1/campaigns/{campaign_id}/score-progress",
    responses={200: {"model": ScoreProgressModel}},
)
def campaign_score_progress(campaign_id: UUID, response: Response):
    """Chart series, oldest ordinal first. Void rounds keep their ordinal."""
    _require_campaign(str(campaign_id))
    points = list_score_progress(str(campaign_id))
    _set_live_round_cache_control(response, [p["status"] for p in points])
    for point in points:
        for entry in point["entries"]:
            entry["hotkey"] = _short_hotkey(entry["hotkey"])
    return {"campaign_id": str(campaign_id), "points": points}


@app.get("/v1/rounds/{round_id}", responses={200: {"model": RoundDetailModel}})
def round_detail(round_id: UUID, response: Response):
    row = get_round(round_id)
    if row is None:
        raise HTTPException(status_code=404, detail="round not found")
    _set_live_round_cache_control(response, [row["status"]])
    return {
        **row,
        # Pod-written columns: names outside the vocabulary are dropped, and
        # progress is clamped to short scalars.
        "phase": coerce_phase(row.get("phase")),
        "progress": coerce_progress(row.get("progress")),
        "entries": list_round_entries(round_id),
    }


@app.get(
    "/v1/rounds/{round_id}/entries/{entry_id}/report",
    responses={200: {"model": RoundEntryReportModel}},
)
def round_entry_report(round_id: UUID, entry_id: int, response: Response):
    """The arithmetic behind one entry's score, prompt by prompt.

    The round score is the named rule applied to `prompts`, so a miner can
    re-derive it and see which prompts paid and which were gated. Absolute
    seconds are served alongside the ratios: a speedup on its own cannot be
    checked against a local run.

    The baseline entry stores its SLA replay rather than a comparison, so it
    comes back with `sla` populated and `prompts` empty. It is the reference
    the rest are measured against, not a competitor with a score of its own.
    """
    row = get_round_entry_report(round_id, entry_id)
    if row is None:
        raise HTTPException(status_code=404, detail="round entry not found")
    _set_live_round_cache_control(response, [row["round_status"]])

    raw = row.get("report") or {}
    if not isinstance(raw, dict):
        raw = {}
    score_report = raw.get("score_report")
    score_report = score_report if isinstance(score_report, dict) else {}
    prompts = score_report.get("prompts")
    prompts = prompts if isinstance(prompts, list) else []
    sla = raw.get("sla")
    if sla is None and "metrics" in raw:
        # The baseline row stores the SLA replay itself, not an entry report.
        sla = raw

    return {
        "round_id": str(row["round_id"]),
        "round_ordinal": row["round_ordinal"],
        "entry_id": row["id"],
        "submission_id": (
            None if row["submission_id"] is None else str(row["submission_id"])
        ),
        "patch_hash": row["patch_hash"],
        "hotkey": row["hotkey"],
        "role": row["role"],
        "status": row["status"],
        "engine_image_ref": row["engine_image_ref"],
        "image_digest": raw.get("image_digest"),
        "score": row["score"],
        # disqualify_reason is the column the worker writes the harness reason
        # into for every non-scored status, infra_failed included.
        "reason": row["disqualify_reason"] or raw.get("reason"),
        "engine_crashed": bool(raw.get("engine_crashed", False)),
        "scoring_rule": row["scoring_rule"] or {},
        "prompt_summary": summarize_prompt_scores(prompts),
        "prompts": prompts,
        "sla": sla,
        "correctness": raw.get("correctness"),
        "started_at": _iso_or_none(row.get("started_at")),
        "completed_at": _iso_or_none(row.get("completed_at")),
    }


@app.get("/v1/weights", responses={200: {"model": WeightsModel}})
def latest_weights(response: Response):
    """The newest stored weight vector. Reads only; never computes.

    `weights[i]` is UID `i`. Pass it to `SetWeights` with
    `uids=range(len(weights))`. A missing or all-zero row is 404: an empty
    vector is a valid on-chain instruction to pay nobody.
    """
    row = get_latest_weight_set()
    values = [float(w) for w in row["weights"]] if row is not None else []
    if not any(w != 0 for w in values):
        # HTTPException builds a new response; headers on `response` would drop.
        raise HTTPException(
            status_code=404,
            detail="no weight set computed yet",
            headers={"Cache-Control": _NO_STORE},
        )
    response.headers["Cache-Control"] = _NO_STORE
    return {
        "computed_at_block": row["computed_at_block"],
        "version_key": row["version_key"],
        "burn_uid": row["burn_uid"],
        "weights": values,
        "breakdown": row["breakdown"],
    }


def _iso_or_none(value: Any) -> str | None:
    """Timestamp as ISO-8601, or None. Nullable columns arrive as None."""
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _submission_detail_payload(row: dict) -> dict:
    events = list_events(row["id"])
    states = list_latest_states([row["id"]])
    jobs = list_submission_jobs(row["id"])
    return {
        "submission": {
            **{k: (str(v) if k in ("id", "campaign_id") else v) for k, v in row.items()}
        },
        "latest_state": states.get(str(row["id"])),
        "jobs": [
            {
                "status": j["status"],
                "last_error": j.get("last_error"),
                "phase": coerce_phase(j.get("phase")),
                "phase_started_at": _iso_or_none(j.get("phase_started_at")),
                "heartbeat_at": _iso_or_none(j.get("heartbeat_at")),
                "progress": coerce_progress(j.get("progress")),
            }
            for j in jobs
        ],
        "events": [
            {
                "state": e["state"],
                "evidence_ref": e.get("evidence_ref"),
                "detail": e.get("detail") or {},
                "created_at": e["created_at"].isoformat()
                if hasattr(e["created_at"], "isoformat")
                else str(e["created_at"]),
            }
            for e in events
        ],
        "round": list_submission_round_entries([row["id"]]).get(str(row["id"])),
    }


def _resolve_unambiguous_submission(patch_hash: str) -> dict:
    row = get_submission(patch_hash)
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    if count_submission_campaigns(patch_hash) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                "patch_hash exists in multiple campaigns; "
                "use /v1/campaigns/{campaign_id}/submissions/{patch_hash}"
            ),
        )
    return row


@app.get(
    "/v1/campaigns/{campaign_id}/submissions/{patch_hash}",
    responses={200: {"model": SubmissionDetailModel}},
)
def campaign_submission_detail(campaign_id: str, patch_hash: str, response: Response):
    row = get_submission_for_campaign(campaign_id, patch_hash)
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    payload = _submission_detail_payload(row)
    _set_live_submission_cache_control(response, payload.get("latest_state"))
    return payload


@app.get(
    "/v1/submissions/{patch_hash}",
    responses={200: {"model": SubmissionDetailModel}},
)
def submission_detail(patch_hash: str, response: Response):
    payload = _submission_detail_payload(_resolve_unambiguous_submission(patch_hash))
    _set_live_submission_cache_control(response, payload.get("latest_state"))
    return payload


_BUILD_LOG_MAX_TAIL = 2000


def _build_log_response(row: dict, tail: int) -> PlainTextResponse:
    log_path = config.BUILD_LOG_DIR / str(row["id"]) / "build.log"
    if not log_path.is_file():
        raise HTTPException(status_code=404, detail="build log not found")
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            f.seek(max(0, size - 256 * 1024))
            raw_tail = f.read()
    except OSError:
        raise HTTPException(status_code=404, detail="build log not found") from None
    text = raw_tail.decode("utf-8", errors="replace")
    clean = _CONTROL_CHARS.sub("", _ANSI_SEQ.sub("", text))
    resp = PlainTextResponse("\n".join(clean.splitlines()[-tail:]) + "\n")
    states = list_latest_states([row["id"]])
    _set_live_submission_cache_control(resp, states.get(str(row["id"])))
    return resp


@app.get(
    "/v1/campaigns/{campaign_id}/submissions/{patch_hash}/build-log",
    response_class=PlainTextResponse,
)
def campaign_submission_build_log(
    campaign_id: str,
    patch_hash: str,
    tail: int = Query(default=200, ge=1, le=_BUILD_LOG_MAX_TAIL),
):
    row = get_submission_for_campaign(campaign_id, patch_hash)
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    return _build_log_response(row, tail)


@app.get("/v1/submissions/{patch_hash}/build-log", response_class=PlainTextResponse)
def submission_build_log(
    patch_hash: str,
    tail: int = Query(default=200, ge=1, le=_BUILD_LOG_MAX_TAIL),
):
    """Last `tail` lines of the durable build log (PAR-37 path), sanitized.

    Content is miner-influenced build output: ANSI/control chars stripped,
    served as text/plain. Non-terminal submissions are Cache-Control: no-store
    so live tails are not held by CDN/browser caches (PAR-44).
    """
    row = _resolve_unambiguous_submission(patch_hash)
    return _build_log_response(row, tail)


@app.post(
    "/v1/uploads/patch",
    response_model=PresignResponse,
    responses={403: {"description": "Hotkey is disqualified from this campaign"}},
)
def presign_patch(body: PresignRequest):
    c = get_campaign(body.campaign_id)
    if c is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    if c.status != "open":
        raise HTTPException(status_code=400, detail="campaign is not open")
    if campaign_hotkey_is_disqualified(body.campaign_id, body.hotkey):
        raise HTTPException(
            status_code=403,
            detail="hotkey is disqualified from campaign",
        )
    try:
        result = create_presigned_patch_upload(
            campaign_id=body.campaign_id,
            hotkey=body.hotkey,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"presign failed: {exc}") from exc
    return {
        "upload_url": result.upload_url,
        "retrieval_url": result.retrieval_url,
        "object_key": result.object_key,
        "expires_in": result.expires_in,
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
