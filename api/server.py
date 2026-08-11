"""Pareton Stage 0 API: campaigns, submissions, presigned patch uploads."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

import config
from builder.hermetic import _ANSI_SEQ, _CONTROL_CHARS
from campaign.store import (
    count_submission_campaigns,
    derive_bench_verdict_from_events,
    get_campaign,
    get_public_stats,
    get_submission,
    get_submission_for_campaign,
    list_bench_reports,
    list_bench_summaries,
    list_campaigns,
    list_events,
    list_latest_states,
    list_submission_jobs,
    list_submissions,
)
from db.exceptions import DatabaseNotConfigured, DatabaseUnavailable
from storage.s3 import create_presigned_patch_upload

V1_CACHE_CONTROL = "public, max-age=30, stale-while-revalidate=300"
# Live pipeline endpoints use no-store until the submission reaches a terminal
# state (PAR-44). Lists / campaigns keep the shared short TTL above.
_TERMINAL_SUBMISSION_STATES = frozenset({"benched", "rejected"})
_NO_STORE = "no-store"


def _is_terminal_submission_state(state: str | None) -> bool:
    return state in _TERMINAL_SUBMISSION_STATES


def _set_live_submission_cache_control(
    response: Response, latest_state: str | None
) -> None:
    """Detail + build-log stay fresh while the pipeline is still moving."""
    if not _is_terminal_submission_state(latest_state):
        response.headers["Cache-Control"] = _NO_STORE


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


@app.get("/v1/campaigns/{campaign_id}/submissions")
def campaign_submissions(
    campaign_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    c = get_campaign(campaign_id)
    if c is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    page = list_submissions(campaign_id, limit=limit, offset=offset)
    rows = page["items"]
    ids = [r["id"] for r in rows]
    summaries = list_bench_summaries(campaign_id, submission_ids=ids)
    states = list_latest_states(ids)
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
                },
                "latest_state": states.get(str(r["id"])),
                "bench_verdict": summaries.get(str(r["id"])),
            }
            for r in rows
        ],
    }


@app.get("/v1/stats")
def stats():
    return get_public_stats()


def _submission_detail_payload(row: dict) -> dict:
    events = list_events(row["id"])
    reports = list_bench_reports(row["id"])
    states = list_latest_states([row["id"]])
    jobs = list_submission_jobs(row["id"])
    return {
        "submission": {
            **{k: (str(v) if k in ("id", "campaign_id") else v) for k, v in row.items()}
        },
        "latest_state": states.get(str(row["id"])),
        "jobs": [
            {
                "kind": j["kind"],
                "status": j["status"],
                "last_error": j.get("last_error"),
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
        "bench_reports": [
            {
                "task_id": r["task_id"],
                "stage": r["stage"],
                "verdict": r["verdict"],
                "report": r.get("report") or {},
                "evidence_s3_url": r.get("evidence_s3_url"),
                "gpu_sku": r.get("gpu_sku"),
                "mock": bool(r.get("mock")),
                "created_at": r["created_at"].isoformat()
                if hasattr(r["created_at"], "isoformat")
                else str(r["created_at"]),
            }
            for r in reports
        ],
        "bench_verdict": derive_bench_verdict_from_events(events),
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


@app.get("/v1/campaigns/{campaign_id}/submissions/{patch_hash}")
def campaign_submission_detail(campaign_id: str, patch_hash: str, response: Response):
    row = get_submission_for_campaign(campaign_id, patch_hash)
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    payload = _submission_detail_payload(row)
    _set_live_submission_cache_control(response, payload.get("latest_state"))
    return payload


@app.get("/v1/submissions/{patch_hash}")
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


@app.post("/v1/uploads/patch")
def presign_patch(body: PresignRequest):
    c = get_campaign(body.campaign_id)
    if c is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    if c.status != "open":
        raise HTTPException(status_code=400, detail="campaign is not open")
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
