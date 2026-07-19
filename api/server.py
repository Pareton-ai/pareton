"""Pareton Stage 0 API: campaigns, submissions, presigned patch uploads."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import config
from campaign.store import (
    derive_bench_verdict_from_events,
    get_campaign,
    get_submission,
    list_bench_reports,
    list_bench_summaries,
    list_campaigns,
    list_events,
    list_submissions,
)
from db.exceptions import DatabaseNotConfigured, DatabaseUnavailable
from storage.s3 import create_presigned_patch_upload

app = FastAPI(
    title="Pareton API",
    description="Stage 0 campaign + submission surface (SN10).",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

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
def campaign_submissions(campaign_id: str):
    c = get_campaign(campaign_id)
    if c is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    rows = list_submissions(campaign_id)
    summaries = list_bench_summaries(campaign_id)
    return {
        "campaign_id": campaign_id,
        "submissions": [
            {
                **{
                    k: (str(v) if k in ("id", "campaign_id") else v)
                    for k, v in r.items()
                },
                "bench_verdict": summaries.get(str(r["id"])),
            }
            for r in rows
        ],
    }


@app.get("/v1/submissions/{patch_hash}")
def submission_detail(patch_hash: str):
    row = get_submission(patch_hash)
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    events = list_events(row["id"])
    reports = list_bench_reports(row["id"])
    return {
        "submission": {
            **{k: (str(v) if k in ("id", "campaign_id") else v) for k, v in row.items()}
        },
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
