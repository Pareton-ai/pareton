"""GET /api/eval-progress -- live eval round progress."""

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.helpers.state_reader import sanitize_floats
from cacheon_db.readers import get_eval_progress

router = APIRouter()

_STALE_THRESHOLD_S = 1800  # 30 minutes


@router.get(
    "/api/eval-progress",
    tags=["Overview"],
    summary="Live eval progress",
    description=(
        "Returns the current eval round progress, including phase, "
        "per-challenger status, GPU info, and a timestamped step timeline. "
        'Returns {"status": "idle"} when no eval is running. '
        "Finished rounds linger as complete for 15 minutes."
    ),
)
def eval_progress():
    data = get_eval_progress()
    if data is None:
        return JSONResponse(
            content={"status": "idle"},
            headers={"Cache-Control": "public, max-age=5"},
        )
    updated = data.get("updated_at", 0)
    if data.get("status") == "running" and time.time() - updated > _STALE_THRESHOLD_S:
        data["possibly_stale"] = True
    return JSONResponse(
        content=sanitize_floats(data),
        headers={"Cache-Control": "public, max-age=5"},
    )
