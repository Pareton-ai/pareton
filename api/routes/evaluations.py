"""GET /api/evaluations -- all eval records.
GET /api/evaluations/hotkey/{hotkey} -- eval history for a specific hotkey (stable identity).
GET /api/evaluations/{uid} -- eval history for a specific UID (slot, not stable across recycling)."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from api.helpers.state_reader import sanitize_floats
from cacheon_db.readers import (
    get_evaluations_by_hotkey,
    get_evaluations_by_uid,
    list_evaluations,
)

router = APIRouter()


@router.get(
    "/api/evaluations",
    tags=["Evaluations"],
    summary="All evaluation records",
    description=(
        "Every completed evaluation, newest first. "
        "Filter with ?status=dq for disqualified only, ?status=active for passing only."
    ),
)
def list_all_evaluations(
    status: str | None = Query(
        None,
        description="Filter: 'dq' for disqualified, 'active' for non-disqualified",
    ),
):
    evals = list_evaluations(status=status)
    return JSONResponse(
        content=sanitize_floats({"evaluations": evals, "total": len(evals)}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/evaluations/hotkey/{hotkey}",
    tags=["Evaluations"],
    summary="Evaluations for a specific hotkey",
    description=(
        "All evaluation records for a given hotkey, newest first. "
        "Hotkey is the stable miner identity; UIDs can be recycled when a slot changes owner."
    ),
)
def evaluations_by_hotkey(hotkey: str):
    evals = get_evaluations_by_hotkey(hotkey)
    if not evals:
        return JSONResponse(
            status_code=404,
            content={"detail": f"No evaluations found for hotkey {hotkey}"},
        )
    return JSONResponse(
        content=sanitize_floats(
            {"hotkey": hotkey, "evaluations": evals, "total": len(evals)}
        ),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/evaluations/{uid}",
    tags=["Evaluations"],
    summary="Evaluations for a specific UID (slot)",
    description=(
        "All evaluation records for a given UID, newest first. "
        "Note: UIDs are Bittensor slot numbers (0-255) and can be recycled when a slot changes owner. "
        "Prefer GET /api/evaluations/hotkey/{hotkey} for stable lookups."
    ),
)
def evaluations_by_uid(uid: int):
    evals = get_evaluations_by_uid(uid)
    if not evals:
        return JSONResponse(
            status_code=404,
            content={"detail": f"No evaluations found for UID {uid}"},
        )
    return JSONResponse(
        content=sanitize_floats(
            {"uid": uid, "evaluations": evals, "total": len(evals)}
        ),
        headers={"Cache-Control": "public, max-age=30"},
    )
