"""GET /api/leader -- current leader record.
GET /api/leader/history -- overtake timeline."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.helpers.state_reader import sanitize_floats
from cacheon_db.readers import get_leader_history, get_leader_state

router = APIRouter()


@router.get(
    "/api/leader",
    tags=["Leader"],
    summary="Current leader",
    description="Full record of the reigning champion: UID, score, image, per-prompt stats.",
)
def get_leader():
    leader, runner_up = get_leader_state()
    if leader is None:
        return JSONResponse(
            content={"leader": None, "runner_up": None, "message": "No leader yet"},
            headers={"Cache-Control": "public, max-age=30"},
        )
    return JSONResponse(
        content=sanitize_floats({"leader": leader, "runner_up": runner_up}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/leader/history",
    tags=["Leader"],
    summary="Overtake history",
    description="Chronological list of leader changes. Each entry shows the new leader, the previous leader, and the margin.",
)
def leader_history():
    history = get_leader_history()
    return JSONResponse(
        content=sanitize_floats({"history": history, "total": len(history)}),
        headers={"Cache-Control": "public, max-age=30"},
    )
