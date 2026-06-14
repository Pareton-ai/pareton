"""GET /api/rounds -- eval rounds grouped by evaluation_block.
GET /api/eval-job -- current pending eval job."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.helpers.state_reader import sanitize_floats
from cacheon_db.readers import get_pending_eval_job, list_rounds

router = APIRouter()


@router.get(
    "/api/rounds",
    tags=["Rounds"],
    summary="Evaluation rounds",
    description=(
        "Evaluations grouped by round (evaluation_block). Each round shows "
        "the block, timestamp, and the list of challengers with their outcomes."
    ),
)
def rounds():
    rounds_list = list_rounds()
    return JSONResponse(
        content=sanitize_floats({"rounds": rounds_list, "total": len(rounds_list)}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/eval-job",
    tags=["Rounds"],
    summary="Current pending eval job",
    description="The eval job queued for the next GPU eval round. Shows queued challengers.",
)
def eval_job():
    job = get_pending_eval_job()
    if job is None:
        return JSONResponse(
            content={"eval_job": None, "message": "No pending eval job"},
            headers={"Cache-Control": "public, max-age=30"},
        )
    return JSONResponse(
        content=sanitize_floats({"eval_job": job}),
        headers={"Cache-Control": "public, max-age=30"},
    )
