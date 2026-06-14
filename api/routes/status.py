"""GET /api/status -- detailed validator overview."""

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.helpers.state_reader import sanitize_floats
from cacheon_db.readers import count_evaluations, get_leader_state, get_validator_meta

router = APIRouter()


@router.get(
    "/api/status",
    tags=["Overview"],
    summary="Validator status overview",
    description=(
        "Current leader, evaluation counts, time since last eval, "
        "and chain scan / weight-set block numbers."
    ),
)
def status():
    meta = get_validator_meta()
    leader, _runner_up = get_leader_state()
    total, n_active, n_dq, last_eval_ts = count_evaluations()

    last_eval_age_min = (
        round((time.time() - last_eval_ts) / 60, 1) if last_eval_ts else None
    )

    return JSONResponse(
        content=sanitize_floats(
            {
                "leader_uid": leader.get("uid") if leader else None,
                "leader_score": leader.get("score") if leader else None,
                "leader_image": leader.get("image") if leader else None,
                "n_evaluated": total,
                "n_active": n_active,
                "n_disqualified": n_dq,
                "last_eval_ts": last_eval_ts,
                "last_eval_age_min": last_eval_age_min,
                "last_scan_block": meta.get("last_scan_block"),
                "last_weights_set_block": meta.get("last_weights_set_block"),
            }
        ),
        headers={"Cache-Control": "public, max-age=30"},
    )
