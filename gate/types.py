"""Gate result types and the submission state vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SubmissionState(StrEnum):
    """The pipeline state vocabulary. This is the only definition.

    `campaign.store.KNOWN_SUBMISSION_STATES`, the OpenAPI schema, and the
    frontend `SubmissionState` union all derive from these members. Adding a
    state here is the whole change on the backend. Member order is the pipeline
    order and is used for `/v1/stats` bucket ordering, so append new states in
    the position the worker reaches them.
    """

    COMMITTED = "committed"
    PICKED_UP = "picked_up"
    FETCHED = "fetched"
    VERIFIED = "verified"
    APPLIED = "applied"
    SURFACE_OK = "surface_ok"
    BUILDING = "building"
    IMAGE_PUSHED = "image_pushed"
    BUILT = "built"
    BENCH_QUEUED = "bench_queued"
    SAMPLED = "sampled"
    CORRECT = "correct"
    SCREENED = "screened"
    BENCHED = "benched"
    REJECTED = "rejected"


SUBMISSION_STATES: tuple[str, ...] = tuple(s.value for s in SubmissionState)


@dataclass(frozen=True)
class GateResult:
    ok: bool
    state: str
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, state: str, **evidence: Any) -> GateResult:
        return cls(ok=True, state=state, evidence=evidence)

    @classmethod
    def reject(cls, reason: str, **evidence: Any) -> GateResult:
        return cls(
            ok=False,
            state=SubmissionState.REJECTED,
            reason=reason,
            evidence=evidence,
        )
