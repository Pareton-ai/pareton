"""Gate result types and submission state names."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SubmissionState:
    COMMITTED = "committed"
    FETCHED = "fetched"
    VERIFIED = "verified"
    APPLIED = "applied"
    SURFACE_OK = "surface_ok"
    BUILDING = "building"
    BUILT = "built"
    CORRECT = "correct"
    SCREENED = "screened"
    BENCHED = "benched"
    REJECTED = "rejected"


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
