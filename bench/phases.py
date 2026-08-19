"""What a running bench job is doing right now — not a pipeline milestone.

Pod-reported names are untrusted; `coerce_phase` is the only entry point.
"""

from __future__ import annotations

from typing import Any

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 - reuse the backport, do not define a second
    from gate.types import StrEnum


class BenchPhase(StrEnum):
    """Current operation. Multi-SKU runs cycle, so this is not monotonic."""

    PROVISIONING = "provisioning"
    BOOTSTRAPPING = "bootstrapping"
    PULLING_IMAGE = "pulling_image"
    DOWNLOADING_MODEL = "downloading_model"
    STARTING_ENGINE = "starting_engine"
    CORRECTNESS = "correctness"
    SLA_BENCH = "sla_bench"
    TEARDOWN = "teardown"


BENCH_PHASES: tuple[str, ...] = tuple(p.value for p in BenchPhase)

# Worker owns pod lifecycle; a pod claiming provisioning/teardown is dropped.
POD_REPORTABLE_PHASES: frozenset[str] = frozenset(
    {
        BenchPhase.DOWNLOADING_MODEL.value,
        BenchPhase.STARTING_ENGINE.value,
        BenchPhase.CORRECTNESS.value,
        BenchPhase.SLA_BENCH.value,
    }
)

# Display-only, from a pod-writable file: short scalars only.
MAX_PROGRESS_KEYS = 8
MAX_PROGRESS_VALUE_LEN = 64


def coerce_phase(value: Any, *, allowed: frozenset[str] | None = None) -> str | None:
    """Return `value` if it names a phase, else None.

    `allowed` further restricts (pod-reported). Unknown names are dropped, not repaired.
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    if name not in BENCH_PHASES:
        return None
    if allowed is not None and name not in allowed:
        return None
    return name


def coerce_progress(value: Any) -> dict[str, Any] | None:
    """Clamp untrusted progress detail to short scalar entries, else None."""
    if not isinstance(value, dict) or not value:
        return None
    out: dict[str, Any] = {}
    for key, item in value.items():
        if len(out) >= MAX_PROGRESS_KEYS:
            break
        if not isinstance(key, str) or not key.isidentifier():
            continue
        if isinstance(item, bool) or isinstance(item, (int, float)):
            out[key] = item
        elif isinstance(item, str):
            out[key] = item[:MAX_PROGRESS_VALUE_LEN]
    return out or None
