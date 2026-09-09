"""Structured lifecycle event emitter for Axiom observability.

Every public function emits one single-line JSON object through the
``pareton.lifecycle`` logger at INFO level.  journald captures it;
Vector ships it to Axiom.

Rules (from the integration brief):
- No secrets, wallet material, or patch contents.
- Job-scoped events carry non-empty ``submission_id``.
- stdlib ``logging`` + ``json`` only (no new dependencies).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

_logger = logging.getLogger("pareton.lifecycle")

# Fields that must never appear in event payloads.
_FORBIDDEN_KEYS = frozenset(
    {
        "token",
        "secret",
        "password",
        "mnemonic",
        "private_key",
        "patch_bytes",
        "patch_content",
    }
)


def _emit(event: str, **kwargs: Any) -> dict[str, Any]:
    """Build and log a lifecycle event.  Returns the dict for testing."""
    payload: dict[str, Any] = {"event": event}
    for k, v in kwargs.items():
        if k in _FORBIDDEN_KEYS:
            continue
        if v is None or v == "":
            continue
        payload[k] = v
    _logger.info(json.dumps(payload, default=str, separators=(",", ":")))
    return payload


# -- Worker heartbeat --------------------------------------------------------


def heartbeat(*, queue_depth: int | None = None) -> dict[str, Any]:
    return _emit("heartbeat", queue_depth=queue_depth)


# -- Chain / ingestion -------------------------------------------------------


def chain_scanned(
    *,
    block: int,
    commitments_seen: int,
    ingested: int,
) -> dict[str, Any]:
    """One completed chain read.

    Emitted on every successful scan, including scans that ingest nothing:
    a quiet chain and a stalled scanner are indistinguishable from
    ``submission_ingested`` alone, so absence of this event is the signal.
    """
    return _emit(
        "chain_scanned",
        block=block,
        commitments_seen=commitments_seen,
        ingested=ingested,
    )


def submission_ingested(
    *,
    submission_id: str,
    campaign_id: str,
    patch_sha256: str,
    hotkey: str,
) -> dict[str, Any]:
    return _emit(
        "submission_ingested",
        submission_id=submission_id,
        campaign_id=campaign_id,
        patch_sha256=patch_sha256,
        hotkey=hotkey[:16] if hotkey else "",
    )


# -- Gates -------------------------------------------------------------------


def gate_passed(
    *,
    submission_id: str,
    gate: str,
    patch_sha256: str = "",
) -> dict[str, Any]:
    return _emit(
        "gate_passed",
        submission_id=submission_id,
        gate=gate,
        patch_sha256=patch_sha256,
    )


def gate_failed(
    *,
    submission_id: str,
    gate: str,
    error: str = "",
    patch_sha256: str = "",
) -> dict[str, Any]:
    return _emit(
        "gate_failed",
        submission_id=submission_id,
        gate=gate,
        error=error,
        patch_sha256=patch_sha256,
    )


# -- Hermetic build ----------------------------------------------------------


def build_started(
    *,
    submission_id: str,
    patch_sha256: str,
) -> dict[str, Any]:
    return _emit(
        "build_started",
        submission_id=submission_id,
        patch_sha256=patch_sha256,
    )


def build_succeeded(
    *,
    submission_id: str,
    patch_sha256: str,
    image_digest: str = "",
    duration_s: float | None = None,
) -> dict[str, Any]:
    return _emit(
        "build_succeeded",
        submission_id=submission_id,
        patch_sha256=patch_sha256,
        image_digest=image_digest,
        duration_s=round(duration_s, 1) if duration_s is not None else None,
    )


def build_failed(
    *,
    submission_id: str,
    patch_sha256: str,
    error: str = "",
    duration_s: float | None = None,
) -> dict[str, Any]:
    return _emit(
        "build_failed",
        submission_id=submission_id,
        patch_sha256=patch_sha256,
        error=error,
        duration_s=round(duration_s, 1) if duration_s is not None else None,
    )


# -- GPU pods ----------------------------------------------------------------


def pod_provisioned(
    *,
    pod: str,
    provider: str,
    submission_id: str = "",
    job_id: str = "",
) -> dict[str, Any]:
    return _emit(
        "pod_provisioned",
        pod=pod,
        provider=provider,
        submission_id=submission_id,
        job_id=job_id,
    )


def pod_provision_failed(
    *,
    provider: str,
    error: str = "",
    submission_id: str = "",
) -> dict[str, Any]:
    return _emit(
        "pod_provision_failed",
        provider=provider,
        error=error,
        submission_id=submission_id,
    )


def pod_destroyed(
    *,
    pod: str,
    provider: str,
) -> dict[str, Any]:
    return _emit("pod_destroyed", pod=pod, provider=provider)


def destroy_failed(
    *,
    pod: str,
    provider: str,
    error: str = "",
    volume_uid: str = "",
) -> dict[str, Any]:
    return _emit(
        "destroy_failed",
        pod=pod,
        provider=provider,
        error=error,
        volume_uid=volume_uid,
    )


def pod_ttl_exceeded(
    *,
    pod: str,
    provider: str,
) -> dict[str, Any]:
    return _emit("pod_ttl_exceeded", pod=pod, provider=provider)


# -- Bench stages ------------------------------------------------------------


def bench_started(
    *,
    submission_id: str,
    job_id: str = "",
    stage: str = "bench",
    pod: str = "",
    provider: str = "",
    image_digest: str = "",
    patch_sha256: str = "",
) -> dict[str, Any]:
    return _emit(
        "bench_started",
        submission_id=submission_id,
        job_id=job_id,
        stage=stage,
        pod=pod,
        provider=provider,
        image_digest=image_digest,
        patch_sha256=patch_sha256,
    )


def bench_completed(
    *,
    submission_id: str,
    job_id: str = "",
    stage: str = "bench",
    pod: str = "",
    provider: str = "",
    image_digest: str = "",
    patch_sha256: str = "",
    duration_s: float | None = None,
    verdict: str = "",
    evidence_s3_url: str = "",
    evidence_sha256: str = "",
) -> dict[str, Any]:
    return _emit(
        "bench_completed",
        submission_id=submission_id,
        job_id=job_id,
        stage=stage,
        pod=pod,
        provider=provider,
        image_digest=image_digest,
        patch_sha256=patch_sha256,
        duration_s=round(duration_s, 1) if duration_s is not None else None,
        verdict=verdict,
        evidence_s3_url=evidence_s3_url,
        evidence_sha256=evidence_sha256,
    )


def bench_failed(
    *,
    submission_id: str,
    job_id: str = "",
    stage: str = "bench",
    pod: str = "",
    provider: str = "",
    image_digest: str = "",
    patch_sha256: str = "",
    duration_s: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    return _emit(
        "bench_failed",
        submission_id=submission_id,
        job_id=job_id,
        stage=stage,
        pod=pod,
        provider=provider,
        image_digest=image_digest,
        patch_sha256=patch_sha256,
        duration_s=round(duration_s, 1) if duration_s is not None else None,
        error=error,
    )


# -- Weights -----------------------------------------------------------------


def weights_computed(
    *,
    computed_at_block: int,
    version_key: int,
    uid_count: int,
    burn_share: float,
    set_ok: bool | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """A chain-cadence compute completed. Absence is the stall alert."""
    return _emit(
        "weights_computed",
        computed_at_block=computed_at_block,
        version_key=version_key,
        uid_count=uid_count,
        burn_share=round(burn_share, 4),
        set_ok=set_ok,
        enabled=enabled,
    )


def round_voided(
    *,
    round_id: str,
    campaign_id: str,
    void_reason: str,
) -> dict[str, Any]:
    """One round abandoned without a verdict.

    A stale round and an absent worker heartbeat are different failures: the
    worker process can be healthy while the round it claimed is dead. Alert on
    this event, not on the heartbeat monitor.
    """
    return _emit(
        "round_voided",
        round_id=round_id,
        campaign_id=campaign_id,
        void_reason=void_reason,
    )


# -- Provider balance --------------------------------------------------------


def provider_balance_low(
    *,
    provider: str,
    balance: float | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    return _emit(
        "provider_balance_low",
        provider=provider,
        balance=balance,
        threshold=threshold,
    )


# -- Generic terminal failure ------------------------------------------------


def job_failed(
    *,
    submission_id: str,
    job_id: str = "",
    stage: str = "",
    pod: str = "",
    provider: str = "",
    image_digest: str = "",
    patch_sha256: str = "",
    duration_s: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    return _emit(
        "job_failed",
        submission_id=submission_id,
        job_id=job_id,
        stage=stage,
        pod=pod,
        provider=provider,
        image_digest=image_digest,
        patch_sha256=patch_sha256,
        duration_s=round(duration_s, 1) if duration_s is not None else None,
        error=error,
    )


class Timer:
    """Context manager that tracks elapsed seconds for duration_s fields."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_s: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_s = time.monotonic() - self._start
