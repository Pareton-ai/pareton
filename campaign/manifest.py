"""Manifest hashing and freeze helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from .engine import validate_engine
from .models import (
    CampaignManifest,
    CustomerSignoff,
    SLA,
    validate_priority_metric,
)


def _canon(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _canon(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canon(v) for v in value]
    return value


def freeze_manifest_fields(
    *,
    campaign_id: UUID | None,
    profile_id: UUID | None,
    baseline_repo: str,
    baseline_commit: str,
    base_image_digest: str,
    gpu_skus: list[str],
    workload_trace_sha256: str,
    workload_trace_url: str,
    sla: SLA | dict[str, Any],
    scoring_config_sha256: str | None,
    scoring_config_url: str | None,
    allowed_paths: list[str],
    denied_paths: list[str],
    window_opens_at: datetime,
    window_closes_at: datetime,
    priority_metric: str,
    success_threshold: str,
    bench: dict[str, Any] | None = None,
    engine: dict[str, Any] | None = None,
    workload_pool: list[dict[str, Any]] | None = None,
    sampling_rule: dict[str, Any] | None = None,
    z_threshold: float | None = None,
) -> dict[str, Any]:
    """Return the pin set used for manifest_hash (excludes status/signoff).

    Include ``bench`` only when not None so pre-WS-D campaign hashes stay valid.
    ``engine`` follows the same rule: absent means the vLLM default, and every
    campaign hashed before engine profiles existed keeps its hash.
    ``workload_pool``, ``sampling_rule``, and ``z_threshold`` use the same
    absent-means-unpinned rule.
    """
    sla_obj = sla if isinstance(sla, SLA) else SLA.from_dict(sla)
    out: dict[str, Any] = {
        "campaign_id": str(campaign_id) if campaign_id else None,
        "profile_id": str(profile_id) if profile_id else None,
        "baseline_repo": baseline_repo,
        "baseline_commit": baseline_commit.lower(),
        "base_image_digest": base_image_digest.lower(),
        "gpu_skus": list(gpu_skus),
        "workload_trace_sha256": workload_trace_sha256.lower(),
        "workload_trace_url": workload_trace_url,
        "sla": sla_obj.to_dict(),
        "scoring_config_sha256": (
            scoring_config_sha256.lower() if scoring_config_sha256 else None
        ),
        "scoring_config_url": scoring_config_url,
        "allowed_paths": list(allowed_paths),
        "denied_paths": list(denied_paths),
        "window": {
            "opens_at": window_opens_at.isoformat(),
            "closes_at": window_closes_at.isoformat(),
        },
        "priority_metric": validate_priority_metric(priority_metric),
        "success_threshold": success_threshold,
    }
    if bench is not None:
        out["bench"] = bench
    if engine is not None:
        out["engine"] = validate_engine(engine)
    if workload_pool is not None:
        out["workload_pool"] = _canon(list(workload_pool))
    if sampling_rule is not None:
        out["sampling_rule"] = _canon(dict(sampling_rule))
    if z_threshold is not None:
        out["z_threshold"] = float(z_threshold)
    return out


def compute_manifest_hash(fields: dict[str, Any]) -> str:
    """SHA-256 over canonical JSON of the pin set. Returns sha256:<hex>."""
    payload = json.dumps(_canon(fields), separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_manifest(
    *,
    campaign_id: UUID | None,
    profile_id: UUID | None,
    baseline_repo: str,
    baseline_commit: str,
    base_image_digest: str,
    gpu_skus: list[str],
    workload_trace_sha256: str,
    workload_trace_url: str,
    sla: SLA | dict[str, Any],
    scoring_config_sha256: str | None,
    scoring_config_url: str | None,
    allowed_paths: list[str],
    denied_paths: list[str],
    window_opens_at: datetime,
    window_closes_at: datetime,
    priority_metric: str,
    success_threshold: str,
    status: str = "draft",
    customer_signoff: CustomerSignoff | None = None,
    manifest_hash: str | None = None,
    bench: dict[str, Any] | None = None,
    engine: dict[str, Any] | None = None,
    workload_pool: list[dict[str, Any]] | None = None,
    sampling_rule: dict[str, Any] | None = None,
    z_threshold: float | None = None,
) -> CampaignManifest:
    fields = freeze_manifest_fields(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=baseline_repo,
        baseline_commit=baseline_commit,
        base_image_digest=base_image_digest,
        gpu_skus=gpu_skus,
        workload_trace_sha256=workload_trace_sha256,
        workload_trace_url=workload_trace_url,
        sla=sla,
        scoring_config_sha256=scoring_config_sha256,
        scoring_config_url=scoring_config_url,
        allowed_paths=allowed_paths,
        denied_paths=denied_paths,
        window_opens_at=window_opens_at,
        window_closes_at=window_closes_at,
        priority_metric=priority_metric,
        success_threshold=success_threshold,
        bench=bench,
        engine=engine,
        workload_pool=workload_pool,
        sampling_rule=sampling_rule,
        z_threshold=z_threshold,
    )
    mh = manifest_hash or compute_manifest_hash(fields)
    sla_obj = sla if isinstance(sla, SLA) else SLA.from_dict(sla)
    # Carry the normalized form onto the object so callers and the DB see the
    # same bytes that were hashed.
    engine_obj = fields.get("engine")
    pool_obj = fields.get("workload_pool")
    rule_obj = fields.get("sampling_rule")
    z_obj = fields.get("z_threshold")
    return CampaignManifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=baseline_repo,
        baseline_commit=baseline_commit.lower(),
        base_image_digest=base_image_digest.lower(),
        gpu_skus=list(gpu_skus),
        workload_trace_sha256=workload_trace_sha256.lower(),
        workload_trace_url=workload_trace_url,
        sla=sla_obj,
        scoring_config_sha256=(
            scoring_config_sha256.lower() if scoring_config_sha256 else None
        ),
        scoring_config_url=scoring_config_url,
        allowed_paths=list(allowed_paths),
        denied_paths=list(denied_paths),
        window_opens_at=window_opens_at,
        window_closes_at=window_closes_at,
        manifest_hash=mh,
        customer_signoff=customer_signoff,
        status=status,
        priority_metric=validate_priority_metric(priority_metric),
        success_threshold=success_threshold,
        bench=bench,
        engine=engine_obj,
        workload_pool=list(pool_obj) if isinstance(pool_obj, list) else None,
        sampling_rule=dict(rule_obj) if isinstance(rule_obj, dict) else None,
        z_threshold=float(z_obj) if z_obj is not None else None,
    )
