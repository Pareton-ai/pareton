"""Manifest hashing and freeze helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from .models import CampaignManifest, CustomerSignoff, SLA


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
) -> dict[str, Any]:
    """Return the pin set used for manifest_hash (excludes status/signoff)."""
    sla_obj = sla if isinstance(sla, SLA) else SLA.from_dict(sla)
    return {
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
    }


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
    status: str = "draft",
    customer_signoff: CustomerSignoff | None = None,
    manifest_hash: str | None = None,
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
    )
    mh = manifest_hash or compute_manifest_hash(fields)
    sla_obj = sla if isinstance(sla, SLA) else SLA.from_dict(sla)
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
    )
