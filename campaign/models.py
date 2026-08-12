"""Domain objects for profiles and campaign manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


# Priority metrics a campaign can optimize for (BD memo vocabulary).
PRIORITY_METRICS = frozenset(
    {
        "throughput",
        "gpu_hours",
        "latency",
        "utilization",
        "cost_per_request",
    }
)


def validate_priority_metric(value: str) -> str:
    """Validate campaigns.priority_metric; return the normalized value."""
    cleaned = str(value).strip().lower()
    if cleaned not in PRIORITY_METRICS:
        raise ValueError(
            f"priority_metric must be one of {sorted(PRIORITY_METRICS)}, got {value!r}"
        )
    return cleaned


@dataclass
class SLA:
    p99_ttft_ms: float | None = None
    p99_itl_ms: float | None = None
    quality_floor_spec: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SLA:
        data = data or {}
        return cls(
            p99_ttft_ms=data.get("p99_ttft_ms"),
            p99_itl_ms=data.get("p99_itl_ms"),
            quality_floor_spec=str(data.get("quality_floor_spec") or ""),
        )


@dataclass
class CustomerSignoff:
    approved_manifest_hash: str
    approver: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved_manifest_hash": self.approved_manifest_hash,
            "approver": self.approver,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CustomerSignoff | None:
        if not data:
            return None
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            approved_manifest_hash=str(data["approved_manifest_hash"]),
            approver=str(data["approver"]),
            timestamp=ts,
        )


@dataclass
class Profile:
    id: UUID | None
    name: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id) if self.id else None,
            "name": self.name,
            "data": self.data,
        }


@dataclass
class CampaignManifest:
    """Pinned, content-addressed campaign configuration."""

    campaign_id: UUID | None
    profile_id: UUID | None
    baseline_repo: str
    baseline_commit: str
    base_image_digest: str
    gpu_skus: list[str]
    workload_trace_sha256: str
    workload_trace_url: str
    sla: SLA
    scoring_config_sha256: str | None
    scoring_config_url: str | None
    allowed_paths: list[str]
    denied_paths: list[str]
    window_opens_at: datetime
    window_closes_at: datetime
    manifest_hash: str
    customer_signoff: CustomerSignoff | None
    status: str  # draft | open | closed
    priority_metric: str  # one of PRIORITY_METRICS
    success_threshold: str  # human-readable win condition for the pilot
    bench: dict[str, Any] | None = None
    # Build/launch recipe (campaign.engine). None ⇒ the vLLM default, and stays
    # out of the manifest pin set so pre-engine campaign hashes remain valid.
    engine: dict[str, Any] | None = None
    # Dynamic workload pool + sampler + z-score promotion. None stays out of
    # the manifest pin set (same back-compat rule as bench/engine).
    workload_pool: list[dict[str, Any]] | None = None
    sampling_rule: dict[str, Any] | None = None
    z_threshold: float | None = None

    def to_public_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "campaign_id": str(self.campaign_id) if self.campaign_id else None,
            "profile_id": str(self.profile_id) if self.profile_id else None,
            "baseline_repo": self.baseline_repo,
            "baseline_commit": self.baseline_commit,
            "base_image_digest": self.base_image_digest,
            "gpu_skus": list(self.gpu_skus),
            "workload_trace_sha256": self.workload_trace_sha256,
            "workload_trace_url": self.workload_trace_url,
            "sla": self.sla.to_dict(),
            "scoring_config_sha256": self.scoring_config_sha256,
            "scoring_config_url": self.scoring_config_url,
            "allowed_paths": list(self.allowed_paths),
            "denied_paths": list(self.denied_paths),
            "window": {
                "opens_at": self.window_opens_at.isoformat(),
                "closes_at": self.window_closes_at.isoformat(),
            },
            "manifest_hash": self.manifest_hash,
            "customer_signoff": (
                self.customer_signoff.to_dict() if self.customer_signoff else None
            ),
            "status": self.status,
            "priority_metric": self.priority_metric,
            "success_threshold": self.success_threshold,
            "bench": self.bench,
            "engine": self.engine,
        }
        if self.workload_pool is not None:
            out["workload_pool"] = list(self.workload_pool)
        if self.sampling_rule is not None:
            out["sampling_rule"] = dict(self.sampling_rule)
        if self.z_threshold is not None:
            out["z_threshold"] = self.z_threshold
        return out
