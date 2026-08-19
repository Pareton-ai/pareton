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


# Named ranking rules. One implementation ships today; bench/score.py dispatches
# on the name. TODO(PAR-76): move dispatch and metric computation into
# bench/score.py once the scoring seam lands.
SCORING_RULE_NAMES = frozenset({"median_e2e_speedup"})

DEFAULT_SCORING_RULE: dict[str, Any] = {"name": "median_e2e_speedup"}


def validate_scoring_rule(rule: dict[str, Any] | None) -> dict[str, Any]:
    """Validate campaigns.scoring_rule; return a normalized copy.

    None means the default rule. The rule is pinned in manifest_hash, so two
    campaigns with the same hash rank identically.
    """
    if rule is None:
        return dict(DEFAULT_SCORING_RULE)
    if not isinstance(rule, dict):
        raise ValueError("scoring_rule must be an object")
    name = str(rule.get("name") or "").strip()
    if name not in SCORING_RULE_NAMES:
        raise ValueError(
            f"scoring_rule.name must be one of {sorted(SCORING_RULE_NAMES)}, "
            f"got {rule.get('name')!r}"
        )
    out = {k: rule[k] for k in sorted(rule) if k != "name"}
    return {"name": name, **out}


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
    manifest_hash: str
    customer_signoff: CustomerSignoff | None
    status: str  # draft | open | closed
    priority_metric: str  # one of PRIORITY_METRICS
    success_threshold: str  # human-readable win condition for the pilot
    bench: dict[str, Any] | None = None
    # Build/launch recipe (campaign.engine). None ⇒ the vLLM default, and stays
    # out of the manifest pin set so pre-engine campaign hashes remain valid.
    engine: dict[str, Any] | None = None
    # Dynamic sampling. None stays out of the manifest pin set (same
    # back-compat rule as bench/engine). workload_pool is unused by the
    # sampler; sampling_rule.type hf_rows generates traces.
    workload_pool: list[dict[str, Any]] | None = None
    sampling_rule: dict[str, Any] | None = None
    # Named ranking rule. Always pinned in manifest_hash and NOT NULL in the DB.
    scoring_rule: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_SCORING_RULE)
    )
    # Row creation time, read from the DB rather than pinned. Campaigns run
    # open ended, so this is the only date they carry; None for a manifest
    # built in memory and not yet inserted.
    created_at: datetime | None = None

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
            "created_at": (self.created_at.isoformat() if self.created_at else None),
            "manifest_hash": self.manifest_hash,
            "customer_signoff": (
                self.customer_signoff.to_dict() if self.customer_signoff else None
            ),
            "status": self.status,
            "priority_metric": self.priority_metric,
            "success_threshold": self.success_threshold,
            "bench": self.bench,
            "engine": self.engine,
            "scoring_rule": dict(self.scoring_rule),
        }
        if self.workload_pool is not None:
            out["workload_pool"] = list(self.workload_pool)
        if self.sampling_rule is not None:
            out["sampling_rule"] = dict(self.sampling_rule)
        return out
