"""Seed a Pareton-owned synthetic campaign for Stage 0.

Usage:
    PARETON_DATABASE_URL=... python -m campaign.seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import config
from campaign.manifest import build_manifest
from campaign.models import CustomerSignoff, SLA
from campaign.store import insert_campaign, insert_profile, list_campaigns

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TRACE = (
    REPO_ROOT / "fixtures" / "campaigns" / "synthetic_v0" / "workload_trace.json"
)

# Manual pins for the first synthetic campaign (ops can override via flags).
DEFAULT_BASELINE_REPO = "https://github.com/vllm-project/vllm.git"
# vLLM v0.24.0 (latest stable at time of pinning)
DEFAULT_BASELINE_COMMIT = "ee0da84ab9e04ac7610e28580af62c365e898389"
# Placeholder until the Pareton baseline image is built and published;
# replace with the real digest before opening a campaign with real builds.
DEFAULT_BASE_IMAGE_DIGEST = "sha256:" + ("b" * 64)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def seed_synthetic_campaign(
    *,
    baseline_repo: str = DEFAULT_BASELINE_REPO,
    baseline_commit: str = DEFAULT_BASELINE_COMMIT,
    base_image_digest: str = DEFAULT_BASE_IMAGE_DIGEST,
    force: bool = False,
) -> str:
    existing = list_campaigns(status="open")
    if existing and not force:
        cid = existing[0].campaign_id
        print(f"open campaign already exists: {cid}")
        return str(cid)

    if not FIXTURE_TRACE.is_file():
        raise FileNotFoundError(f"missing fixture trace: {FIXTURE_TRACE}")

    trace_sha = _sha256_file(FIXTURE_TRACE)
    # Content-addressed URL placeholder until uploaded to S3 in ops.
    trace_url = f"file://{FIXTURE_TRACE.resolve()}"

    profile_id = insert_profile(
        name="pareton-synthetic-v0",
        data={
            "model": "Qwen2.5-72B-Instruct",
            "quantization": "FP8",
            "serving_stack": "vLLM",
            "tensor_parallel": 8,
            "hardware": ["H200-SXM-141GB"],
            "priority_metric": "throughput",
            "success_threshold": ">=10% GPU-hour reduction at SLA",
            "fixture": True,
        },
    )

    campaign_id = uuid4()
    now = datetime.now(timezone.utc)
    opens = now - timedelta(minutes=1)
    closes = now + timedelta(days=90)

    fields_manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=baseline_repo,
        baseline_commit=baseline_commit,
        base_image_digest=base_image_digest,
        gpu_skus=["H200-SXM-141GB", "B200"],
        workload_trace_sha256=trace_sha,
        workload_trace_url=trace_url,
        sla=SLA(
            p99_ttft_ms=2000.0,
            p99_itl_ms=50.0,
            quality_floor_spec="greedy token-match >= 0.99 vs baseline",
        ),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=list(config.DEFAULT_ALLOWED_PATHS),
        denied_paths=list(config.DEFAULT_DENIED_PATHS),
        window_opens_at=opens,
        window_closes_at=closes,
        status="open",
        customer_signoff=None,
    )

    signoff = CustomerSignoff(
        approved_manifest_hash=fields_manifest.manifest_hash,
        approver="pareton-admin",
        timestamp=now,
    )
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=baseline_repo,
        baseline_commit=baseline_commit,
        base_image_digest=base_image_digest,
        gpu_skus=["H200-SXM-141GB", "B200"],
        workload_trace_sha256=trace_sha,
        workload_trace_url=trace_url,
        sla=SLA(
            p99_ttft_ms=2000.0,
            p99_itl_ms=50.0,
            quality_floor_spec="greedy token-match >= 0.99 vs baseline",
        ),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=list(config.DEFAULT_ALLOWED_PATHS),
        denied_paths=list(config.DEFAULT_DENIED_PATHS),
        window_opens_at=opens,
        window_closes_at=closes,
        status="open",
        customer_signoff=signoff,
        manifest_hash=fields_manifest.manifest_hash,
    )

    inserted = insert_campaign(manifest)
    print(json.dumps(manifest.to_public_dict(), indent=2, default=str))
    print(f"seeded campaign_id={inserted}")
    return str(inserted)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Seed synthetic Pareton Stage 0 campaign")
    p.add_argument("--baseline-repo", default=DEFAULT_BASELINE_REPO)
    p.add_argument("--baseline-commit", default=DEFAULT_BASELINE_COMMIT)
    p.add_argument("--base-image-digest", default=DEFAULT_BASE_IMAGE_DIGEST)
    p.add_argument(
        "--force",
        action="store_true",
        help="Insert even if an open campaign already exists",
    )
    args = p.parse_args(argv)
    try:
        seed_synthetic_campaign(
            baseline_repo=args.baseline_repo,
            baseline_commit=args.baseline_commit,
            base_image_digest=args.base_image_digest,
            force=args.force,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
