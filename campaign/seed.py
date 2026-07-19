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
from campaign.cross_env import DEFAULT_CROSS_ENV, validate_cross_env
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

DEFAULT_BENCH_MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BENCH_MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
DEFAULT_BENCH_DTYPE = "bfloat16"
DEFAULT_BENCH_MAX_MODEL_LEN = 8192
# Placeholder until WS-A2b publishes the real baseline engine image.
DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST = "sha256:" + ("c" * 64)
DEFAULT_BENCH_GPU_COUNT = 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def build_seed_bench_spec(
    *,
    model_repo: str = DEFAULT_BENCH_MODEL_REPO,
    model_revision: str = DEFAULT_BENCH_MODEL_REVISION,
    dtype: str = DEFAULT_BENCH_DTYPE,
    max_model_len: int = DEFAULT_BENCH_MAX_MODEL_LEN,
    baseline_engine_image_digest: str = DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST,
    gpu_count: int = DEFAULT_BENCH_GPU_COUNT,
    serve_args: list[str] | None = None,
    cross_env: dict | None = None,
) -> dict:
    ce = DEFAULT_CROSS_ENV if cross_env is None else cross_env
    return {
        "model": {
            "hf_repo": model_repo,
            "hf_revision": model_revision,
            "dtype": dtype,
            "quantization": None,
            "max_model_len": max_model_len,
        },
        "baseline_engine_image_digest": baseline_engine_image_digest,
        "gpu_count": gpu_count,
        "serve_args": list(serve_args) if serve_args else None,
        "correctness": None,
        "perf_screen": None,
        "cross_env": validate_cross_env(ce),
    }


def seed_synthetic_campaign(
    *,
    baseline_repo: str = DEFAULT_BASELINE_REPO,
    baseline_commit: str = DEFAULT_BASELINE_COMMIT,
    base_image_digest: str = DEFAULT_BASE_IMAGE_DIGEST,
    force: bool = False,
    bench_model_repo: str = DEFAULT_BENCH_MODEL_REPO,
    bench_model_revision: str = DEFAULT_BENCH_MODEL_REVISION,
    bench_dtype: str = DEFAULT_BENCH_DTYPE,
    bench_max_model_len: int = DEFAULT_BENCH_MAX_MODEL_LEN,
    baseline_engine_image_digest: str = DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST,
    bench_gpu_count: int = DEFAULT_BENCH_GPU_COUNT,
    bench_serve_args: list[str] | None = None,
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
    bench = build_seed_bench_spec(
        model_repo=bench_model_repo,
        model_revision=bench_model_revision,
        dtype=bench_dtype,
        max_model_len=bench_max_model_len,
        baseline_engine_image_digest=baseline_engine_image_digest,
        gpu_count=bench_gpu_count,
        serve_args=bench_serve_args,
    )

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
        bench=bench,
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
        bench=bench,
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
    p.add_argument("--bench-model-repo", default=DEFAULT_BENCH_MODEL_REPO)
    p.add_argument("--bench-model-revision", default=DEFAULT_BENCH_MODEL_REVISION)
    p.add_argument("--bench-dtype", default=DEFAULT_BENCH_DTYPE)
    p.add_argument(
        "--bench-max-model-len", type=int, default=DEFAULT_BENCH_MAX_MODEL_LEN
    )
    p.add_argument(
        "--baseline-engine-image-digest",
        default=DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST,
    )
    p.add_argument("--bench-gpu-count", type=int, default=DEFAULT_BENCH_GPU_COUNT)
    p.add_argument(
        "--bench-serve-args",
        action="append",
        default=None,
        help="Extra serve arg (repeatable); prepended after --model /model pins",
    )
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
            bench_model_repo=args.bench_model_repo,
            bench_model_revision=args.bench_model_revision,
            bench_dtype=args.bench_dtype,
            bench_max_model_len=args.bench_max_model_len,
            baseline_engine_image_digest=args.baseline_engine_image_digest,
            bench_gpu_count=args.bench_gpu_count,
            bench_serve_args=args.bench_serve_args,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
