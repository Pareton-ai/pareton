"""Seed a Pareton-owned synthetic campaign for Stage 0.

Usage:
    PARETON_DATABASE_URL=... python -m campaign.seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import config
from campaign.cross_env import DEFAULT_CROSS_ENV, validate_cross_env
from campaign.manifest import build_manifest
from campaign.models import CustomerSignoff, SLA, validate_priority_metric
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
# First live campaign: single Hopper SKU (engine is arch 9.0). Flip to open later.
DEFAULT_GPU_SKUS: list[str] = ["H200-SXM-141GB"]
DEFAULT_STATUS = "draft"
KNOWN_SEED_STATUSES = frozenset({"draft", "open", "closed"})
# Partner-facing framing; at pinned gpu_count this is the same lever as throughput.
DEFAULT_PRIORITY_METRIC = "gpu_hours"
DEFAULT_SUCCESS_THRESHOLD = ">=10% GPU-hour reduction at SLA"
# Seed bench floors for the 10% partner bar (do not parse success_threshold text).
# gpu_hours at pinned gpu_count ⇒ throughput ratio 1/0.9; throughput ⇒ 1.10.
_SEED_MIN_SPEEDUP_BY_METRIC: dict[str, float] = {
    "throughput": 1.10,
    "gpu_hours": 1.0 / 0.9,
}

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _seed_min_speedup_each(priority_metric: str) -> float:
    """Return seed floor for a normalized priority_metric (lowercase)."""
    return _SEED_MIN_SPEEDUP_BY_METRIC.get(
        priority_metric, float(DEFAULT_CROSS_ENV["min_speedup_each"])
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _is_placeholder_digest(digest: str) -> bool:
    lowered = digest.lower()
    return lowered in {
        DEFAULT_BASE_IMAGE_DIGEST.lower(),
        DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST.lower(),
    }


def _normalize_trace_url(url: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("workload_trace_url is empty")
    if cleaned.startswith("https://") or cleaned.startswith("file://"):
        return cleaned
    raise ValueError(
        f"workload_trace_url must start with file:// or https:// (got {cleaned!r})"
    )


def _require_sha256(value: str) -> str:
    cleaned = value.strip().lower()
    if not _SHA256_RE.fullmatch(cleaned):
        raise ValueError(
            f"workload_trace_sha256 must be sha256:<64 hex> (got {value!r})"
        )
    return cleaned


def _normalize_gpu_skus(gpu_skus: list[str]) -> list[str]:
    cleaned = [s.strip() for s in gpu_skus if s and s.strip()]
    if not cleaned:
        raise ValueError("gpu_skus must contain at least one non-empty SKU")
    return cleaned


def _normalize_status(status: str) -> str:
    cleaned = status.strip().lower()
    if cleaned not in KNOWN_SEED_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(KNOWN_SEED_STATUSES)} (got {status!r})"
        )
    return cleaned


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
    workload_trace_url: str | None = None,
    workload_trace_sha256: str | None = None,
    allow_placeholders: bool = False,
    priority_metric: str = DEFAULT_PRIORITY_METRIC,
    success_threshold: str = DEFAULT_SUCCESS_THRESHOLD,
    gpu_skus: list[str] | None = None,
    status: str = DEFAULT_STATUS,
    no_bench: bool = False,
) -> str:
    # Normalize before floor lookup / profile insert (build_manifest also validates).
    priority_metric = validate_priority_metric(priority_metric)
    status = _normalize_status(status)
    skus = _normalize_gpu_skus(
        list(DEFAULT_GPU_SKUS) if gpu_skus is None else list(gpu_skus)
    )

    if not allow_placeholders and (
        _is_placeholder_digest(base_image_digest)
        or _is_placeholder_digest(baseline_engine_image_digest)
    ):
        raise ValueError(
            "placeholder digests refused; pass real --base-image-digest and "
            "--baseline-engine-image-digest, or --allow-placeholders"
        )

    if workload_trace_url is None:
        trace_url = f"file://{FIXTURE_TRACE.resolve()}"
    else:
        trace_url = _normalize_trace_url(workload_trace_url)

    if not FIXTURE_TRACE.is_file():
        raise FileNotFoundError(f"missing fixture trace: {FIXTURE_TRACE}")

    trace_sha = _sha256_file(FIXTURE_TRACE)
    if workload_trace_sha256 is not None:
        expected = _require_sha256(workload_trace_sha256)
        if expected != trace_sha.lower():
            raise ValueError(
                "workload_trace_sha256 does not match local fixture "
                f"(expected {trace_sha}, got {workload_trace_sha256})"
            )
    elif trace_url.startswith("https://"):
        raise ValueError(
            "https workload_trace_url requires --workload-trace-sha256 "
            "matching the local fixture"
        )

    if status == "open":
        existing = list_campaigns(status="open")
        if existing and not force:
            cid = existing[0].campaign_id
            print(f"open campaign already exists: {cid}")
            return str(cid)

    profile_id = insert_profile(
        name="pareton-synthetic-v0",
        data={
            "model": "Qwen2.5-72B-Instruct",
            "quantization": "FP8",
            "serving_stack": "vLLM",
            "tensor_parallel": 8,
            "hardware": list(skus),
            "priority_metric": priority_metric,
            "success_threshold": success_threshold,
            "fixture": True,
        },
    )

    campaign_id = uuid4()
    now = datetime.now(timezone.utc)
    opens = now - timedelta(minutes=1)
    closes = now + timedelta(days=90)
    # no_bench: intake/build e2e tests must not auto-enqueue real GPU bench jobs.
    bench = (
        None
        if no_bench
        else build_seed_bench_spec(
            model_repo=bench_model_repo,
            model_revision=bench_model_revision,
            dtype=bench_dtype,
            max_model_len=bench_max_model_len,
            baseline_engine_image_digest=baseline_engine_image_digest,
            gpu_count=bench_gpu_count,
            serve_args=bench_serve_args,
            cross_env={
                **DEFAULT_CROSS_ENV,
                "min_speedup_each": _seed_min_speedup_each(priority_metric),
            },
        )
    )

    fields_manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=baseline_repo,
        baseline_commit=baseline_commit,
        base_image_digest=base_image_digest,
        gpu_skus=skus,
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
        priority_metric=priority_metric,
        success_threshold=success_threshold,
        status=status,
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
        gpu_skus=skus,
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
        priority_metric=priority_metric,
        success_threshold=success_threshold,
        status=status,
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
        help="Insert even if an open campaign already exists (only applies with --status open)",
    )
    p.add_argument(
        "--gpu-skus",
        action="append",
        default=None,
        help=(
            "GPU SKU for the campaign (repeatable). Default: single "
            f"{DEFAULT_GPU_SKUS[0]} (first live campaign should stay single-SKU)"
        ),
    )
    p.add_argument(
        "--status",
        default=DEFAULT_STATUS,
        choices=sorted(KNOWN_SEED_STATUSES),
        help="Campaign status (default: draft; open only after fee/caps + launch ops)",
    )
    p.add_argument(
        "--no-bench",
        action="store_true",
        help="Omit the bench spec so built submissions do not enqueue GPU bench jobs",
    )
    p.add_argument(
        "--workload-trace-url",
        default=None,
        help="file:// or https:// URL stored in the campaign (default: local fixture)",
    )
    p.add_argument(
        "--workload-trace-sha256",
        default=None,
        help="Must match local fixture hash; required when URL is https://",
    )
    p.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow default placeholder base/engine digests (dev only)",
    )
    p.add_argument(
        "--priority-metric",
        default=DEFAULT_PRIORITY_METRIC,
        help="What the campaign optimizes for (throughput, gpu_hours, latency, "
        "utilization, cost_per_request)",
    )
    p.add_argument(
        "--success-threshold",
        default=DEFAULT_SUCCESS_THRESHOLD,
        help="Human-readable win condition for the pilot",
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
            workload_trace_url=args.workload_trace_url,
            workload_trace_sha256=args.workload_trace_sha256,
            allow_placeholders=args.allow_placeholders,
            priority_metric=args.priority_metric,
            success_threshold=args.success_threshold,
            gpu_skus=args.gpu_skus,
            status=args.status,
            no_bench=args.no_bench,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
