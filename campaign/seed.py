"""Seed a Pareton-owned synthetic campaign for Stage 0.

Usage:
    PARETON_DATABASE_URL=... python -m campaign.seed
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import config
from bench.sampler import parse_sampling_rule
from campaign.engine import ENGINE_PRESETS
from campaign.engine import preset as engine_preset
from campaign.fees import validate_submission_fee
from campaign.manifest import build_manifest
from campaign.models import (
    SLA,
    CustomerSignoff,
    validate_emission_rule,
    validate_priority_metric,
    validate_scoring_rule,
)
from campaign.store import insert_campaign, insert_profile, list_campaigns

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SAMPLING_RULE = (
    REPO_ROOT / "fixtures" / "campaigns" / "synthetic_v1" / "sampling_rule.json"
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


def _is_placeholder_digest(digest: str) -> bool:
    lowered = digest.lower()
    return lowered in {
        DEFAULT_BASE_IMAGE_DIGEST.lower(),
        DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST.lower(),
    }


def _load_sampling_rule(rule: dict | None) -> dict:
    """Normalize an hf_rows pin. Missing input loads the synthetic fixture."""
    if rule is None:
        if not FIXTURE_SAMPLING_RULE.is_file():
            raise ValueError(
                "sampling_rule is required; pass --sampling-rule-json or add "
                f"{FIXTURE_SAMPLING_RULE}"
            )
        raw = json.loads(FIXTURE_SAMPLING_RULE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("sampling_rule fixture must be a JSON object")
        rule = raw
    return parse_sampling_rule(rule)


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


CORRECTNESS_THRESHOLD_KEYS = (
    "min_mean_logprob",
    "min_token_logprob",
    "min_token_quantile",
    "min_coverage_ratio",
    "max_mean_logprob_drop",
)


def _correctness_thresholds(thresholds: dict | None) -> dict:
    """Complete correctness bars, defaulting from config."""
    src = dict(thresholds or {})
    defaults = {
        "min_mean_logprob": config.BENCH_CORRECTNESS_MIN_MEAN_LOGPROB,
        "min_token_logprob": config.BENCH_CORRECTNESS_MIN_TOKEN_LOGPROB,
        "min_token_quantile": config.BENCH_CORRECTNESS_MIN_TOKEN_QUANTILE,
        "min_coverage_ratio": config.BENCH_CORRECTNESS_MIN_COVERAGE_RATIO,
        "max_mean_logprob_drop": config.BENCH_CORRECTNESS_MAX_MEAN_LOGPROB_DROP,
    }
    out = {k: float(src.get(k, defaults[k])) for k in CORRECTNESS_THRESHOLD_KEYS}
    if not 0.0 < out["min_coverage_ratio"] <= 1.0:
        raise ValueError(
            "bench.correctness.thresholds.min_coverage_ratio must be in (0, 1]"
        )
    if not 0.0 <= out["min_token_quantile"] < 1.0:
        raise ValueError(
            "bench.correctness.thresholds.min_token_quantile must be in [0, 1)"
        )
    if (
        not math.isfinite(out["max_mean_logprob_drop"])
        or out["max_mean_logprob_drop"] <= 0.0
    ):
        raise ValueError(
            "bench.correctness.thresholds.max_mean_logprob_drop must be > 0"
        )
    return out


def require_correctness_thresholds(bench: dict | None) -> None:
    """A campaign may not open without its correctness bars in the manifest.

    The bars are fixed seed-time policy, so nothing has to be measured before
    a campaign opens. They still have to be written down: a bar that is not in
    the manifest can move under a live campaign without the manifest hash
    changing.
    """
    corr = (bench or {}).get("correctness") if isinstance(bench, dict) else None
    thr = corr.get("thresholds") if isinstance(corr, dict) else None
    missing = (
        list(CORRECTNESS_THRESHOLD_KEYS)
        if not isinstance(thr, dict)
        else [k for k in CORRECTNESS_THRESHOLD_KEYS if k not in thr]
    )
    if missing:
        raise ValueError(
            "status=open requires campaigns.bench.correctness.thresholds with "
            f"{sorted(CORRECTNESS_THRESHOLD_KEYS)}; missing {sorted(missing)}"
        )


DEFAULT_EMISSION_RULE_NAME = "linear_decay"


def _emission_rule(rule: dict | None) -> dict:
    """Complete the campaign's pay schedule, defaulting from config."""
    merged = {
        "name": DEFAULT_EMISSION_RULE_NAME,
        "start_weight": config.EMISSION_START_WEIGHT,
        "floor_weight": config.EMISSION_FLOOR_WEIGHT,
        "decay_blocks": config.EMISSION_DECAY_BLOCKS,
        **dict(rule or {}),
    }
    return validate_emission_rule(merged)


def require_emission_rule(emission_rule: dict | None) -> None:
    """A campaign may not open without its pay schedule in the manifest.

    No rule means the campaign pays nothing at all: honest for a draft, and a
    silent no-emission bug for an open one. Like the correctness bars, the rule
    is pinned in manifest_hash, so it has to be written down before miners
    start competing under it.
    """
    if not emission_rule:
        raise ValueError(
            "status=open requires campaigns.emission_rule; a campaign without "
            "one is excluded from the weight vector and pays nothing"
        )


def build_seed_bench_spec(
    *,
    model_repo: str = DEFAULT_BENCH_MODEL_REPO,
    model_revision: str = DEFAULT_BENCH_MODEL_REVISION,
    dtype: str = DEFAULT_BENCH_DTYPE,
    max_model_len: int = DEFAULT_BENCH_MAX_MODEL_LEN,
    quantization: str | None = None,
    baseline_engine_image_digest: str = DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST,
    gpu_count: int = DEFAULT_BENCH_GPU_COUNT,
    serve_args: list[str] | None = None,
    correctness_num_prompts: int | None = None,
    correctness_thresholds: dict | None = None,
) -> dict:
    # Every campaign pins its own correctness thresholds. The values default
    # from config, but they are copied into the manifest here rather than read
    # from the environment at bench time, so editing an env var on a pod
    # cannot move a live campaign's correctness bar.
    correctness: dict = {"thresholds": _correctness_thresholds(correctness_thresholds)}
    if correctness_num_prompts is not None:
        correctness["num_prompts"] = int(correctness_num_prompts)
    return {
        "model": {
            "hf_repo": model_repo,
            "hf_revision": model_revision,
            "dtype": dtype,
            "quantization": quantization,
            "max_model_len": max_model_len,
        },
        "baseline_engine_image_digest": baseline_engine_image_digest,
        "gpu_count": gpu_count,
        "serve_args": list(serve_args) if serve_args else None,
        "correctness": correctness,
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
    bench_quantization: str | None = None,
    baseline_engine_image_digest: str = DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST,
    bench_gpu_count: int = DEFAULT_BENCH_GPU_COUNT,
    bench_serve_args: list[str] | None = None,
    bench_correctness_num_prompts: int | None = None,
    bench_correctness_thresholds: dict | None = None,
    workload_pool: list[dict] | None = None,
    sampling_rule: dict | None = None,
    scoring_rule: dict | None = None,
    emission_rule: dict | None = None,
    allow_placeholders: bool = False,
    priority_metric: str = DEFAULT_PRIORITY_METRIC,
    success_threshold: str = DEFAULT_SUCCESS_THRESHOLD,
    gpu_skus: list[str] | None = None,
    status: str = DEFAULT_STATUS,
    no_bench: bool = False,
    engine: str | None = None,
) -> str:
    # Normalize before floor lookup / profile insert (build_manifest also validates).
    priority_metric = validate_priority_metric(priority_metric)
    status = _normalize_status(status)
    # None (not "vllm") is the default: it keeps engine out of the manifest pin
    # set, so re-seeding an existing campaign reproduces its original hash.
    engine_profile = None if engine is None else engine_preset(engine)
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

    rule = _load_sampling_rule(sampling_rule)

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
    # no_bench: intake/build e2e tests must not auto-enqueue real GPU bench jobs.
    bench = (
        None
        if no_bench
        else build_seed_bench_spec(
            model_repo=bench_model_repo,
            model_revision=bench_model_revision,
            dtype=bench_dtype,
            max_model_len=bench_max_model_len,
            quantization=bench_quantization,
            baseline_engine_image_digest=baseline_engine_image_digest,
            gpu_count=bench_gpu_count,
            serve_args=bench_serve_args,
            correctness_num_prompts=bench_correctness_num_prompts,
            correctness_thresholds=bench_correctness_thresholds,
        )
    )

    emission = _emission_rule(emission_rule)
    fee = validate_submission_fee(
        {
            "amount_tao": config.SUBMISSION_FEE_TAO,
            "recipient": config.PAYMENT_RECIPIENT_ADDRESS,
        }
    )

    if status == "open":
        require_correctness_thresholds(bench)
        require_emission_rule(emission)

    pool = list(workload_pool) if workload_pool is not None else None
    scoring = validate_scoring_rule(scoring_rule)

    fields_manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo=baseline_repo,
        baseline_commit=baseline_commit,
        base_image_digest=base_image_digest,
        gpu_skus=skus,
        workload_trace_sha256=None,
        workload_trace_url=None,
        sla=SLA(
            p99_ttft_ms=2000.0,
            p99_itl_ms=50.0,
            quality_floor_spec="greedy token-match >= 0.99 vs baseline",
        ),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=list(config.DEFAULT_ALLOWED_PATHS),
        denied_paths=list(config.DEFAULT_DENIED_PATHS),
        priority_metric=priority_metric,
        success_threshold=success_threshold,
        status=status,
        customer_signoff=None,
        bench=bench,
        engine=engine_profile,
        workload_pool=pool,
        sampling_rule=rule,
        scoring_rule=scoring,
        emission_rule=emission,
        submission_fee=fee,
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
        workload_trace_sha256=None,
        workload_trace_url=None,
        sla=SLA(
            p99_ttft_ms=2000.0,
            p99_itl_ms=50.0,
            quality_floor_spec="greedy token-match >= 0.99 vs baseline",
        ),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=list(config.DEFAULT_ALLOWED_PATHS),
        denied_paths=list(config.DEFAULT_DENIED_PATHS),
        priority_metric=priority_metric,
        success_threshold=success_threshold,
        status=status,
        customer_signoff=signoff,
        manifest_hash=fields_manifest.manifest_hash,
        bench=bench,
        engine=engine_profile,
        workload_pool=pool,
        sampling_rule=rule,
        scoring_rule=scoring,
        emission_rule=emission,
        submission_fee=fee,
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
        "--bench-quantization",
        default=None,
        help="Model quantization passed to the engine (example: fp8)",
    )
    p.add_argument(
        "--bench-correctness-num-prompts",
        type=int,
        default=None,
        help="Correctness prompts (default: every request in the workload trace)",
    )
    p.add_argument(
        "--bench-correctness-min-mean-logprob",
        type=float,
        default=None,
        help="Scorer bar: mean logprob a candidate's own output must clear",
    )
    p.add_argument(
        "--bench-correctness-min-token-logprob",
        type=float,
        default=None,
        help="Scorer bar: lowest allowed token logprob at --*-min-token-quantile",
    )
    p.add_argument(
        "--bench-correctness-min-token-quantile",
        type=float,
        default=None,
        help="Fraction of worst-scored positions the min-token bar ignores",
    )
    p.add_argument(
        "--bench-correctness-min-coverage-ratio",
        type=float,
        default=None,
        help="Below this share of streamed tokens scored, the run is infra_failed",
    )
    p.add_argument(
        "--bench-correctness-max-mean-logprob-drop",
        type=float,
        default=None,
        help="How far below the baseline's mean logprob a candidate may score",
    )
    p.add_argument(
        "--emission-start-weight",
        type=float,
        default=None,
        help="Share of subnet emission a fresh leader earns (default: config)",
    )
    p.add_argument(
        "--emission-floor-weight",
        type=float,
        default=None,
        help="Share a leader still earns once the decay has run out",
    )
    p.add_argument(
        "--emission-decay-blocks",
        type=int,
        default=None,
        help="Blocks from start_weight down to floor_weight (12s per block)",
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
        help=(
            "Insert even if an open campaign already exists "
            "(only applies with --status open)"
        ),
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
        "--engine",
        default=None,
        choices=sorted(ENGINE_PRESETS),
        help=(
            "Engine build/launch profile. Omit for the implied vLLM default "
            "(keeps engine out of manifest_hash); pass sglang for SGLang campaigns"
        ),
    )
    p.add_argument(
        "--workload-pool-json",
        default=None,
        help="Path to JSON list of {sha256,url} (legacy; unused by hf_rows sampler)",
    )
    p.add_argument(
        "--sampling-rule-json",
        default=str(FIXTURE_SAMPLING_RULE),
        help="Path to JSON hf_rows sampling rule (default: synthetic_v1 fixture)",
    )
    p.add_argument(
        "--scoring-rule-json",
        default=None,
        help="Path to a JSON scoring rule (default: median_e2e_speedup)",
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
        pool = None
        if args.workload_pool_json:
            pool_path = Path(args.workload_pool_json)
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            if not isinstance(pool, list):
                raise ValueError("--workload-pool-json must be a JSON list")
        rule = None
        if args.sampling_rule_json:
            rule_path = Path(args.sampling_rule_json)
            rule = json.loads(rule_path.read_text(encoding="utf-8"))
            if not isinstance(rule, dict):
                raise ValueError("--sampling-rule-json must be a JSON object")
        scoring = None
        if args.scoring_rule_json:
            scoring_path = Path(args.scoring_rule_json)
            scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
            if not isinstance(scoring, dict):
                raise ValueError("--scoring-rule-json must be a JSON object")
        overrides = {
            "min_mean_logprob": args.bench_correctness_min_mean_logprob,
            "min_token_logprob": args.bench_correctness_min_token_logprob,
            "min_token_quantile": args.bench_correctness_min_token_quantile,
            "min_coverage_ratio": args.bench_correctness_min_coverage_ratio,
            "max_mean_logprob_drop": args.bench_correctness_max_mean_logprob_drop,
        }
        correctness_thresholds = {k: v for k, v in overrides.items() if v is not None}
        emission_overrides = {
            "start_weight": args.emission_start_weight,
            "floor_weight": args.emission_floor_weight,
            "decay_blocks": args.emission_decay_blocks,
        }
        emission = {k: v for k, v in emission_overrides.items() if v is not None}
        seed_synthetic_campaign(
            baseline_repo=args.baseline_repo,
            baseline_commit=args.baseline_commit,
            base_image_digest=args.base_image_digest,
            force=args.force,
            bench_model_repo=args.bench_model_repo,
            bench_model_revision=args.bench_model_revision,
            bench_dtype=args.bench_dtype,
            bench_max_model_len=args.bench_max_model_len,
            bench_quantization=args.bench_quantization,
            baseline_engine_image_digest=args.baseline_engine_image_digest,
            bench_gpu_count=args.bench_gpu_count,
            bench_serve_args=args.bench_serve_args,
            bench_correctness_num_prompts=args.bench_correctness_num_prompts,
            bench_correctness_thresholds=correctness_thresholds,
            workload_pool=pool,
            sampling_rule=rule,
            scoring_rule=scoring,
            emission_rule=emission,
            allow_placeholders=args.allow_placeholders,
            priority_metric=args.priority_metric,
            success_threshold=args.success_threshold,
            gpu_skus=args.gpu_skus,
            status=args.status,
            no_bench=args.no_bench,
            engine=args.engine,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
