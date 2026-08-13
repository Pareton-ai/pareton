"""B7 + per-campaign correctness calibration.

Usage:
  python -m bench.calibrate prepare --engine-digest sha256:... --gpu-sku SKU \\
      --output-dir out/b7/RUN
  python -m bench.calibrate prepare --campaign-id UUID --output-dir out/calib/RUN
  python -m bench.calibrate analyze --runs-dir out/calib/RUN/runs \\
      --output out/calib/RUN/calibration_summary.json
  python -m bench.calibrate apply --campaign-id UUID \\
      --summary out/calib/RUN/calibration_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import config
from bench.mock_engine import response_shape_fingerprint
from bench.validate import RequestValidationError, validate_report_dict
from builder.registry import baseline_engine_image_ref, normalize_digest
from worker.bench_job import (
    BenchInfraError,
    build_bench_request_dict,
    campaign_calibration_fingerprint,
    materialize_trace,
)

# Default min positions written by apply (schema default stays 1 for old fixtures).
APPLY_DEFAULT_MIN_POSITIONS = 1024

REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_FIXTURE = REPO_ROOT / "fixtures" / "bench" / "vllm_completion_response_shape.json"
SHAPE_EVIDENCE_NAME = "completion_response_shape.json"

# Pinned B7 profile (same pins as campaign.seed defaults / sample_request).
B7_MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
B7_MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
B7_DTYPE = "bfloat16"
B7_MAX_MODEL_LEN = 8192

# A3a uploaded synthetic_v0 workload trace (same bytes as fixtures/campaigns/...).
A3A_TRACE_URL = (
    "https://pareton-s3.s3.us-east-2.amazonaws.com/"
    "stage0/campaigns/synthetic_v0/workload_trace.json"
)
A3A_TRACE_SHA256 = (
    "sha256:16eb6d275f99775332d376a97595da09bdd66ef587b32cd95457c53c2c283536"
)

DEFAULT_SAFETY_FACTOR = 2.0

# Permissive gates so placeholder production thresholds do not censor jitter.
_CALIB_CORR_MEAN = 1e9
_CALIB_CORR_MAX = 1e9
_CALIB_CORR_ARGMAX = 1.000001
_CALIB_PERF_RATIO = 0.0
_CALIB_SLA_P99 = 1e9


class CalibrationError(Exception):
    """Operator/input error for calibrate CLI (exit 2)."""


def _b7_bench_spec(
    *,
    baseline_engine_image_digest: str,
    trace_request_count: int,
) -> dict[str, Any]:
    """Minimal campaign.bench dict for calibration (no campaign.seed import)."""
    if trace_request_count < 1:
        raise CalibrationError("workload trace has no requests")
    # A3a synthetic_v0 has 2 prompts; clamp config defaults (often 8) to the trace.
    num_prompts = min(int(config.BENCH_CORRECTNESS_NUM_PROMPTS), trace_request_count)
    num_requests = min(int(config.BENCH_PERF_NUM_REQUESTS), trace_request_count)
    concurrency = min(int(config.BENCH_PERF_CONCURRENCY), num_requests)
    return {
        "model": {
            "hf_repo": B7_MODEL_REPO,
            "hf_revision": B7_MODEL_REVISION,
            "dtype": B7_DTYPE,
            "quantization": None,
            "max_model_len": B7_MAX_MODEL_LEN,
        },
        "baseline_engine_image_digest": baseline_engine_image_digest,
        "gpu_count": 1,
        "serve_args": None,
        "correctness": {
            "num_prompts": num_prompts,
            "max_new_tokens": int(config.BENCH_CORRECTNESS_MAX_NEW_TOKENS),
            "thresholds": {
                "mean_abs_logprob_diff": _CALIB_CORR_MEAN,
                "max_abs_logprob_diff": _CALIB_CORR_MAX,
                "argmax_mismatch_rate": _CALIB_CORR_ARGMAX,
            },
        },
        "perf_screen": {
            "num_requests": num_requests,
            "concurrency": concurrency,
            "min_throughput_ratio": _CALIB_PERF_RATIO,
        },
    }


def prepare_calibration_request(
    *,
    engine_digest: str,
    gpu_sku: str,
    output_dir: Path,
    trace_url: str = A3A_TRACE_URL,
    trace_sha256: str = A3A_TRACE_SHA256,
    fetcher: Callable[[str], bytes] | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Write validated baseline==candidate bench_request.json under output_dir."""
    digest = normalize_digest(engine_digest)
    engine_ref = baseline_engine_image_ref(digest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        trace_path = materialize_trace(
            url=trace_url,
            expected_sha256=trace_sha256,
            dest_dir=output_dir,
            fetcher=fetcher,
        )
    except BenchInfraError as exc:
        raise CalibrationError(str(exc)) from exc

    try:
        trace_doc = json.loads(trace_path.read_text(encoding="utf-8"))
        trace_n = len(trace_doc.get("requests") or [])
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise CalibrationError(f"unreadable workload trace: {exc}") from exc

    bench = _b7_bench_spec(
        baseline_engine_image_digest=digest,
        trace_request_count=trace_n,
    )

    row: dict[str, Any] = {
        "engine_image_ref": engine_ref,
        "workload_trace_sha256": trace_sha256,
        "gpu_skus": [gpu_sku],
        "bench": bench,
        "sla": {"p99_ttft_ms": _CALIB_SLA_P99, "p99_itl_ms": _CALIB_SLA_P99},
    }
    try:
        req = build_bench_request_dict(
            row,
            task_id=task_id or str(uuid4()),
            trace_path=str(trace_path.resolve()),
            require_calibration=False,
        )
    except BenchInfraError as exc:
        raise CalibrationError(str(exc)) from exc

    # Same digest on both sides (build_bench_request_dict expands baseline only).
    req["engines"]["candidate"]["image"] = engine_ref
    req["engines"]["baseline"]["image"] = engine_ref

    out_req = output_dir / "bench_request.json"
    out_req.write_text(json.dumps(req, indent=2) + "\n", encoding="utf-8")
    return req


def prepare_campaign_calibration_request(
    *,
    campaign_id: str,
    output_dir: Path,
    fetcher: Callable[[str], bytes] | None = None,
    task_id: str | None = None,
    get_campaign_fn: Callable[[str], Any] | None = None,
    trace_url: str | None = None,
    trace_sha256: str | None = None,
    mode: str = "correctness",
) -> dict[str, Any]:
    """Build a baseline==candidate bench request from a campaign row.

    Default mode is correctness (threshold calibration). Pool z-calibration
    passes mode="all" so perf_screen runs; analyze-z needs throughput_ratio.

    trace_url/trace_sha256 override the campaign's pinned trace (used by the
    generated-sample path). The model, gpu_count, and serve_args always come
    from campaign.bench.
    """
    if get_campaign_fn is None:
        from campaign.store import get_campaign as get_campaign_fn  # type: ignore[no-redef]

    manifest = get_campaign_fn(campaign_id)
    if manifest is None:
        raise CalibrationError(f"campaign not found: {campaign_id}")
    bench = manifest.bench
    if not isinstance(bench, dict):
        raise CalibrationError("campaign.bench missing")
    model = bench.get("model")
    if not isinstance(model, dict):
        raise CalibrationError("campaign.bench.model missing")
    baseline_digest = bench.get("baseline_engine_image_digest")
    if not baseline_digest:
        raise CalibrationError("campaign.bench.baseline_engine_image_digest missing")
    digest = normalize_digest(str(baseline_digest))
    engine_ref = baseline_engine_image_ref(digest)

    gpu_skus = list(manifest.gpu_skus or [])
    if not gpu_skus:
        raise CalibrationError("campaign.gpu_skus empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_trace_url = trace_url if trace_url is not None else manifest.workload_trace_url
    use_trace_sha = (
        trace_sha256 if trace_sha256 is not None else manifest.workload_trace_sha256
    )
    try:
        trace_path = materialize_trace(
            url=use_trace_url,
            expected_sha256=use_trace_sha,
            dest_dir=output_dir,
            fetcher=fetcher,
        )
    except BenchInfraError as exc:
        raise CalibrationError(str(exc)) from exc

    try:
        trace_doc = json.loads(trace_path.read_text(encoding="utf-8"))
        trace_n = len(trace_doc.get("requests") or [])
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise CalibrationError(f"unreadable workload trace: {exc}") from exc
    if trace_n < 1:
        raise CalibrationError("workload trace has no requests")

    corr_cfg = dict(bench.get("correctness") or {})
    num_prompts = min(
        int(corr_cfg.get("num_prompts", config.BENCH_CORRECTNESS_NUM_PROMPTS)),
        trace_n,
    )
    max_new = int(
        corr_cfg.get("max_new_tokens", config.BENCH_CORRECTNESS_MAX_NEW_TOKENS)
    )
    calib_bench = {
        "model": dict(model),
        "baseline_engine_image_digest": digest,
        "gpu_count": int(bench.get("gpu_count") or 1),
        "serve_args": list(bench.get("serve_args") or []) or None,
        "correctness": {
            "num_prompts": num_prompts,
            "max_new_tokens": max_new,
            "min_positions_compared": int(corr_cfg.get("min_positions_compared", 1)),
            "thresholds": {
                "mean_abs_logprob_diff": _CALIB_CORR_MEAN,
                "max_abs_logprob_diff": _CALIB_CORR_MAX,
                "argmax_mismatch_rate": _CALIB_CORR_ARGMAX,
            },
        },
        "perf_screen": {
            "num_requests": min(
                int(config.BENCH_PERF_NUM_REQUESTS),
                trace_n,
            ),
            "concurrency": 1,
            "min_throughput_ratio": _CALIB_PERF_RATIO,
        },
    }
    sla = manifest.sla.to_dict() if manifest.sla is not None else {}
    row: dict[str, Any] = {
        "engine_image_ref": engine_ref,
        "workload_trace_sha256": use_trace_sha,
        "gpu_skus": gpu_skus,
        "bench": calib_bench,
        "sla": {
            "p99_ttft_ms": sla.get("p99_ttft_ms") or _CALIB_SLA_P99,
            "p99_itl_ms": sla.get("p99_itl_ms") or _CALIB_SLA_P99,
        },
    }
    try:
        req = build_bench_request_dict(
            row,
            task_id=task_id or str(uuid4()),
            trace_path=str(trace_path.resolve()),
            require_calibration=False,
        )
    except BenchInfraError as exc:
        raise CalibrationError(str(exc)) from exc

    req["mode"] = mode
    req["engines"]["candidate"]["image"] = engine_ref
    req["engines"]["baseline"]["image"] = engine_ref

    out_req = output_dir / "bench_request.json"
    out_req.write_text(json.dumps(req, indent=2) + "\n", encoding="utf-8")
    return req


def correctness_dict_from_summary(
    summary: dict[str, Any],
    *,
    existing_correctness: dict[str, Any] | None = None,
    campaign_fingerprint: dict[str, Any],
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> dict[str, Any]:
    """Build campaigns.bench.correctness from an analyze summary."""
    corr = summary.get("correctness")
    if not isinstance(corr, dict):
        raise CalibrationError("summary missing correctness section")
    thresholds: dict[str, float] = {}
    for key in (
        "mean_abs_logprob_diff",
        "max_abs_logprob_diff",
        "argmax_mismatch_rate",
    ):
        block = corr.get(key)
        if not isinstance(block, dict) or block.get("suggested") is None:
            raise CalibrationError(f"summary correctness.{key}.suggested missing")
        thresholds[key] = float(block["suggested"])

    existing = dict(existing_correctness or {})
    num_prompts = int(existing.get("num_prompts", config.BENCH_CORRECTNESS_NUM_PROMPTS))
    max_new_tokens = int(
        existing.get("max_new_tokens", config.BENCH_CORRECTNESS_MAX_NEW_TOKENS)
    )
    planned = num_prompts * max_new_tokens
    if planned < 1:
        raise CalibrationError(
            f"num_prompts * max_new_tokens must be >= 1, got {planned}"
        )
    # Cap at planned size: positions_compared can never exceed the generation budget.
    desired_min = int(
        existing.get("min_positions_compared", APPLY_DEFAULT_MIN_POSITIONS)
    )
    min_positions = min(desired_min, planned)
    out: dict[str, Any] = {
        "num_prompts": num_prompts,
        "max_new_tokens": max_new_tokens,
        "min_positions_compared": min_positions,
        "thresholds": thresholds,
        "calibration": {
            "thresholds": dict(thresholds),
            "fingerprint": dict(campaign_fingerprint),
            "safety_factor": float(safety_factor),
            "calibrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observed": corr.get("observed") or {},
            "summary_fingerprint": summary.get("fingerprint") or {},
        },
    }
    return out


def discover_run_dirs(runs_dir: Path) -> list[Path]:
    """Return ordered run directories (run-NNN) or a single flat output dir."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise CalibrationError(f"runs dir not found: {runs_dir}")
    numbered = sorted(
        p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run-")
    )
    if numbered:
        return numbered
    if (runs_dir / "bench_report.json").is_file():
        return [runs_dir]
    raise CalibrationError(
        f"no run-* directories or bench_report.json under {runs_dir}"
    )


def _require_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CalibrationError(f"{name} is not numeric")
    f = float(value)
    if not math.isfinite(f) or f < 0:
        raise CalibrationError(f"{name} must be finite and >= 0, got {value!r}")
    return f


def _load_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot load report {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"report is not an object: {path}")
    try:
        validate_report_dict(data)
    except RequestValidationError as exc:
        raise CalibrationError(f"invalid report {path}: {exc}") from exc
    return data


def _fingerprint_key(report: dict[str, Any]) -> tuple[str, str, str, str]:
    fp = report["inputs_fingerprint"]
    base = str(fp["baseline_image_digest"]).lower()
    cand = str(fp["candidate_image_digest"]).lower()
    if base != cand:
        raise CalibrationError(
            "baseline and candidate digests differ "
            f"(baseline={base} candidate={cand}); not a self-check report"
        )
    return (
        base,
        str(fp["model_revision"]),
        str(fp["trace_sha256"]).lower(),
        str(fp["model_repo"]),
    )


def suggest_threshold(
    observed_max: float,
    *,
    safety_factor: float,
    floor: float,
) -> dict[str, Any]:
    """max(observed * factor, floor); flag all-zero observations."""
    if safety_factor <= 0:
        raise CalibrationError(f"safety_factor must be > 0, got {safety_factor}")
    raw = observed_max * safety_factor
    suggested = max(raw, floor)
    return {
        "observed_max": observed_max,
        "suggested": suggested,
        "all_observed_zero": observed_max == 0.0,
        "floor": floor,
        "safety_factor": safety_factor,
    }


def correctness_would_pass(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> bool:
    """Mirror Module A strict-< gate."""
    return (
        metrics["mean_abs_logprob_diff"] < thresholds["mean_abs_logprob_diff"]
        and metrics["max_abs_logprob_diff"] < thresholds["max_abs_logprob_diff"]
        and metrics["argmax_mismatch_rate"] < thresholds["argmax_mismatch_rate"]
    )


def _relative_range(values: list[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    if med == 0:
        return 0.0
    return float((max(values) - min(values)) / med)


def _fingerprint_paths_diff(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    """Return human-readable paths where structural fingerprints differ."""
    diffs: list[str] = []
    if type(expected) is not type(actual):
        diffs.append(
            f"{prefix or '<'}: type {type(expected).__name__} vs {type(actual).__name__}"
        )
        return diffs
    if isinstance(expected, dict):
        exp_keys = set(expected)
        act_keys = set(actual)
        for k in sorted(exp_keys - act_keys):
            diffs.append(f"{prefix}/{k}: missing in actual")
        for k in sorted(act_keys - exp_keys):
            diffs.append(f"{prefix}/{k}: unexpected in actual")
        for k in sorted(exp_keys & act_keys):
            diffs.extend(
                _fingerprint_paths_diff(
                    expected[k], actual[k], prefix=f"{prefix}/{k}" if prefix else k
                )
            )
        return diffs
    if isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(
                f"{prefix or '<'}: list length {len(expected)} vs {len(actual)}"
            )
            return diffs
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            diffs.extend(_fingerprint_paths_diff(e, a, prefix=f"{prefix}[{i}]"))
        return diffs
    if expected != actual:
        diffs.append(f"{prefix or '<'}: {expected!r} vs {actual!r}")
    return diffs


def compare_shape_to_fixture(
    captured: dict[str, Any],
    *,
    fixture_path: Path = SHAPE_FIXTURE,
) -> dict[str, Any]:
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    expected = fixture.get("fingerprint")
    if not isinstance(expected, dict):
        raise CalibrationError(f"fixture missing fingerprint object: {fixture_path}")
    # Captured evidence may be raw response or already a fingerprint.
    if "fingerprint" in captured and isinstance(captured["fingerprint"], dict):
        actual = captured["fingerprint"]
    elif "choices" in captured and isinstance(captured.get("choices"), list):
        # Looks like a live response object — fingerprint it.
        actual = response_shape_fingerprint(captured)
    else:
        actual = captured
    diffs = _fingerprint_paths_diff(expected, actual)
    return {"match": not diffs, "diffs": diffs}


def analyze_reports(
    report_paths: list[Path],
    *,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    shape_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Compute suggested thresholds from validated self-check reports."""
    if not report_paths:
        raise CalibrationError("no reports to analyze")

    reports: list[dict[str, Any]] = []
    for p in report_paths:
        reports.append(_load_report(Path(p)))

    keys = [_fingerprint_key(r) for r in reports]
    if len(set(keys)) != 1:
        raise CalibrationError(
            "reports disagree on digest/model/trace fingerprint: "
            + ", ".join(repr(k) for k in sorted(set(keys)))
        )

    means: list[float] = []
    maxes: list[float] = []
    argmaxes: list[float] = []
    inner_rel: list[float] = []
    cand_p99_ttft: list[float] = []
    has_sla = False
    warnings: list[str] = []

    for r in reports:
        corr = r.get("correctness")
        if not isinstance(corr, dict):
            raise CalibrationError("report missing correctness section")
        means.append(
            _require_finite("mean_abs_logprob_diff", corr["mean_abs_logprob_diff"])
        )
        maxes.append(
            _require_finite("max_abs_logprob_diff", corr["max_abs_logprob_diff"])
        )
        argmaxes.append(
            _require_finite("argmax_mismatch_rate", corr["argmax_mismatch_rate"])
        )
        sla = r.get("sla_bench")
        if not isinstance(sla, dict):
            continue
        has_sla = True
        var = sla.get("cross_rep_variance") or {}
        if "p99_ttft_ms_rel_range" not in var:
            raise CalibrationError(
                "sla_bench.cross_rep_variance missing p99_ttft_ms_rel_range"
            )
        inner_rel.append(
            _require_finite("p99_ttft_ms_rel_range", var["p99_ttft_ms_rel_range"])
        )
        cand = sla.get("candidate") or {}
        ttft = (cand.get("ttft_ms") or {}).get("p99")
        if ttft is None:
            raise CalibrationError("sla_bench.candidate.ttft_ms.p99 missing")
        cand_p99_ttft.append(_require_finite("candidate.ttft_ms.p99", ttft))

    mean_s = suggest_threshold(
        max(means),
        safety_factor=safety_factor,
        floor=float(config.BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF),
    )
    max_s = suggest_threshold(
        max(maxes),
        safety_factor=safety_factor,
        floor=float(config.BENCH_CORRECTNESS_MAX_ABS_LOGPROB_DIFF),
    )
    arg_s = suggest_threshold(
        max(argmaxes),
        safety_factor=safety_factor,
        floor=float(config.BENCH_CORRECTNESS_ARGMAX_MISMATCH_RATE),
    )
    if arg_s["suggested"] > 1.0:
        warnings.append(
            f"suggested argmax_mismatch_rate {arg_s['suggested']} > 1.0; unusable"
        )
        arg_s["suggested"] = None

    sla_repro: dict[str, Any] | None = None
    if has_sla:
        outer_rel = _relative_range(cand_p99_ttft)
        repro_obs = max(max(inner_rel), outer_rel)
        repro_s = suggest_threshold(
            repro_obs,
            safety_factor=safety_factor,
            floor=0.335,  # current REPRO_BAR_MAX_REL_RANGE (B7 2026-08-03)
        )
        sla_repro = {
            "inner_p99_ttft_ms_rel_range_max": max(inner_rel),
            "outer_p99_ttft_ms_rel_range": outer_rel,
            "suggested_rel_range": repro_s,
            "note": "suggestion only; do not auto-edit REPRO_BAR_MAX_REL_RANGE",
        }

    shape_status: dict[str, Any] = {"checked": False, "match": None, "diffs": []}
    if shape_paths:
        for sp in shape_paths:
            if not Path(sp).is_file():
                continue
            try:
                captured = json.loads(Path(sp).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(f"shape evidence unreadable {sp}: {exc}")
                continue
            if not isinstance(captured, dict):
                warnings.append(f"shape evidence not an object: {sp}")
                continue
            cmp = compare_shape_to_fixture(captured)
            shape_status = {
                "checked": True,
                "match": cmp["match"],
                "diffs": cmp["diffs"],
                "path": str(sp),
            }
            if not cmp["match"]:
                warnings.append(f"response shape drift vs fixture: {cmp['diffs']}")
            break

    digest, revision, trace_sha, model_repo = keys[0]
    return {
        "schema_version": 1,
        "n_reports": len(reports),
        "fingerprint": {
            "engine_digest": digest,
            "model_repo": model_repo,
            "model_revision": revision,
            "trace_sha256": trace_sha,
        },
        "correctness": {
            "mean_abs_logprob_diff": mean_s,
            "max_abs_logprob_diff": max_s,
            "argmax_mismatch_rate": arg_s,
            "observed": {
                "mean_abs_logprob_diff": means,
                "max_abs_logprob_diff": maxes,
                "argmax_mismatch_rate": argmaxes,
            },
        },
        "sla_repro": sla_repro,
        "shape": shape_status,
        "warnings": warnings,
        "report_paths": [str(p) for p in report_paths],
    }


def analyze_runs_dir(
    runs_dir: Path,
    *,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> dict[str, Any]:
    run_dirs = discover_run_dirs(runs_dir)
    reports = [d / "bench_report.json" for d in run_dirs]
    missing = [str(p) for p in reports if not p.is_file()]
    if missing:
        raise CalibrationError(f"missing bench_report.json: {missing}")
    shapes = [d / "evidence" / "correctness" / SHAPE_EVIDENCE_NAME for d in run_dirs]
    return analyze_reports(reports, safety_factor=safety_factor, shape_paths=shapes)


def prepare_pool_calibration_requests(
    *,
    campaign_id: str,
    output_dir: Path,
    max_samples: int | None = None,
    fetcher: Callable[[str], bytes] | None = None,
    get_campaign_fn: Callable[[str], Any] | None = None,
    row_fetcher: Callable[[int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Write one baseline==candidate request per generated calib trace.

    Requires campaign.sampling_rule.type == hf_rows. Caps at ``max_samples``
    (default ``PARETON_CALIB_MIN_SAMPLES``).
    """
    from bench.sampler import calib_seed, fetch_hf_row, generate_trace, parse_sampling_rule

    if get_campaign_fn is None:
        from campaign.store import get_campaign as get_campaign_fn  # type: ignore[no-redef]

    manifest = get_campaign_fn(campaign_id)
    if manifest is None:
        raise CalibrationError(f"campaign not found: {campaign_id}")
    if not isinstance(manifest.bench, dict):
        raise CalibrationError("campaign.bench missing")
    baseline_digest = manifest.bench.get("baseline_engine_image_digest")
    if not baseline_digest:
        raise CalibrationError("campaign.bench.baseline_engine_image_digest missing")
    gpu_skus = list(manifest.gpu_skus or [])
    if not gpu_skus:
        raise CalibrationError("campaign.gpu_skus empty")

    try:
        rule = parse_sampling_rule(manifest.sampling_rule)
    except Exception as exc:
        raise CalibrationError(f"campaign.sampling_rule invalid: {exc}") from exc

    limit = int(max_samples if max_samples is not None else config.CALIB_MIN_SAMPLES)
    if limit < 1:
        raise CalibrationError("max_samples must be >= 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_set = bool(
        (os.environ.get("HF_TOKEN") or os.environ.get("PARETON_HF_TOKEN") or "").strip()
    )
    print(
        f"prepare --pool campaign={campaign_id} "
        f"dataset={rule['dataset']}@{rule['revision'][:12]} "
        f"n_prompts={rule['n_prompts']} n_rows={rule['n_rows']} "
        f"samples={limit} hf_auth={'yes' if token_set else 'no'}",
        flush=True,
    )
    written: list[dict[str, Any]] = []
    seen: set[str] = set()
    cid = str(campaign_id)
    hf_fetch = row_fetcher or (lambda idx: fetch_hf_row(rule, idx))
    for i in range(limit):
        seed = calib_seed(cid, i)
        print(f"generating sample {i + 1}/{limit} (sample-{i:03d}) ...", flush=True)
        t0 = time.monotonic()
        try:
            sampled = generate_trace(
                rule=rule,
                seed_hex=seed,
                row_fetcher=hf_fetch,
            )
        except Exception as exc:
            raise CalibrationError(f"calib sample {i} generate failed: {exc}") from exc
        sha = sampled.sha256.lower()
        if sha in seen:
            raise CalibrationError(f"repeated generated trace at sample {i}: {sha}")
        seen.add(sha)
        sample_dir = output_dir / f"sample-{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        trace_path = sample_dir / "workload_trace.json"
        trace_path.write_bytes(sampled.body)
        body = sampled.body
        req = prepare_campaign_calibration_request(
            campaign_id=campaign_id,
            output_dir=sample_dir,
            fetcher=fetcher or (lambda _u, _b=body: _b),
            get_campaign_fn=lambda _cid: manifest,
            trace_url=f"file://{trace_path.resolve()}",
            trace_sha256=sha,
            mode="all",
        )
        written.append(req)
        elapsed = time.monotonic() - t0
        print(
            f"wrote sample-{i:03d} prompts={len(sampled.row_indices)} "
            f"{sha[:19]} {elapsed:.1f}s",
            flush=True,
        )
    return written


def analyze_z_calibration(
    report_paths: list[Path],
    *,
    min_samples: int | None = None,
) -> dict[str, Any]:
    """Build campaigns.calibration mean/std from baseline-vs-baseline reports.

    Rejects repeated trace_sha256 values and zero-variance metrics.
    Requires at least ``min_samples`` reports (default PARETON_CALIB_MIN_SAMPLES).
    """
    from bench.promote import PROMOTION_METRICS, PromoteError, extract_observed_metrics

    floor = int(min_samples if min_samples is not None else config.CALIB_MIN_SAMPLES)
    if not report_paths:
        raise CalibrationError("no reports to analyze")
    if len(report_paths) < floor:
        raise CalibrationError(
            f"need at least {floor} reports for z-calibration, got {len(report_paths)}"
        )

    reports = [_load_report(Path(p)) for p in report_paths]
    traces: list[str] = []
    for r in reports:
        fp = r.get("inputs_fingerprint") or {}
        sha = str(fp.get("trace_sha256") or "").lower()
        if not sha:
            raise CalibrationError("report missing inputs_fingerprint.trace_sha256")
        traces.append(sha)
    if len(set(traces)) != len(traces):
        raise CalibrationError(
            "repeated traces in z-calibration set; each sample must be distinct"
        )

    # Self-check: baseline digest must equal candidate.
    for r in reports:
        _fingerprint_key(r)

    vectors: list[dict[str, float]] = []
    for r in reports:
        try:
            vectors.append(extract_observed_metrics(r))
        except PromoteError as exc:
            raise CalibrationError(str(exc)) from exc

    metrics_out: dict[str, Any] = {}
    for name in PROMOTION_METRICS:
        values = [v[name] for v in vectors]
        mean = float(statistics.fmean(values))
        # population std so a fixed calibrator set is reproducible
        std = float(statistics.pstdev(values))
        if std == 0.0:
            raise CalibrationError(
                f"zero variance for metric {name}; refuse z-calibration"
            )
        metrics_out[name] = {
            "mean": mean,
            "std": std,
            "n": len(values),
            "min": min(values),
            "max": max(values),
        }

    digest, revision, _trace, model_repo = _fingerprint_key(reports[0])
    return {
        "schema_version": 1,
        "n_reports": len(reports),
        "min_samples": floor,
        "metrics": metrics_out,
        "trace_sha256s": traces,
        "fingerprint": {
            "engine_digest": digest,
            "model_repo": model_repo,
            "model_revision": revision,
        },
        "report_paths": [str(p) for p in report_paths],
        "calibrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def cmd_prepare(args: argparse.Namespace) -> int:
    try:
        if getattr(args, "pool", False):
            if not args.campaign_id:
                raise CalibrationError("--pool requires --campaign-id")
            reqs = prepare_pool_calibration_requests(
                campaign_id=str(args.campaign_id),
                output_dir=Path(args.output_dir),
                max_samples=args.max_samples,
            )
            print(f"wrote {len(reqs)} generated sample requests under {args.output_dir}")
            print(
                f"calib knobs: pods={config.CALIB_PODS} "
                f"samples_per_pod={config.CALIB_SAMPLES_PER_POD} "
                f"min_samples={config.CALIB_MIN_SAMPLES}"
            )
            return 0
        if args.campaign_id:
            req = prepare_campaign_calibration_request(
                campaign_id=str(args.campaign_id),
                output_dir=Path(args.output_dir),
            )
        else:
            if not args.engine_digest or not args.gpu_sku:
                raise CalibrationError(
                    "prepare requires --campaign-id, or both --engine-digest and --gpu-sku"
                )
            req = prepare_calibration_request(
                engine_digest=args.engine_digest,
                gpu_sku=args.gpu_sku,
                output_dir=Path(args.output_dir),
                trace_url=args.trace_url,
                trace_sha256=args.trace_sha256,
            )
    except (CalibrationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {Path(args.output_dir) / 'bench_request.json'}")
    print(f"baseline=candidate={req['engines']['baseline']['image']}")
    print(f"model={req['model']['hf_repo']}@{req['model']['hf_revision']}")
    print(f"gpu_sku={req['hardware']['gpu_sku_expected']}")
    print(f"mode={req['mode']}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        summary = analyze_runs_dir(
            Path(args.runs_dir), safety_factor=float(args.safety_factor)
        )
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for w in summary.get("warnings") or []:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def assert_summary_matches_campaign(
    summary: dict[str, Any], *, bench: dict[str, Any], trace_sha256: str
) -> None:
    """Refuse apply when analyze summary was measured on a different campaign pin."""
    summary_fp = summary.get("fingerprint")
    if not isinstance(summary_fp, dict):
        raise CalibrationError("summary missing fingerprint object")
    model = bench.get("model") or {}
    try:
        campaign_digest = normalize_digest(
            str(bench.get("baseline_engine_image_digest") or "")
        )
    except ValueError as exc:
        raise CalibrationError(
            f"campaign baseline_engine_image_digest invalid: {exc}"
        ) from exc
    expected = {
        "engine_digest": campaign_digest,
        "model_repo": str(model.get("hf_repo") or ""),
        "model_revision": str(model.get("hf_revision") or ""),
        "trace_sha256": str(trace_sha256 or "").lower(),
    }
    for key, expect in expected.items():
        got_raw = summary_fp.get(key)
        if key == "engine_digest":
            try:
                got = normalize_digest(str(got_raw or ""))
            except ValueError as exc:
                raise CalibrationError(
                    f"summary fingerprint.engine_digest invalid: {exc}"
                ) from exc
        elif key == "trace_sha256":
            got = str(got_raw or "").lower()
        else:
            got = str(got_raw or "")
        if got != expect:
            raise CalibrationError(
                f"summary fingerprint {key} mismatch: {got!r} != {expect!r}"
            )


def cmd_apply(args: argparse.Namespace) -> int:
    try:
        summary_path = Path(args.summary)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"cannot load summary: {exc}") from exc
        if not isinstance(summary, dict):
            raise CalibrationError("summary must be a JSON object")

        from campaign.store import (
            apply_campaign_correctness_calibration,
            get_campaign,
        )

        manifest = get_campaign(str(args.campaign_id))
        if manifest is None:
            raise CalibrationError(f"campaign not found: {args.campaign_id}")
        if not isinstance(manifest.bench, dict):
            raise CalibrationError("campaign.bench missing")
        assert_summary_matches_campaign(
            summary,
            bench=manifest.bench,
            trace_sha256=str(manifest.workload_trace_sha256 or ""),
        )
        row = {
            "workload_trace_sha256": manifest.workload_trace_sha256,
        }
        fp = campaign_calibration_fingerprint(manifest.bench, row)
        correctness = correctness_dict_from_summary(
            summary,
            existing_correctness=manifest.bench.get("correctness")
            if isinstance(manifest.bench.get("correctness"), dict)
            else None,
            campaign_fingerprint=fp,
            safety_factor=float(args.safety_factor),
        )
        new_hash = apply_campaign_correctness_calibration(
            str(args.campaign_id),
            correctness,
            approver=str(args.approver),
        )
    except (CalibrationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"applied calibration to campaign {args.campaign_id}")
    print(f"manifest_hash={new_hash}")
    return 0


def cmd_analyze_z(args: argparse.Namespace) -> int:
    try:
        run_dirs = discover_run_dirs(Path(args.runs_dir))
        reports = [d / "bench_report.json" for d in run_dirs]
        missing = [str(p) for p in reports if not p.is_file()]
        if missing:
            raise CalibrationError(f"missing bench_report.json: {missing}")
        summary = analyze_z_calibration(
            reports, min_samples=int(args.min_samples) if args.min_samples else None
        )
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"n_reports={summary['n_reports']} metrics={list(summary['metrics'])}")
    return 0


def cmd_apply_z(args: argparse.Namespace) -> int:
    try:
        summary_path = Path(args.summary)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"cannot load summary: {exc}") from exc
        if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), dict):
            raise CalibrationError("summary must include metrics object")
        from campaign.store import apply_campaign_z_calibration

        manifest_hash = apply_campaign_z_calibration(
            str(args.campaign_id),
            summary,
            approver=str(args.approver),
        )
    except (CalibrationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"applied z-calibration to campaign {args.campaign_id}")
    print(f"manifest_hash={manifest_hash}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench.calibrate",
        description="Correctness + z-score calibration: prepare, analyze, apply",
    )
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser(
        "prepare", help="Write baseline-vs-baseline bench_request.json"
    )
    prep.add_argument("--campaign-id", default=None)
    prep.add_argument("--engine-digest", default=None)
    prep.add_argument("--gpu-sku", default=None)
    prep.add_argument("--output-dir", required=True, type=Path)
    prep.add_argument("--trace-url", default=A3A_TRACE_URL)
    prep.add_argument("--trace-sha256", default=A3A_TRACE_SHA256)
    prep.add_argument(
        "--pool",
        action="store_true",
        help="With --campaign-id, generate N on-the-fly traces and write one request each",
    )
    prep.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=f"Cap generated prepare count (default {config.CALIB_MIN_SAMPLES})",
    )
    prep.set_defaults(func=cmd_prepare)

    ana = sub.add_parser("analyze", help="Suggest thresholds from self-check reports")
    ana.add_argument("--runs-dir", required=True, type=Path)
    ana.add_argument("--safety-factor", type=float, default=DEFAULT_SAFETY_FACTOR)
    ana.add_argument("--output", required=True, type=Path)
    ana.set_defaults(func=cmd_analyze)

    ana_z = sub.add_parser(
        "analyze-z",
        help="Build z-score mean/std from distinct-trace self-check reports",
    )
    ana_z.add_argument("--runs-dir", required=True, type=Path)
    ana_z.add_argument("--output", required=True, type=Path)
    ana_z.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=f"Minimum reports required (default {config.CALIB_MIN_SAMPLES})",
    )
    ana_z.set_defaults(func=cmd_analyze_z)

    apply_p = sub.add_parser(
        "apply", help="Write calibrated correctness into a draft campaign"
    )
    apply_p.add_argument("--campaign-id", required=True)
    apply_p.add_argument("--summary", required=True, type=Path)
    apply_p.add_argument("--safety-factor", type=float, default=DEFAULT_SAFETY_FACTOR)
    apply_p.add_argument("--approver", default="pareton-admin")
    apply_p.set_defaults(func=cmd_apply)

    apply_z = sub.add_parser(
        "apply-z", help="Write z-score distribution into campaigns.calibration"
    )
    apply_z.add_argument("--campaign-id", required=True)
    apply_z.add_argument("--summary", required=True, type=Path)
    apply_z.add_argument("--approver", default="pareton-admin")
    apply_z.set_defaults(func=cmd_apply_z)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
