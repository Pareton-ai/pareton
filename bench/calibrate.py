"""B7 calibration: prepare baseline-vs-baseline request + analyze reports.

Usage:
  python -m bench.calibrate prepare --engine-digest sha256:... --gpu-sku SKU \\
      --output-dir out/b7/RUN
  python -m bench.calibrate analyze --runs-dir out/b7/RUN/runs \\
      --output out/b7/RUN/calibration_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
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
    materialize_trace,
)

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
        )
    except BenchInfraError as exc:
        raise CalibrationError(str(exc)) from exc

    # Same digest on both sides (build_bench_request_dict expands baseline only).
    req["engines"]["candidate"]["image"] = engine_ref
    req["engines"]["baseline"]["image"] = engine_ref

    out_req = output_dir / "bench_request.json"
    out_req.write_text(json.dumps(req, indent=2) + "\n", encoding="utf-8")
    return req


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
            raise CalibrationError("report missing sla_bench section")
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
    warnings: list[str] = []
    if arg_s["suggested"] > 1.0:
        warnings.append(
            f"suggested argmax_mismatch_rate {arg_s['suggested']} > 1.0; unusable"
        )
        arg_s["suggested"] = None

    outer_rel = _relative_range(cand_p99_ttft)
    repro_obs = max(max(inner_rel), outer_rel)
    repro_s = suggest_threshold(
        repro_obs,
        safety_factor=safety_factor,
        floor=0.10,  # current REPRO_BAR_MAX_REL_RANGE
    )

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
        "sla_repro": {
            "inner_p99_ttft_ms_rel_range_max": max(inner_rel),
            "outer_p99_ttft_ms_rel_range": outer_rel,
            "suggested_rel_range": repro_s,
            "note": "suggestion only; do not auto-edit REPRO_BAR_MAX_REL_RANGE",
        },
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


def cmd_prepare(args: argparse.Namespace) -> int:
    try:
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench.calibrate",
        description="B7 calibration request builder + offline threshold analysis",
    )
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser(
        "prepare", help="Write baseline-vs-baseline bench_request.json"
    )
    prep.add_argument("--engine-digest", required=True)
    prep.add_argument("--gpu-sku", required=True)
    prep.add_argument("--output-dir", required=True, type=Path)
    prep.add_argument("--trace-url", default=A3A_TRACE_URL)
    prep.add_argument("--trace-sha256", default=A3A_TRACE_SHA256)
    prep.set_defaults(func=cmd_prepare)

    ana = sub.add_parser("analyze", help="Suggest thresholds from self-check reports")
    ana.add_argument("--runs-dir", required=True, type=Path)
    ana.add_argument("--safety-factor", type=float, default=DEFAULT_SAFETY_FACTOR)
    ana.add_argument("--output", required=True, type=Path)
    ana.set_defaults(func=cmd_analyze)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
