"""Build bench_request.json and run Stages 1–3 for a claimed bench job."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import config
from bench.validate import (
    RequestValidationError,
    extract_image_digest,
    is_digest_pinned_image,
    sha256_bytes,
    validate_bench_request_dict,
    validate_report_dict,
)
from builder.registry import baseline_engine_image_ref
from bench.promote import PromoteError, decide_promotion
from bench.sampler import (
    SamplerError,
    parse_sampling_rule,
    sample_workload,
)
from campaign.cross_env import SPEEDUP_METRIC_KEYS, validate_cross_env
from campaign.store import finalize_bench_job, record_submission_sample, set_job_status
from gate.types import SubmissionState
from observability import events as obs
from observability.events import Timer

logger = logging.getLogger(__name__)

REASON_CANDIDATE_NOT_PINNED = "candidate_image_not_digest_pinned"

# Worst-verdict precedence (lower index wins).
_VERDICT_RANK = {
    "fail_correctness": 0,
    "fail_perf_screen": 1,
    "fail_sla": 2,
    "fail_promotion": 3,
    "pass": 4,
}


class BenchInfraError(Exception):
    """Infra/operator failure: fail the job, no rejection events."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def fetch_trace_bytes(
    url: str,
    *,
    max_bytes: int | None = None,
    fetcher: Callable[[str], bytes] | None = None,
) -> bytes:
    """Load workload trace bytes from file:// or https (bounded)."""
    limit = max_bytes if max_bytes is not None else config.TRACE_MAX_BYTES
    if fetcher is not None:
        data = fetcher(url)
        if len(data) > limit:
            raise BenchInfraError("trace_too_large", f">{limit} bytes")
        return data
    if url.startswith("file://"):
        path = Path(url[7:])
        data = path.read_bytes()
        if len(data) > limit:
            raise BenchInfraError("trace_too_large", f">{limit} bytes")
        return data
    if url.startswith("https://") or url.startswith("http://"):
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read(limit + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BenchInfraError("trace_fetch_failed", str(exc)) from exc
        if len(data) > limit:
            raise BenchInfraError("trace_too_large", f">{limit} bytes")
        return data
    raise BenchInfraError("trace_url_unsupported", url)


def materialize_trace(
    *,
    url: str,
    expected_sha256: str,
    dest_dir: Path,
    fetcher: Callable[[str], bytes] | None = None,
) -> Path:
    raw = fetch_trace_bytes(url, fetcher=fetcher)
    actual = sha256_bytes(raw)
    if actual.lower() != expected_sha256.lower():
        raise BenchInfraError(
            "trace_sha256_mismatch",
            f"expected {expected_sha256}, got {actual}",
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "workload_trace.json"
    path.write_bytes(raw)
    return path


def campaign_calibration_fingerprint(
    bench: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    """Key fields that must match a stored correctness.calibration fingerprint.

    Per-submission trace sha256 is not included: sampled campaigns generate a
    new trace each job. Bind dataset pin fields instead when sampling is on.
    """
    model = bench.get("model") or {}
    quant = model.get("quantization")
    fp: dict[str, Any] = {
        "model_repo": str(model.get("hf_repo") or ""),
        "model_revision": str(model.get("hf_revision") or ""),
        "baseline_engine_image_digest": str(
            bench.get("baseline_engine_image_digest") or ""
        ).lower(),
        "serve_args": [str(x) for x in (bench.get("serve_args") or [])],
        "dtype": str(model.get("dtype") or ""),
        "quantization": "" if quant is None else str(quant),
        "max_model_len": int(model.get("max_model_len") or 0),
        "gpu_count": int(bench.get("gpu_count") or 1),
    }
    rule = _parse_json_field(row.get("sampling_rule"))
    if isinstance(rule, dict) and str(rule.get("type") or "") == "hf_rows":
        fp["sampling_dataset"] = str(rule.get("dataset") or "")
        fp["sampling_revision"] = str(rule.get("revision") or "")
        fp["sampling_n_prompts"] = int(rule.get("n_prompts") or 0)
        fp["sampling_algo_version"] = int(rule.get("algo_version") or 0)
    return fp


def build_bench_request_dict(
    row: dict[str, Any],
    *,
    task_id: str | None = None,
    trace_path: str,
    gpu_sku: str | None = None,
    fetcher: Callable[[str], bytes] | None = None,
    require_calibration: bool = False,
) -> dict[str, Any]:
    """Operator-pinned bench_request from campaign + submission (image only from miner).

    ``require_calibration`` is True for real GPU benches. Mock/calibrate paths leave
    it False so unit tests and B7 prepare stay unblocked.
    """
    del fetcher  # binding happens in materialize_trace before this is called
    bench = _parse_json_field(row.get("bench"))
    if not isinstance(bench, dict):
        raise BenchInfraError("campaign_bench_missing")
    model = bench.get("model")
    if not isinstance(model, dict):
        raise BenchInfraError("campaign_bench_model_missing")

    candidate = row.get("engine_image_ref") or ""
    if not is_digest_pinned_image(str(candidate)):
        raise BenchInfraError(REASON_CANDIDATE_NOT_PINNED, str(candidate))

    baseline_raw = bench.get("baseline_engine_image_digest")
    if not baseline_raw or not is_digest_pinned_image(str(baseline_raw)):
        raise BenchInfraError("baseline_image_not_digest_pinned", str(baseline_raw))
    try:
        baseline = baseline_engine_image_ref(str(baseline_raw))
    except ValueError as exc:
        raise BenchInfraError("baseline_image_not_digest_pinned", str(exc)) from exc

    gpu_skus = _parse_json_field(row.get("gpu_skus")) or []
    if not gpu_skus:
        raise BenchInfraError("gpu_skus_empty")
    sku = str(gpu_sku) if gpu_sku is not None else str(gpu_skus[0])
    gpu_count = int(bench.get("gpu_count") or 1)

    max_model_len = int(model["max_model_len"])
    dtype = str(model.get("dtype") or "bfloat16")
    extra_serve = list(bench.get("serve_args") or [])
    serve_args = [
        "--model",
        "/model",
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
    ]
    quantization = model.get("quantization")
    if quantization is not None and str(quantization).strip() != "":
        serve_args.extend(["--quantization", str(quantization)])
    serve_args.extend(str(x) for x in extra_serve)

    corr_cfg = dict(bench.get("correctness") or {})
    if require_calibration:
        cal = corr_cfg.get("calibration")
        if not isinstance(cal, dict) or cal.get("thresholds") is None:
            raise BenchInfraError("campaign_correctness_calibration_missing")
        expect_fp = campaign_calibration_fingerprint(bench, row)
        got_fp = cal.get("fingerprint")
        if not isinstance(got_fp, dict):
            raise BenchInfraError(
                "campaign_correctness_calibration_stale", "fingerprint"
            )
        for key, expect in expect_fp.items():
            if got_fp.get(key) != expect:
                raise BenchInfraError(
                    "campaign_correctness_calibration_stale",
                    f"{key}: {got_fp.get(key)!r} != {expect!r}",
                )
    thresholds = dict(corr_cfg.get("thresholds") or {})
    correctness = {
        "num_prompts": int(
            corr_cfg.get("num_prompts", config.BENCH_CORRECTNESS_NUM_PROMPTS)
        ),
        "max_new_tokens": int(
            corr_cfg.get("max_new_tokens", config.BENCH_CORRECTNESS_MAX_NEW_TOKENS)
        ),
        "min_positions_compared": int(corr_cfg.get("min_positions_compared", 1)),
        "thresholds": {
            "mean_abs_logprob_diff": float(
                thresholds.get(
                    "mean_abs_logprob_diff",
                    config.BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF,
                )
            ),
            "max_abs_logprob_diff": float(
                thresholds.get(
                    "max_abs_logprob_diff",
                    config.BENCH_CORRECTNESS_MAX_ABS_LOGPROB_DIFF,
                )
            ),
            "argmax_mismatch_rate": float(
                thresholds.get(
                    "argmax_mismatch_rate",
                    config.BENCH_CORRECTNESS_ARGMAX_MISMATCH_RATE,
                )
            ),
        },
    }
    planned = correctness["num_prompts"] * correctness["max_new_tokens"]
    floor = 1.5 / planned if planned > 0 else 0.0
    correctness["thresholds"]["argmax_mismatch_rate"] = max(
        correctness["thresholds"]["argmax_mismatch_rate"], floor
    )

    perf_cfg = dict(bench.get("perf_screen") or {})
    perf_screen = {
        "num_requests": int(
            perf_cfg.get("num_requests", config.BENCH_PERF_NUM_REQUESTS)
        ),
        "concurrency": int(perf_cfg.get("concurrency", config.BENCH_PERF_CONCURRENCY)),
        "min_throughput_ratio": float(
            perf_cfg.get("min_throughput_ratio", config.BENCH_PERF_MIN_THROUGHPUT_RATIO)
        ),
    }

    sla = _parse_json_field(row.get("sla")) or {}
    ttft = sla.get("p99_ttft_ms")
    itl = sla.get("p99_itl_ms")
    if ttft is None or itl is None:
        raise BenchInfraError("sla_thresholds_missing")
    sla_bench = {
        "repetitions": int(config.BENCH_SLA_REPETITIONS),
        "warmup_requests": int(config.BENCH_SLA_WARMUP_REQUESTS),
        "thresholds": {
            "p99_ttft_ms": float(ttft),
            "p99_itl_ms": float(itl),
        },
    }

    req = {
        "schema_version": 1,
        "task_id": task_id or str(uuid4()),
        "mode": "all",
        "model": {
            "hf_repo": str(model["hf_repo"]),
            "hf_revision": str(model["hf_revision"]),
            "dtype": dtype,
            "quantization": model.get("quantization"),
            "max_model_len": max_model_len,
        },
        "hardware": {
            "gpu_count": gpu_count,
            "gpu_sku_expected": sku,
        },
        "engines": {
            "baseline": {
                "image": str(baseline),
                "serve_args": list(serve_args),
                "env": {},
            },
            "candidate": {
                "image": str(candidate),
                "serve_args": list(serve_args),
                "env": {},
            },
        },
        "workload_trace": {
            "path": trace_path,
            "sha256": str(row["workload_trace_sha256"]),
        },
        "correctness": correctness,
        "perf_screen": perf_screen,
        "sla_bench": sla_bench,
        "hf_token_env": "HF_TOKEN",
    }
    try:
        validate_bench_request_dict(req)
    except RequestValidationError as exc:
        raise BenchInfraError("bench_request_invalid", str(exc)) from exc
    return req


def bind_report_to_run(
    report: dict[str, Any],
    *,
    request_task_id: str,
    executed_request_bytes: bytes,
    baseline_digest: str,
    candidate_digest: str,
    trace_sha256: str,
) -> None:
    """Cross-check report against executed request. Raises BenchInfraError on mismatch."""
    try:
        validate_report_dict(report)
    except RequestValidationError as exc:
        raise BenchInfraError("bench_report_invalid", str(exc)) from exc
    if str(report.get("task_id")) != request_task_id:
        raise BenchInfraError("bench_report_task_id_mismatch")
    fp = report.get("inputs_fingerprint") or {}
    expect_req = sha256_bytes(executed_request_bytes)
    if str(fp.get("request_sha256", "")).lower() != expect_req.lower():
        raise BenchInfraError("bench_report_request_sha256_mismatch")
    try:
        exp_base = extract_image_digest(baseline_digest)
        exp_cand = extract_image_digest(candidate_digest)
    except RequestValidationError as exc:
        raise BenchInfraError("bench_digest_parse", str(exc)) from exc
    if str(fp.get("baseline_image_digest", "")).lower() != exp_base.lower():
        raise BenchInfraError("bench_report_baseline_digest_mismatch")
    if str(fp.get("candidate_image_digest", "")).lower() != exp_cand.lower():
        raise BenchInfraError("bench_report_candidate_digest_mismatch")
    if str(fp.get("trace_sha256", "")).lower() != trace_sha256.lower():
        raise BenchInfraError("bench_report_trace_sha256_mismatch")


def _module_rows(
    report: dict[str, Any],
    *,
    task_id: str,
    gpu_sku: str,
    evidence_s3_url: str | None,
    mock: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in ("correctness", "perf_screen", "sla_bench"):
        sub = report.get(stage)
        if not isinstance(sub, dict):
            continue
        rows.append(
            {
                "task_id": task_id,
                "stage": stage,
                "verdict": str(sub.get("verdict") or report.get("verdict") or "error"),
                "report": sub,
                "evidence_s3_url": evidence_s3_url,
                "gpu_sku": gpu_sku,
                "mock": mock,
            }
        )
    if not rows and report.get("verdict") == "error":
        rows.append(
            {
                "task_id": task_id,
                "stage": "correctness",
                "verdict": "error",
                "report": report,
                "evidence_s3_url": evidence_s3_url,
                "gpu_sku": gpu_sku,
                "mock": mock,
            }
        )
    return rows


def _event_detail_base(
    *,
    task_id: str,
    report_sha256: str,
    evidence_s3_url: str | None,
    evidence_sha256: str | None,
    evidence_size: int | None,
    mock: bool,
    inputs_fingerprint: dict[str, Any],
    provider: str | None = None,
    pod_name: str | None = None,
    speedup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "task_id": task_id,
        "report_sha256": report_sha256,
        "evidence_s3_url": evidence_s3_url,
        "evidence_sha256": evidence_sha256,
        "evidence_size": evidence_size,
        "inputs_fingerprint": inputs_fingerprint,
        "mock": mock,
    }
    if provider:
        detail["provider"] = provider
    if pod_name:
        detail["pod_name"] = pod_name
    if speedup is not None:
        detail["speedup"] = speedup
    return detail


def _events_for_verdict(
    verdict: str,
    *,
    base: dict[str, Any],
    error_role: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    if verdict == "pass":
        return [
            (SubmissionState.CORRECT, dict(base)),
            (SubmissionState.SCREENED, dict(base)),
            (SubmissionState.BENCHED, dict(base)),
        ]
    if verdict == "fail_correctness":
        return [
            (
                SubmissionState.REJECTED,
                {**base, "reason": "fail_correctness"},
            )
        ]
    if verdict == "fail_perf_screen":
        return [
            (SubmissionState.CORRECT, dict(base)),
            (
                SubmissionState.REJECTED,
                {**base, "reason": "fail_perf_screen"},
            ),
        ]
    if verdict == "fail_sla":
        return [
            (SubmissionState.CORRECT, dict(base)),
            (SubmissionState.SCREENED, dict(base)),
            (SubmissionState.REJECTED, {**base, "reason": "fail_sla"}),
        ]
    if verdict == "fail_cross_env_speedup":
        return [
            (SubmissionState.CORRECT, dict(base)),
            (SubmissionState.SCREENED, dict(base)),
            (
                SubmissionState.REJECTED,
                {**base, "reason": "fail_cross_env_speedup"},
            ),
        ]
    if verdict == "fail_promotion":
        return [
            (SubmissionState.CORRECT, dict(base)),
            (SubmissionState.SCREENED, dict(base)),
            (
                SubmissionState.REJECTED,
                {**base, "reason": "fail_promotion"},
            ),
        ]
    if verdict == "error":
        if error_role == "candidate":
            return [
                (
                    SubmissionState.REJECTED,
                    {
                        **base,
                        "reason": "fail_engine_candidate",
                        "error_role": error_role,
                    },
                )
            ]
        raise BenchInfraError(
            "bench_engine_baseline_or_unknown", error_role or "absent"
        )
    raise BenchInfraError("bench_verdict_unknown", verdict)


def _fail_job(row: dict[str, Any], code: str) -> str:
    job_id = int(row["job_id"])
    set_job_status(
        row["id"],
        "failed",
        job_id=job_id,
        kind="bench",
        last_error=code,
    )
    return code


def parse_cross_env(bench: dict[str, Any]) -> dict[str, Any] | None:
    """Return normalized cross_env or None when absent (no speedup floor)."""
    raw = bench.get("cross_env")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise BenchInfraError("cross_env_invalid", "must be an object")
    try:
        return validate_cross_env(raw)
    except ValueError as exc:
        raise BenchInfraError("cross_env_invalid", str(exc)) from exc


def sku_speedup_value(report: dict[str, Any], metric: str) -> float | None:
    """Scalar from sla_bench.speedup[metric]; None if missing/NaN/non-numeric."""
    if metric not in SPEEDUP_METRIC_KEYS:
        return None
    speedup = (report.get("sla_bench") or {}).get("speedup")
    if not isinstance(speedup, dict) or metric not in speedup:
        return None
    try:
        val = float(speedup[metric])
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val


def worst_verdict(verdicts: list[str]) -> str:
    """Return worst harness verdict by fail_correctness > fail_perf_screen > fail_sla > pass."""
    best_rank = _VERDICT_RANK["pass"]
    winner = "pass"
    for v in verdicts:
        rank = _VERDICT_RANK.get(v)
        if rank is None:
            continue
        if rank < best_rank:
            best_rank = rank
            winner = v
    return winner


def aggregate_sku_outcomes(
    results: list[SkuRunResult],
    *,
    cross_env: dict[str, Any] | None,
) -> tuple[str, str | None]:
    """Aggregate per-SKU outcomes -> (verdict, error_role).

    Candidate engine errors map to verdict ``error`` with role candidate.
    Baseline/absent error_role raises BenchInfraError (infra).
    When all harness verdicts pass and cross_env is set, apply speedup floor.
    """
    error_roles = [r.error_role for r in results if r.verdict == "error"]
    if error_roles:
        if any(role == "candidate" for role in error_roles):
            return "error", "candidate"
        raise BenchInfraError(
            "bench_engine_baseline_or_unknown",
            next((r for r in error_roles if r), "absent") or "absent",
        )

    harness = [r.verdict for r in results if r.verdict != "error"]
    agg = worst_verdict(harness) if harness else "pass"
    if agg != "pass" or cross_env is None:
        return agg, None

    metric = str(cross_env["speedup_metric"])
    floor = float(cross_env["min_speedup_each"])
    for r in results:
        val = sku_speedup_value(r.report, metric)
        # Missing, NaN, or zero-baseline ratio (0.0) counts as below floor.
        if val is None or val < floor:
            return "fail_cross_env_speedup", None
    return "pass", None


@dataclass
class SkuRunResult:
    gpu_sku: str
    task_id: str
    exit_code: int
    report: dict[str, Any]
    report_sha256: str
    evidence_s3_url: str | None = None
    evidence_sha256: str | None = None
    evidence_size: int | None = None
    provider: str | None = None
    pod_name: str | None = None
    verdict: str = "error"
    error_role: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)


def _env_entry(r: SkuRunResult) -> dict[str, Any]:
    return {
        "gpu_sku": r.gpu_sku,
        "task_id": r.task_id,
        "verdict": r.verdict,
        "speedup": (r.report.get("sla_bench") or {}).get("speedup"),
        "report_sha256": r.report_sha256,
        "evidence_s3_url": r.evidence_s3_url,
        "evidence_sha256": r.evidence_sha256,
        "evidence_size": r.evidence_size,
    }


def _event_base_for_results(
    results: list[SkuRunResult],
    *,
    mock: bool,
    cross_env: dict[str, Any] | None,
) -> dict[str, Any]:
    multi = len(results) > 1
    if not multi:
        r = results[0]
        return _event_detail_base(
            task_id=r.task_id,
            report_sha256=r.report_sha256,
            evidence_s3_url=r.evidence_s3_url,
            evidence_sha256=r.evidence_sha256,
            evidence_size=r.evidence_size,
            mock=mock,
            inputs_fingerprint=dict(r.report.get("inputs_fingerprint") or {}),
            provider=r.provider,
            pod_name=r.pod_name,
            speedup=(r.report.get("sla_bench") or {}).get("speedup"),
        )

    detail: dict[str, Any] = {
        "environments": [_env_entry(r) for r in results],
        "mock": mock,
    }
    providers = {r.provider for r in results if r.provider}
    if len(providers) == 1:
        detail["provider"] = next(iter(providers))
    if cross_env is not None:
        metric = str(cross_env["speedup_metric"])
        values = [sku_speedup_value(r.report, metric) for r in results]
        present = [v for v in values if v is not None]
        if present:
            detail["effective_speedup"] = min(present)
            detail["speedup_metric"] = metric
    return detail


def campaign_uses_dynamic_sampling(row: dict[str, Any]) -> bool:
    """True when campaign pins an hf_rows sampling_rule."""
    rule = _parse_json_field(row.get("sampling_rule"))
    return isinstance(rule, dict) and str(rule.get("type") or "") == "hf_rows"


def realize_submission_sample(
    row: dict[str, Any],
    *,
    block_hash_fn: Callable[[int], str] | None = None,
    record_sample_fn: Callable[..., None] | None = None,
    row_fetcher: Callable[[int], dict[str, Any]] | None = None,
    upload_trace_fn: Callable[..., str] | None = None,
) -> dict[str, str]:
    """Generate a trace from the future-block seed; persist receipt; return url/sha.

    Raises BenchInfraError when the seed block hash is unavailable (job fail,
    no rejection event). HF fetch failures are infra, never candidate FAIL.
    """
    try:
        rule = parse_sampling_rule(_parse_json_field(row.get("sampling_rule")))
    except SamplerError as exc:
        raise BenchInfraError("sampling_config_invalid", str(exc)) from exc

    commit_block = row.get("commit_block")
    if commit_block is None:
        raise BenchInfraError("commit_block_missing")
    seed_block = int(commit_block) + int(rule["seed_block_offset"])

    if block_hash_fn is None:
        from chain.rpc import ChainError, fetch_finalized_block_hash

        import bittensor as bt

        def _default_hash(block_number: int) -> str:
            sub = bt.Subtensor(network=config.SUBTENSOR_NETWORK)
            try:
                return fetch_finalized_block_hash(
                    sub,
                    block_number,
                    finality_depth=config.CHAIN_FINALITY_DEPTH,
                    attempts=config.CHAIN_RETRY_ATTEMPTS,
                    delay_s=float(config.CHAIN_RETRY_DELAY_S),
                )
            except ChainError as exc:
                raise BenchInfraError("sample_seed_block_unavailable", str(exc)) from exc

        block_hash_fn = _default_hash

    try:
        block_hash = block_hash_fn(seed_block)
    except BenchInfraError:
        raise
    except Exception as exc:
        raise BenchInfraError("sample_seed_block_unavailable", str(exc)) from exc
    if not block_hash:
        raise BenchInfraError("sample_seed_block_unavailable", "empty hash")

    try:
        sampled = sample_workload(
            rule=rule,
            commit_block=int(commit_block),
            block_hash=str(block_hash),
            patch_hash=str(row["patch_hash"]),
            campaign_id=str(row["campaign_id"]),
            row_fetcher=row_fetcher,
        )
    except SamplerError as exc:
        raise BenchInfraError("sampling_failed", str(exc)) from exc

    if upload_trace_fn is None:
        from storage.s3 import upload_realized_trace

        def _default_upload(*, campaign_id: str, body: bytes, sha256: str) -> str:
            try:
                return upload_realized_trace(
                    campaign_id=campaign_id, body=body, sha256=sha256
                )
            except Exception as exc:
                raise BenchInfraError("realized_trace_upload_failed", str(exc)) from exc

        upload_trace_fn = _default_upload

    try:
        url = upload_trace_fn(
            campaign_id=str(row["campaign_id"]),
            body=sampled.body,
            sha256=sampled.sha256,
        )
    except BenchInfraError:
        raise
    except Exception as exc:
        raise BenchInfraError("realized_trace_upload_failed", str(exc)) from exc

    receipt = dict(sampled.receipt)
    receipt["sampled_trace_url"] = url
    receipt["patch_hash"] = str(row["patch_hash"]).strip().lower()
    receipt["campaign_id"] = str(row["campaign_id"]).strip().lower()

    writer = record_sample_fn or record_submission_sample
    writer(
        submission_id=str(row["id"]),
        sample_seed_block=sampled.sample_seed_block,
        sample_seed_block_hash=sampled.sample_seed_block_hash,
        sampled_trace_sha256=sampled.sha256,
        sampling_receipt=receipt,
    )
    row["workload_trace_url"] = url
    row["workload_trace_sha256"] = sampled.sha256
    row["sampled_trace_sha256"] = sampled.sha256
    row["sample_seed_block"] = sampled.sample_seed_block
    row["sample_seed_block_hash"] = sampled.sample_seed_block_hash
    row["sampling_receipt"] = receipt
    return {"url": url, "sha256": sampled.sha256}


def process_bench_job(
    row: dict[str, Any],
    *,
    mock_bench: bool,
    mock_tampered_candidate: bool = False,
    mock_baseline_token_latency_s: float = 0.0,
    mock_candidate_token_latency_s: float = 0.0,
    work_root: Path | None = None,
    run_bench_fn: Callable[..., int] | None = None,
    run_pod_fn: Callable[..., int] | None = None,
    upload_evidence_fn: Callable[..., tuple[str, str, int]] | None = None,
    trace_fetcher: Callable[[str], bytes] | None = None,
    injected_exit: int | None = None,
    injected_report: dict[str, Any] | None = None,
    block_hash_fn: Callable[[int], str] | None = None,
    record_sample_fn: Callable[..., None] | None = None,
    row_fetcher: Callable[[int], dict[str, Any]] | None = None,
    upload_trace_fn: Callable[..., str] | None = None,
) -> str:
    """Run one bench job across all gpu_skus. Returns 'ok' or last_error code."""
    submission_id = str(row["id"])
    job_id = int(row["job_id"])
    root = Path(
        work_root or (config.WORK_DIR / f"bench-{submission_id}-{uuid4().hex[:8]}")
    )
    root.mkdir(parents=True, exist_ok=True)
    bench_timer = Timer()
    patch_sha256 = str(row.get("patch_hash", ""))
    image_digest = str(row.get("engine_image_ref", ""))

    obs.bench_started(
        submission_id=submission_id,
        job_id=str(job_id),
        stage="bench",
        image_digest=image_digest,
        patch_sha256=patch_sha256,
    )

    try:
        bench_timer.__enter__()
        bench = _parse_json_field(row.get("bench"))
        if not isinstance(bench, dict):
            raise BenchInfraError("campaign_bench_missing")
        cross_env = parse_cross_env(bench)

        if campaign_uses_dynamic_sampling(row):
            realize_submission_sample(
                row,
                block_hash_fn=block_hash_fn,
                record_sample_fn=record_sample_fn,
                row_fetcher=row_fetcher,
                upload_trace_fn=upload_trace_fn,
            )

        trace_path = materialize_trace(
            url=str(row["workload_trace_url"]),
            expected_sha256=str(row["workload_trace_sha256"]),
            dest_dir=root / "trace",
            fetcher=trace_fetcher,
        )
        gpu_skus = [str(s) for s in (_parse_json_field(row.get("gpu_skus")) or [])]
        if not gpu_skus:
            raise BenchInfraError("gpu_skus_empty")

        # injected_* stays single-SKU (first entry) for existing unit tests.
        skus_to_run = gpu_skus[:1] if injected_exit is not None else gpu_skus
        results: list[SkuRunResult] = []

        for sku in skus_to_run:
            result = _run_one_sku(
                row,
                bench=bench,
                gpu_sku=sku,
                trace_path=trace_path,
                root=root,
                mock_bench=mock_bench,
                mock_tampered_candidate=mock_tampered_candidate,
                mock_baseline_token_latency_s=mock_baseline_token_latency_s,
                mock_candidate_token_latency_s=mock_candidate_token_latency_s,
                run_bench_fn=run_bench_fn,
                run_pod_fn=run_pod_fn,
                upload_evidence_fn=upload_evidence_fn,
                injected_exit=injected_exit,
                injected_report=injected_report,
            )
            results.append(result)

            if result.exit_code == 75:
                logger.error(
                    "ALERT bench_destroy_failed submission=%s task_id=%s sku=%s "
                    "(pod/volume may still be billing); partial rows persisted, "
                    "no verdict events",
                    submission_id,
                    result.task_id,
                    sku,
                )
                all_rows: list[dict[str, Any]] = []
                for r in results:
                    all_rows.extend(r.rows)
                # Multi-SKU: forensics rows only, no events. Single-SKU WS-D
                # kept complete-run events when destroy fails after a bound report.
                if len(gpu_skus) == 1:
                    base = _event_base_for_results(
                        results, mock=mock_bench, cross_env=cross_env
                    )
                    if result.verdict == "error":
                        events = _events_for_verdict(
                            "error", base=base, error_role=result.error_role
                        )
                    else:
                        events = _events_for_verdict(result.verdict, base=base)
                else:
                    events = []
                finalize_bench_job(
                    submission_id=submission_id,
                    job_id=job_id,
                    task_id=results[0].task_id,
                    report_rows=all_rows,
                    events=events,
                    job_status="failed",
                    last_error="bench_destroy_failed",
                )
                bench_timer.__exit__(None, None, None)
                obs.bench_failed(
                    submission_id=submission_id,
                    job_id=str(job_id),
                    stage="bench",
                    image_digest=image_digest,
                    patch_sha256=patch_sha256,
                    duration_s=bench_timer.elapsed_s,
                    error="bench_destroy_failed",
                )
                obs.job_failed(
                    submission_id=submission_id,
                    job_id=str(job_id),
                    stage="bench",
                    error="bench_destroy_failed",
                )
                return "bench_destroy_failed"

        # Aggregate verdict across completed SKUs.
        try:
            agg_verdict, error_role = aggregate_sku_outcomes(
                results, cross_env=cross_env
            )
        except BenchInfraError:
            raise

        base = _event_base_for_results(results, mock=mock_bench, cross_env=cross_env)
        if agg_verdict == "pass" and cross_env is not None:
            metric = str(cross_env["speedup_metric"])
            values = [
                v
                for v in (sku_speedup_value(r.report, metric) for r in results)
                if v is not None
            ]
            if values:
                base["effective_speedup"] = min(values)
                base["speedup_metric"] = metric

        all_rows: list[dict[str, Any]] = []
        for r in results:
            all_rows.extend(r.rows)

        # Z-score promotion (optional): Module A / engine hard fails stay hard.
        # When stages pass and campaign has calibration + z_threshold, promote
        # only if aggregate_z < threshold ("not anomalously worse").
        calibration = _parse_json_field(row.get("calibration"))
        z_threshold = row.get("z_threshold")
        if (
            agg_verdict == "pass"
            and isinstance(calibration, dict)
            and z_threshold is not None
        ):
            try:
                promo = decide_promotion(
                    report=results[0].report,
                    calibration=calibration,
                    z_threshold=float(z_threshold),
                    hard_error=False,
                )
            except PromoteError as exc:
                raise BenchInfraError("promotion_invalid", str(exc)) from exc
            all_rows.append(
                {
                    "task_id": results[0].task_id,
                    "stage": "promotion",
                    "verdict": promo.report["verdict"],
                    "report": promo.report,
                    "evidence_s3_url": results[0].evidence_s3_url,
                    "gpu_sku": results[0].gpu_sku,
                    "mock": mock_bench,
                    "z_scores": promo.z_scores,
                    "aggregate_z": promo.aggregate_z,
                    "promoted": promo.promoted,
                }
            )
            base = {
                **base,
                "aggregate_z": promo.aggregate_z,
                "z_scores": promo.z_scores,
                "promoted": promo.promoted,
            }
            if not promo.promoted:
                agg_verdict = "fail_promotion"

        if agg_verdict == "error":
            events = _events_for_verdict("error", base=base, error_role=error_role)
        else:
            events = _events_for_verdict(agg_verdict, base=base)

        finalize_bench_job(
            submission_id=submission_id,
            job_id=job_id,
            task_id=results[0].task_id,
            report_rows=all_rows,
            events=events,
            job_status="done",
            last_error=None,
        )
        bench_timer.__exit__(None, None, None)
        primary = results[0]
        obs.bench_completed(
            submission_id=submission_id,
            job_id=str(job_id),
            stage="bench",
            provider=primary.provider or "",
            image_digest=image_digest,
            patch_sha256=patch_sha256,
            duration_s=bench_timer.elapsed_s,
            verdict=agg_verdict,
            evidence_s3_url=primary.evidence_s3_url or "",
            evidence_sha256=primary.evidence_sha256 or "",
        )
        return "ok"

    except BenchInfraError as exc:
        bench_timer.__exit__(None, None, None)
        logger.warning("bench infra fail submission=%s: %s", submission_id, exc)
        obs.bench_failed(
            submission_id=submission_id,
            job_id=str(job_id),
            stage="bench",
            image_digest=image_digest,
            patch_sha256=patch_sha256,
            duration_s=bench_timer.elapsed_s,
            error=exc.code,
        )
        obs.job_failed(
            submission_id=submission_id,
            job_id=str(job_id),
            stage="bench",
            error=exc.code,
        )
        return _fail_job(row, exc.code)
    except Exception as exc:
        bench_timer.__exit__(None, None, None)
        logger.exception("bench unexpected failure")
        error_code = f"bench_unexpected:{type(exc).__name__}"
        obs.bench_failed(
            submission_id=submission_id,
            job_id=str(job_id),
            stage="bench",
            image_digest=image_digest,
            patch_sha256=patch_sha256,
            duration_s=bench_timer.elapsed_s,
            error=error_code,
        )
        obs.job_failed(
            submission_id=submission_id,
            job_id=str(job_id),
            stage="bench",
            error=error_code,
        )
        return _fail_job(row, error_code)


def _run_one_sku(
    row: dict[str, Any],
    *,
    bench: dict[str, Any],
    gpu_sku: str,
    trace_path: Path,
    root: Path,
    mock_bench: bool,
    mock_tampered_candidate: bool,
    mock_baseline_token_latency_s: float,
    mock_candidate_token_latency_s: float,
    run_bench_fn: Callable[..., int] | None,
    run_pod_fn: Callable[..., int] | None,
    upload_evidence_fn: Callable[..., tuple[str, str, int]] | None,
    injected_exit: int | None,
    injected_report: dict[str, Any] | None,
) -> SkuRunResult:
    """Provision/bench/bind one SKU. Raises BenchInfraError on infra failure."""
    submission_id = str(row["id"])
    task_id = str(uuid4())
    sku_dir = root / f"sku-{gpu_sku.replace('/', '_')}"
    output_dir = sku_dir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    req = build_bench_request_dict(
        row,
        task_id=task_id,
        trace_path=str(trace_path.resolve()),
        gpu_sku=gpu_sku,
        require_calibration=not mock_bench,
    )
    request_path = sku_dir / "bench_request.json"
    request_bytes = (json.dumps(req, indent=2) + "\n").encode("utf-8")
    request_path.write_bytes(request_bytes)

    provider: str | None = None
    pod_name: str | None = None
    exit_code: int

    if injected_exit is not None:
        exit_code = injected_exit
        if injected_report is not None:
            patched = dict(injected_report)
            # Preserve injected task_id (bind-mismatch tests); default to request.
            if not patched.get("task_id"):
                patched["task_id"] = task_id
            fp = dict(patched.get("inputs_fingerprint") or {})
            fp["request_sha256"] = sha256_bytes(request_bytes)
            patched["inputs_fingerprint"] = fp
            (output_dir / "bench_report.json").write_text(
                json.dumps(patched, indent=2) + "\n",
                encoding="utf-8",
            )
            (output_dir / "bench_request.remote.json").write_bytes(request_bytes)
    elif mock_bench:
        from bench.main import run_bench

        fn = run_bench_fn or run_bench
        exit_code = fn(
            request_path,
            output_dir,
            mock_engine=True,
            mock_tampered_candidate=mock_tampered_candidate,
            mock_baseline_token_latency_s=mock_baseline_token_latency_s,
            mock_candidate_token_latency_s=mock_candidate_token_latency_s,
        )
    else:
        from gpu.orchestrate import run_bench_on_pod
        from gpu.types import PodSpec

        spec = PodSpec(
            provider=config.GPU_PROVIDER,
            gpu_count=int(bench.get("gpu_count") or 1),
            gpu_type=gpu_sku,
            ttl_hours=config.GPU_TTL_HOURS,
            max_hourly_cents=config.GPU_MAX_HOURLY_CENTS,
        )
        fn = run_pod_fn or run_bench_on_pod
        try:
            exit_code = fn(
                spec,
                request_path=request_path,
                output_dir=output_dir,
            )
        except Exception as exc:
            logger.exception("gpu orchestrate failed sku=%s", gpu_sku)
            raise BenchInfraError("bench_gpu_exception", type(exc).__name__) from exc
        provider = spec.provider

    report_path = output_dir / "bench_report.json"
    destroy_failed = exit_code == 75

    if exit_code in (1, 2) and not report_path.is_file():
        raise BenchInfraError(
            "bench_exit_bad_request" if exit_code == 1 else "bench_exit_env"
        )

    if not report_path.is_file():
        if destroy_failed:
            raise BenchInfraError("bench_destroy_failed")
        if exit_code == 3:
            raise BenchInfraError("bench_engine_no_report")
        raise BenchInfraError(f"bench_exit_{exit_code}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mock_bench or injected_exit is not None:
        executed_bytes = request_bytes
    else:
        remote_req = output_dir / "bench_request.remote.json"
        if not remote_req.is_file():
            raise BenchInfraError("bench_remote_request_missing")
        executed_bytes = remote_req.read_bytes()

    bind_report_to_run(
        report,
        request_task_id=task_id,
        executed_request_bytes=executed_bytes,
        baseline_digest=str(bench["baseline_engine_image_digest"]),
        candidate_digest=str(row["engine_image_ref"]),
        trace_sha256=str(row["workload_trace_sha256"]),
    )

    if exit_code not in (0, 3, 75):
        raise BenchInfraError(f"bench_exit_{exit_code}")

    report_raw = report_path.read_bytes()
    report_sha = sha256_bytes(report_raw)
    evidence_url = None
    evidence_sha = None
    evidence_size = None
    if not mock_bench and injected_exit is None:
        from storage.s3 import upload_evidence_bundle

        up = upload_evidence_fn or upload_evidence_bundle
        evidence_url, evidence_sha, evidence_size = up(
            submission_id, task_id, output_dir
        )

    verdict = str(report.get("verdict") or "error")
    error_role = report.get("error_role")
    error_role_s = error_role if isinstance(error_role, str) else None
    if exit_code == 3:
        verdict = "error"

    rows = _module_rows(
        report,
        task_id=task_id,
        gpu_sku=gpu_sku,
        evidence_s3_url=evidence_url,
        mock=mock_bench,
    )
    return SkuRunResult(
        gpu_sku=gpu_sku,
        task_id=task_id,
        exit_code=exit_code,
        report=report,
        report_sha256=report_sha,
        evidence_s3_url=evidence_url,
        evidence_sha256=evidence_sha,
        evidence_size=evidence_size,
        provider=provider,
        pod_name=pod_name,
        verdict=verdict,
        error_role=error_role_s,
        rows=rows,
    )


def tar_gz_directory(src_dir: Path, dest_archive: Path) -> tuple[str, int]:
    """Create tar.gz of src_dir; return (sha256, size)."""
    dest_archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest_archive, "w:gz") as tar:
        tar.add(src_dir, arcname=".")
    data = dest_archive.read_bytes()
    return f"sha256:{hashlib.sha256(data).hexdigest()}", len(data)
