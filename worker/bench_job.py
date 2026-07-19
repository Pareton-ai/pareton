"""Build bench_request.json and run Stages 1–3 for a claimed bench job."""

from __future__ import annotations

import hashlib
import json
import logging
import tarfile
import urllib.error
import urllib.request
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
from campaign.store import finalize_bench_job, set_job_status
from gate.types import SubmissionState

logger = logging.getLogger(__name__)

REASON_CANDIDATE_NOT_PINNED = "candidate_image_not_digest_pinned"


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


def build_bench_request_dict(
    row: dict[str, Any],
    *,
    task_id: str | None = None,
    trace_path: str,
    fetcher: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Operator-pinned bench_request from campaign + submission (image only from miner)."""
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

    baseline = bench.get("baseline_engine_image_digest")
    if not baseline or not is_digest_pinned_image(str(baseline)):
        raise BenchInfraError("baseline_image_not_digest_pinned", str(baseline))

    gpu_skus = _parse_json_field(row.get("gpu_skus")) or []
    if not gpu_skus:
        raise BenchInfraError("gpu_skus_empty")
    gpu_sku = str(gpu_skus[0])
    gpu_count = int(bench.get("gpu_count") or 1)

    max_model_len = int(model["max_model_len"])
    extra_serve = list(bench.get("serve_args") or [])
    serve_args = [
        "--model",
        "/model",
        "--max-model-len",
        str(max_model_len),
        *[str(x) for x in extra_serve],
    ]

    corr_cfg = dict(bench.get("correctness") or {})
    thresholds = dict(corr_cfg.get("thresholds") or {})
    correctness = {
        "num_prompts": int(
            corr_cfg.get("num_prompts", config.BENCH_CORRECTNESS_NUM_PROMPTS)
        ),
        "max_new_tokens": int(
            corr_cfg.get("max_new_tokens", config.BENCH_CORRECTNESS_MAX_NEW_TOKENS)
        ),
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
            "dtype": str(model.get("dtype") or "bfloat16"),
            "quantization": model.get("quantization"),
            "max_model_len": max_model_len,
        },
        "hardware": {
            "gpu_count": gpu_count,
            "gpu_sku_expected": gpu_sku,
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
) -> str:
    """Run one bench job. Returns 'ok' or last_error code."""
    submission_id = str(row["id"])
    job_id = int(row["job_id"])
    root = Path(
        work_root or (config.WORK_DIR / f"bench-{submission_id}-{uuid4().hex[:8]}")
    )
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        bench = _parse_json_field(row.get("bench"))
        if not isinstance(bench, dict):
            raise BenchInfraError("campaign_bench_missing")

        trace_path = materialize_trace(
            url=str(row["workload_trace_url"]),
            expected_sha256=str(row["workload_trace_sha256"]),
            dest_dir=root / "trace",
            fetcher=trace_fetcher,
        )
        task_id = str(uuid4())
        req = build_bench_request_dict(
            row,
            task_id=task_id,
            trace_path=str(trace_path.resolve()),
        )
        request_path = root / "bench_request.json"
        request_bytes = (json.dumps(req, indent=2) + "\n").encode("utf-8")
        request_path.write_bytes(request_bytes)

        gpu_skus = _parse_json_field(row.get("gpu_skus")) or ["unknown"]
        gpu_sku = "mock" if mock_bench else str(gpu_skus[0])
        provider = None
        pod_name = None
        exit_code: int

        if injected_exit is not None:
            exit_code = injected_exit
            if injected_report is not None:
                # Tests inject outcomes; only request_sha256 is path-dependent on
                # materialize location. Other bind fields stay as injected.
                patched = dict(injected_report)
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
            from gpu.orchestrate import EXIT_DESTROY_FAILED, run_bench_on_pod
            from gpu.types import PodSpec

            spec = PodSpec(
                provider=config.GPU_PROVIDER,
                gpu_count=int(bench.get("gpu_count") or 1),
                gpu_type=str(gpu_skus[0]),
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
                logger.exception("gpu orchestrate failed")
                raise BenchInfraError(
                    "bench_gpu_exception", type(exc).__name__
                ) from exc
            provider = spec.provider
            pod_name = None
            # EXIT_DESTROY_FAILED handled below with report binding
            _ = EXIT_DESTROY_FAILED

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
        if mock_bench:
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

        report_raw = report_path.read_bytes()
        report_sha = sha256_bytes(report_raw)
        evidence_url = None
        evidence_sha = None
        evidence_size = None
        if not mock_bench:
            from storage.s3 import upload_evidence_bundle

            up = upload_evidence_fn or upload_evidence_bundle
            evidence_url, evidence_sha, evidence_size = up(
                submission_id, task_id, output_dir
            )

        base = _event_detail_base(
            task_id=task_id,
            report_sha256=report_sha,
            evidence_s3_url=evidence_url,
            evidence_sha256=evidence_sha,
            evidence_size=evidence_size,
            mock=mock_bench,
            inputs_fingerprint=dict(report.get("inputs_fingerprint") or {}),
            provider=provider,
            pod_name=pod_name,
            speedup=(report.get("sla_bench") or {}).get("speedup"),
        )
        verdict = str(report.get("verdict") or "error")
        error_role = report.get("error_role")
        if isinstance(error_role, str):
            error_role_s: str | None = error_role
        else:
            error_role_s = None

        if exit_code == 3 or verdict == "error":
            events = _events_for_verdict("error", base=base, error_role=error_role_s)
        elif exit_code in (0, 75):
            events = _events_for_verdict(verdict, base=base)
        else:
            raise BenchInfraError(f"bench_exit_{exit_code}")

        rows = _module_rows(
            report,
            task_id=task_id,
            gpu_sku=gpu_sku,
            evidence_s3_url=evidence_url,
            mock=mock_bench,
        )
        job_status = "failed" if destroy_failed else "done"
        last_error = "bench_destroy_failed" if destroy_failed else None
        if destroy_failed:
            logger.error(
                "ALERT bench_destroy_failed submission=%s task_id=%s "
                "(pod/volume may still be billing); report persisted",
                submission_id,
                task_id,
            )
        finalize_bench_job(
            submission_id=submission_id,
            job_id=job_id,
            task_id=task_id,
            report_rows=rows,
            events=events,
            job_status=job_status,
            last_error=last_error,
        )
        return last_error or "ok"

    except BenchInfraError as exc:
        logger.warning("bench infra fail submission=%s: %s", submission_id, exc)
        return _fail_job(row, exc.code)
    except Exception as exc:
        logger.exception("bench unexpected failure")
        return _fail_job(row, f"bench_unexpected:{type(exc).__name__}")


def tar_gz_directory(src_dir: Path, dest_archive: Path) -> tuple[str, int]:
    """Create tar.gz of src_dir; return (sha256, size)."""
    dest_archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest_archive, "w:gz") as tar:
        tar.add(src_dir, arcname=".")
    data = dest_archive.read_bytes()
    return f"sha256:{hashlib.sha256(data).hexdigest()}", len(data)
