"""Run one claimed round: one pod, one trace, every image, one leader decision."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import config
from bench.main import MockCandidatePlan, MockPlan, run_bench
from bench.sampler import (
    SamplerError,
    fetch_hf_row,
    generate_trace,
    parse_sampling_rule,
)
from bench.validate import (
    RequestValidationError,
    extract_image_digest,
    is_digest_pinned_image,
    sha256_bytes,
    validate_bench_request_dict,
    validate_report_dict,
)
from builder.digest import image_ref_resolves
from builder.registry import baseline_engine_image_ref
from campaign.engine import resolve_engine
from campaign.store import get_campaign
from gpu.errors import GpuError, NoCapacityError, ProvisionError
from gpu.orchestrate import EXIT_DESTROY_FAILED, run_bench_on_pod
from gpu.types import PodSpec
from observability import events as obs
from round.rank import Entry, rank_round
from round.store import (
    VOID_LEADER_IMAGE_MISSING,
    VOID_POD_FAILED,
    VOID_POD_PROVISION_FAILED,
    VOID_ROUND_TIMEOUT,
    VOID_TRACE_UNAVAILABLE,
    complete_round,
    get_leader,
    list_round_entries,
    set_round_phase,
    touch_round_heartbeat,
    defer_round_for_capacity,
    void_round,
)
from worker.phase_reporter import PhaseReporter

logger = logging.getLogger(__name__)


class RoundDeferred(Exception):
    """The GPU market was empty. Wait and re-claim; the round is not spent."""

    def __init__(self, delay_s: float, detail: str) -> None:
        super().__init__(detail)
        self.delay_s = delay_s
        self.detail = detail


def capacity_retry_delay_s(attempts: int) -> float:
    """Backoff before re-claiming a round no provider could fill.

    Doubles per prior attempt and saturates at PROVISION_RETRY_MAX_S. The
    exponent is clamped so a long outage cannot build an absurd shift value.
    """
    base = float(config.PROVISION_RETRY_BASE_S)
    cap = float(config.PROVISION_RETRY_MAX_S)
    return min(base * float(2 ** min(max(attempts, 0), 20)), cap)


class RoundInfraError(Exception):
    """Infrastructure failure: void the round, do not settle a leader."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _campaign_engine_profile(campaign: Any) -> dict[str, Any]:
    raw = campaign.engine if campaign is not None else None
    return resolve_engine(raw if isinstance(raw, dict) else None)


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
            raise RoundInfraError(VOID_TRACE_UNAVAILABLE, f">{limit} bytes")
        return data
    if url.startswith("file://"):
        path = Path(url[7:])
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise RoundInfraError(VOID_TRACE_UNAVAILABLE, str(exc)) from exc
        if len(data) > limit:
            raise RoundInfraError(VOID_TRACE_UNAVAILABLE, f">{limit} bytes")
        return data
    if url.startswith("https://") or url.startswith("http://"):
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read(limit + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RoundInfraError(VOID_TRACE_UNAVAILABLE, str(exc)) from exc
        if len(data) > limit:
            raise RoundInfraError(VOID_TRACE_UNAVAILABLE, f">{limit} bytes")
        return data
    raise RoundInfraError(VOID_TRACE_UNAVAILABLE, url)


def _write_trace(dest_dir: Path, raw: bytes, expected_sha256: str) -> Path:
    actual = sha256_bytes(raw)
    if actual.lower() != str(expected_sha256).lower():
        raise RoundInfraError(
            VOID_TRACE_UNAVAILABLE,
            f"expected {expected_sha256}, got {actual}",
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "workload_trace.json"
    path.write_bytes(raw)
    return path


def materialize_round_trace(
    round_row: dict[str, Any],
    campaign: Any,
    dest_dir: Path,
    *,
    fetcher: Callable[[str], bytes] | None = None,
    row_fetcher: Callable[[int], dict[str, Any]] | None = None,
) -> Path:
    """Rebuild the round's workload trace and verify it against the snapshot sha."""
    expected = str(round_row["sampled_trace_sha256"])
    receipt = _parse_json_field(round_row.get("sampling_receipt")) or {}
    rtype = str(receipt.get("type") or "")
    if rtype == "hf_rows":
        try:
            rule = parse_sampling_rule(
                {
                    "type": "hf_rows",
                    "dataset": receipt.get("dataset"),
                    "revision": receipt.get("revision"),
                    "config": receipt.get("config"),
                    "split": receipt.get("split"),
                    "n_rows": receipt.get("n_rows"),
                    "n_prompts": receipt.get("n_prompts"),
                    "max_tokens": receipt.get("max_tokens"),
                    "algo_version": receipt.get("algo_version"),
                    "seed_block_offset": receipt.get("seed_block_offset"),
                }
            )
            sampled = generate_trace(
                rule=rule,
                seed_hex=str(
                    receipt.get("seed_hex") or round_row.get("seed_hex") or ""
                ),
                row_fetcher=row_fetcher or (lambda idx: fetch_hf_row(rule, idx)),
                sample_seed_block=int(receipt.get("sample_seed_block") or 0),
                sample_seed_block_hash=str(receipt.get("sample_seed_block_hash") or ""),
            )
        except (SamplerError, TypeError, ValueError, KeyError) as exc:
            raise RoundInfraError(VOID_TRACE_UNAVAILABLE, str(exc)) from exc
        return _write_trace(dest_dir, sampled.body, expected)
    url = str(
        receipt.get("workload_trace_url")
        or getattr(campaign, "workload_trace_url", "")
        or ""
    )
    if not url:
        raise RoundInfraError(VOID_TRACE_UNAVAILABLE, "no workload_trace_url")
    raw = fetch_trace_bytes(url, fetcher=fetcher)
    return _write_trace(dest_dir, raw, expected)


def trace_request_count(trace_path: str | Path) -> int:
    path = Path(trace_path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoundInfraError(VOID_TRACE_UNAVAILABLE, f"{path}: {exc}") from exc
    requests = doc.get("requests") if isinstance(doc, dict) else None
    if not isinstance(requests, list) or not requests:
        raise RoundInfraError(VOID_TRACE_UNAVAILABLE, str(path))
    return len(requests)


def build_round_request(
    round_row: dict[str, Any],
    campaign: Any,
    entries: list[dict[str, Any]],
    *,
    task_id: str,
    trace_path: str,
) -> dict[str, Any]:
    """Operator-pinned bench_request for one round. Miner input is image refs only."""
    bench = campaign.bench if isinstance(getattr(campaign, "bench", None), dict) else {}
    if not bench:
        raise RoundInfraError(VOID_POD_FAILED, "campaign_bench_missing")
    model = bench.get("model")
    if not isinstance(model, dict):
        raise RoundInfraError(VOID_POD_FAILED, "campaign_bench_model_missing")

    baseline_entry = next((e for e in entries if e["role"] == "baseline"), None)
    if baseline_entry is None:
        raise RoundInfraError(VOID_POD_FAILED, "round_missing_baseline")
    candidates = [e for e in entries if e["role"] != "baseline"]
    if not candidates:
        raise RoundInfraError(VOID_POD_FAILED, "round_missing_candidates")

    baseline_raw = bench.get("baseline_engine_image_digest")
    if not baseline_raw or not is_digest_pinned_image(str(baseline_raw)):
        raise RoundInfraError(
            VOID_POD_FAILED, f"baseline_image_not_digest_pinned:{baseline_raw}"
        )
    try:
        baseline_image = str(baseline_entry["engine_image_ref"])
        if not is_digest_pinned_image(baseline_image):
            baseline_image = baseline_engine_image_ref(str(baseline_raw))
    except ValueError as exc:
        raise RoundInfraError(VOID_POD_FAILED, str(exc)) from exc

    for row in candidates:
        ref = str(row.get("engine_image_ref") or "")
        if not is_digest_pinned_image(ref):
            raise RoundInfraError(
                VOID_POD_FAILED, f"candidate_image_not_digest_pinned:{ref}"
            )

    max_model_len = int(model["max_model_len"])
    dtype = str(model.get("dtype") or "bfloat16")
    extra_serve = list(bench.get("serve_args") or [])
    engine_profile = _campaign_engine_profile(campaign)
    cache_dir = str(engine_profile["cache_dir"])
    serve_args = ["--model", "/model"]
    # SGLang rejects --max-model-len (it uses campaign --context-length).
    if engine_profile["name"] != "sglang":
        serve_args.extend(["--max-model-len", str(max_model_len)])
    serve_args.extend(["--dtype", dtype])
    quantization = model.get("quantization")
    if quantization is not None and str(quantization).strip() != "":
        serve_args.extend(["--quantization", str(quantization)])
    serve_args.extend(str(x) for x in extra_serve)

    sla = campaign.sla.to_dict() if getattr(campaign, "sla", None) is not None else {}
    ttft = sla.get("p99_ttft_ms")
    itl = sla.get("p99_itl_ms")
    if ttft is None or itl is None:
        raise RoundInfraError(VOID_POD_FAILED, "sla_thresholds_missing")

    corr_cfg = dict(bench.get("correctness") or {})
    thresholds = dict(corr_cfg.get("thresholds") or {})
    trace_n = trace_request_count(trace_path)
    correctness = {
        "num_prompts": int(corr_cfg.get("num_prompts", trace_n)),
        "thresholds": {
            "min_mean_logprob": float(
                thresholds.get(
                    "min_mean_logprob", config.BENCH_CORRECTNESS_MIN_MEAN_LOGPROB
                )
            ),
            "min_token_logprob": float(
                thresholds.get(
                    "min_token_logprob", config.BENCH_CORRECTNESS_MIN_TOKEN_LOGPROB
                )
            ),
            "min_token_quantile": float(
                thresholds.get(
                    "min_token_quantile",
                    config.BENCH_CORRECTNESS_MIN_TOKEN_QUANTILE,
                )
            ),
            "min_coverage_ratio": float(
                thresholds.get(
                    "min_coverage_ratio",
                    config.BENCH_CORRECTNESS_MIN_COVERAGE_RATIO,
                )
            ),
        },
    }
    # The relative model-quality bar is campaign policy and is forwarded only
    # when the manifest carries it. Repeat-loop rejection is mandatory harness
    # policy in bench/correctness.py and is intentionally absent here.
    if thresholds.get("max_mean_logprob_drop") is not None:
        correctness["thresholds"]["max_mean_logprob_drop"] = float(
            thresholds["max_mean_logprob_drop"]
        )

    scoring_rule = _parse_json_field(round_row.get("scoring_rule")) or {}
    req = {
        "schema_version": 1,
        "task_id": task_id,
        "mode": "all",
        "model": {
            "hf_repo": str(model["hf_repo"]),
            "hf_revision": str(model["hf_revision"]),
            "dtype": dtype,
            "quantization": model.get("quantization"),
            "max_model_len": max_model_len,
        },
        "hardware": {
            "gpu_count": int(bench.get("gpu_count") or 1),
            "gpu_sku_expected": str(round_row["gpu_sku"]),
        },
        "engines": {
            "baseline": {
                "image": baseline_image,
                "serve_args": list(serve_args),
                "env": {},
                "cache_dir": cache_dir,
            },
            "candidates": [
                {
                    "image": str(row["engine_image_ref"]),
                    "serve_args": list(serve_args),
                    "env": {},
                    "cache_dir": cache_dir,
                }
                for row in candidates
            ],
        },
        "workload_trace": {
            "path": trace_path,
            "sha256": str(round_row["sampled_trace_sha256"]),
        },
        "correctness": correctness,
        "sla_bench": {
            "repetitions": int(config.BENCH_SLA_REPETITIONS),
            "thresholds": {
                "p99_ttft_ms": float(ttft),
                "p99_itl_ms": float(itl),
            },
        },
        "scoring_rule": dict(scoring_rule),
        "hf_token_env": "HF_TOKEN",
    }
    try:
        validate_bench_request_dict(req)
    except RequestValidationError as exc:
        raise RoundInfraError(VOID_POD_FAILED, str(exc)) from exc
    return req


def bind_report_to_round(
    report: dict[str, Any],
    *,
    request_task_id: str,
    executed_request_bytes: bytes,
    baseline_digest: str,
    candidate_digests: list[str],
    trace_sha256: str,
) -> None:
    """Cross-check the pod report against the executed request."""
    try:
        validate_report_dict(report)
    except RequestValidationError as exc:
        raise RoundInfraError(VOID_POD_FAILED, str(exc)) from exc
    if str(report.get("task_id")) != request_task_id:
        raise RoundInfraError(VOID_POD_FAILED, "bench_report_task_id_mismatch")
    fp = report.get("inputs_fingerprint") or {}
    expect_req = sha256_bytes(executed_request_bytes)
    if str(fp.get("request_sha256", "")).lower() != expect_req.lower():
        raise RoundInfraError(VOID_POD_FAILED, "bench_report_request_sha256_mismatch")
    try:
        exp_base = extract_image_digest(baseline_digest)
        exp_cands = [extract_image_digest(d) for d in candidate_digests]
    except RequestValidationError as exc:
        raise RoundInfraError(VOID_POD_FAILED, str(exc)) from exc
    if str(fp.get("baseline_image_digest", "")).lower() != exp_base.lower():
        raise RoundInfraError(VOID_POD_FAILED, "bench_report_baseline_digest_mismatch")
    got_cands = fp.get("candidate_image_digest")
    if not isinstance(got_cands, list) or len(got_cands) != len(exp_cands):
        raise RoundInfraError(VOID_POD_FAILED, "bench_report_candidate_digest_mismatch")
    for got, exp in zip(got_cands, exp_cands, strict=True):
        if str(got).lower() != exp.lower():
            raise RoundInfraError(
                VOID_POD_FAILED, "bench_report_candidate_digest_mismatch"
            )
    if str(fp.get("trace_sha256", "")).lower() != trace_sha256.lower():
        raise RoundInfraError(VOID_POD_FAILED, "bench_report_trace_sha256_mismatch")


def entry_results_from_report(
    entries: list[dict[str, Any]],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map harness entries onto round_entries in request (id) order."""
    baseline = next((e for e in entries if e["role"] == "baseline"), None)
    candidates = [e for e in entries if e["role"] != "baseline"]
    payloads = report.get("entries") or []
    if baseline is None or len(payloads) != len(candidates):
        raise RoundInfraError(VOID_POD_FAILED, "bench_report_entry_count_mismatch")

    results: list[dict[str, Any]] = []
    base_report = report.get("baseline") or {}
    results.append(
        {
            "id": baseline["id"],
            "submission_id": baseline.get("submission_id"),
            "role": "baseline",
            "status": "scored" if base_report else "infra_failed",
            "score": 0.0 if base_report else None,
            "disqualify_reason": None
            if base_report
            else "baseline_missing_from_report",
            "report": dict(base_report) if isinstance(base_report, dict) else {},
            "engine_image_ref": baseline.get("engine_image_ref"),
            "hotkey": baseline.get("hotkey"),
        }
    )
    for row, payload in zip(candidates, payloads, strict=True):
        if not isinstance(payload, dict):
            raise RoundInfraError(VOID_POD_FAILED, "bench_report_entry_not_object")
        status = str(payload.get("status") or "infra_failed")
        if (
            row["role"] == "leader"
            and status == "disqualified"
            and payload.get("engine_crashed") is True
        ):
            # A crash-disqualified incumbent keeps the infra path: its image
            # already started and scored on a prior pod, so a startup crash
            # is far more likely infra than a deterministic patch bug, and
            # the leader-infra void keeps the crown parked instead of moving
            # it on a flake. The harness's original verdict stays in report.
            status = "infra_failed"
        score = payload.get("score")
        results.append(
            {
                "id": row["id"],
                "submission_id": row.get("submission_id"),
                "role": row["role"],
                "status": status,
                "score": None if score is None else float(score),
                "disqualify_reason": payload.get("reason"),
                "report": dict(payload),
                "engine_image_ref": row.get("engine_image_ref"),
                "hotkey": row.get("hotkey"),
            }
        )
    return results


def classify_round_failure(
    *,
    provision_error: bool = False,
    timed_out: bool = False,
    exit_code: int | None = None,
    has_report: bool = False,
    bound: bool = False,
) -> str | None:
    """Map a bench outcome to a void reason. None means settle the round."""
    if provision_error:
        return VOID_POD_PROVISION_FAILED
    if timed_out:
        return VOID_ROUND_TIMEOUT
    if has_report and bound and exit_code == EXIT_DESTROY_FAILED:
        return None
    if not has_report or not bound:
        return VOID_POD_FAILED
    if exit_code in (1, 2, 3):
        return VOID_POD_FAILED
    if exit_code not in (0, EXIT_DESTROY_FAILED):
        return VOID_POD_FAILED
    return None


def remaining_round_budget_s(
    started_at: datetime,
    *,
    now: datetime | None = None,
    max_duration_s: int | None = None,
) -> float:
    """Seconds left on the round clock. Raises if the budget is already gone."""
    current = now or datetime.now(timezone.utc)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    elapsed = (current - started_at).total_seconds()
    budget = float(
        config.ROUND_MAX_DURATION_S if max_duration_s is None else max_duration_s
    )
    leftover = budget - elapsed
    if leftover <= 0:
        raise RoundInfraError(VOID_ROUND_TIMEOUT, f"elapsed {elapsed:.0f}s")
    return leftover


def _round_phase_writer(round_id: str) -> Callable[..., bool]:
    def write(*, job_id: Any, attempt: Any, phase: str, progress: Any = None) -> bool:
        del job_id, attempt
        return set_round_phase(round_id=round_id, phase=phase, progress=progress)

    return write


def _round_heartbeat_writer(round_id: str) -> Callable[..., bool]:
    def beat(*, job_id: Any, attempt: Any) -> bool:
        del job_id, attempt
        return touch_round_heartbeat(round_id=round_id)

    return beat


def _void(round_row: dict[str, Any], reason: str) -> None:
    round_id = str(round_row["id"])
    landed = void_round(round_id, reason)
    if not landed:
        logger.info("round %s already settled; skipped void %s", round_id, reason)
        return
    logger.warning(
        "voided round %s (campaign %s): %s",
        round_row.get("ordinal"),
        round_row.get("campaign_id"),
        reason,
    )
    obs.round_voided(
        round_id=round_id,
        campaign_id=str(round_row["campaign_id"]),
        void_reason=reason,
    )


def _defer(round_row: dict[str, Any], exc: RoundDeferred) -> None:
    round_id = str(round_row["id"])
    if not defer_round_for_capacity(round_id, delay_s=exc.delay_s):
        logger.info("round %s already settled; skipped defer", round_id)
        return
    logger.warning(
        "deferred round %s (campaign %s) for %.0fs: %s",
        round_row.get("ordinal"),
        round_row.get("campaign_id"),
        exc.delay_s,
        exc.detail,
    )


def process_round(
    round_row: dict[str, Any],
    *,
    mock_bench: bool = False,
    mock_correctness_fail: bool = False,
    work_root: Path | None = None,
    run_bench_fn: Callable[..., int] | None = None,
    run_pod_fn: Callable[..., int] | None = None,
    upload_evidence_fn: Callable[..., tuple[str, str, int]] | None = None,
    trace_fetcher: Callable[[str], bytes] | None = None,
    row_fetcher: Callable[[int], dict[str, Any]] | None = None,
    resolve_image_fn: Callable[..., bool] | None = None,
    now: datetime | None = None,
) -> str:
    """Run one claimed round.

    Returns 'ok', 'deferred' when an empty GPU market sent the round back to
    the queue, or the void reason.
    """
    try:
        return _process_round(
            round_row,
            mock_bench=mock_bench,
            mock_correctness_fail=mock_correctness_fail,
            work_root=work_root,
            run_bench_fn=run_bench_fn,
            run_pod_fn=run_pod_fn,
            upload_evidence_fn=upload_evidence_fn,
            trace_fetcher=trace_fetcher,
            row_fetcher=row_fetcher,
            resolve_image_fn=resolve_image_fn,
            now=now,
        )
    except RoundDeferred as exc:
        _defer(round_row, exc)
        return "deferred"
    except RoundInfraError as exc:
        _void(round_row, exc.reason)
        return exc.reason


def _process_round(
    round_row: dict[str, Any],
    *,
    mock_bench: bool,
    mock_correctness_fail: bool,
    work_root: Path | None,
    run_bench_fn: Callable[..., int] | None,
    run_pod_fn: Callable[..., int] | None,
    upload_evidence_fn: Callable[..., tuple[str, str, int]] | None,
    trace_fetcher: Callable[[str], bytes] | None,
    row_fetcher: Callable[[int], dict[str, Any]] | None,
    resolve_image_fn: Callable[..., bool] | None,
    now: datetime | None,
) -> str:
    round_id = str(round_row["id"])
    campaign_id = str(round_row["campaign_id"])
    ordinal = int(round_row["ordinal"])
    campaign = get_campaign(campaign_id)
    if campaign is None:
        raise RoundInfraError(VOID_POD_FAILED, "campaign_missing")

    entries = list_round_entries(round_id)
    if not entries:
        raise RoundInfraError(VOID_POD_FAILED, "round_has_no_entries")

    root = Path(work_root or (config.WORK_DIR / f"round-{round_id}"))
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = materialize_round_trace(
        round_row,
        campaign,
        root / "trace",
        fetcher=trace_fetcher,
        row_fetcher=row_fetcher,
    )
    remaining_round_budget_s(round_row["started_at"], now=now)

    leader_entry = next((e for e in entries if e["role"] == "leader"), None)
    if leader_entry is not None and not mock_bench:
        resolver = resolve_image_fn or image_ref_resolves
        if not resolver(str(leader_entry["engine_image_ref"])):
            raise RoundInfraError(
                VOID_LEADER_IMAGE_MISSING, str(leader_entry["engine_image_ref"])
            )

    task_id = str(uuid4())
    req = build_round_request(
        round_row,
        campaign,
        entries,
        task_id=task_id,
        trace_path=str(trace_path.resolve()),
    )
    request_path = root / "bench_request.json"
    request_bytes = (json.dumps(req, indent=2) + "\n").encode("utf-8")
    request_path.write_bytes(request_bytes)

    reporter = PhaseReporter(
        job_id=round_id,
        attempt=0,
        phase_writer=_round_phase_writer(round_id),
        heartbeat_writer=_round_heartbeat_writer(round_id),
        label=f"round {ordinal}",
    )

    candidates = [e for e in entries if e["role"] != "baseline"]
    mock_plan = MockPlan(
        baseline_token_latency_s=0.004 if mock_bench else 0.0,
        candidates=[MockCandidatePlan() for _ in candidates],
    )
    if mock_correctness_fail and mock_plan.candidates:
        mock_plan.candidates[-1] = MockCandidatePlan(garbage=True)

    exit_code: int
    provision_error = False
    timed_out = False
    with reporter:
        if mock_bench:
            fn = run_bench_fn or run_bench
            exit_code = fn(
                request_path,
                output_dir,
                mock_engine=True,
                mock_plan=mock_plan,
            )
            remote_req = output_dir / "bench_request.remote.json"
            if not remote_req.is_file():
                remote_req.write_bytes(request_bytes)
        else:
            bench = campaign.bench if isinstance(campaign.bench, dict) else {}
            ttl_hours = float(config.ROUND_MAX_DURATION_S) / 3600.0 + 1.0
            spec = PodSpec(
                provider=config.GPU_PROVIDER,
                gpu_count=int(bench.get("gpu_count") or 1),
                gpu_type=str(round_row["gpu_sku"]),
                ttl_hours=ttl_hours,
                max_hourly_cents=config.GPU_MAX_HOURLY_CENTS,
            )
            leftover = remaining_round_budget_s(round_row["started_at"], now=now)
            fn = run_pod_fn or run_bench_on_pod
            try:
                exit_code = fn(
                    spec,
                    request_path=request_path,
                    output_dir=output_dir,
                    on_phase=reporter.set,
                    bench_timeout_s=leftover,
                )
            except NoCapacityError as exc:
                # Nothing was rented, so the cohort and seed stay valid. Voiding
                # here is what let an out-of-stock market burn a round every
                # poll interval; keep the round and wait the market out.
                raise RoundDeferred(
                    capacity_retry_delay_s(
                        int(round_row.get("provision_attempts") or 0)
                    ),
                    str(exc),
                ) from exc
            except ProvisionError as exc:
                provision_error = True
                raise RoundInfraError(VOID_POD_PROVISION_FAILED, str(exc)) from exc
            except GpuError as exc:
                timed_out = "timed out" in str(exc).lower()
                reason = VOID_ROUND_TIMEOUT if timed_out else VOID_POD_FAILED
                raise RoundInfraError(reason, str(exc)) from exc

    report_path = output_dir / "bench_report.json"
    has_report = report_path.is_file()
    report: dict[str, Any] | None = None
    bound = False
    if has_report:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if mock_bench:
            executed_bytes = request_bytes
        else:
            remote_req = output_dir / "bench_request.remote.json"
            if not remote_req.is_file():
                raise RoundInfraError(VOID_POD_FAILED, "bench_remote_request_missing")
            executed_bytes = remote_req.read_bytes()
        bind_report_to_round(
            report,
            request_task_id=task_id,
            executed_request_bytes=executed_bytes,
            baseline_digest=str(req["engines"]["baseline"]["image"]),
            candidate_digests=[str(c["image"]) for c in req["engines"]["candidates"]],
            trace_sha256=str(round_row["sampled_trace_sha256"]),
        )
        bound = True

    void_reason = classify_round_failure(
        provision_error=provision_error,
        timed_out=timed_out,
        exit_code=exit_code,
        has_report=has_report,
        bound=bound,
    )
    if void_reason is not None:
        raise RoundInfraError(void_reason, f"exit_code={exit_code}")

    if exit_code == EXIT_DESTROY_FAILED:
        logger.error(
            "ALERT round_destroy_failed round=%s campaign=%s "
            "(pod/volume may still be billing); settling the bound report",
            round_id,
            campaign_id,
        )

    assert report is not None
    results = entry_results_from_report(entries, report)
    drift = report.get("baseline_drift")
    drift_f = None if drift is None else float(drift)
    rank_entries = [
        Entry(
            role=str(r["role"]),
            submission_id=(
                str(r["submission_id"]) if r.get("submission_id") is not None else None
            ),
            score=r.get("score"),
            status=str(r["status"]),
        )
        for r in results
    ]
    # A disqualified incumbent carries no in-round score; rank_round reports
    # the stored leaders.last_score as prev_score on the history row instead.
    incumbent_score = None
    if round_row.get("incumbent_submission_id") is not None:
        leader_row = get_leader(campaign_id)
        if leader_row is not None and leader_row.get("last_score") is not None:
            incumbent_score = float(leader_row["last_score"])
    decision = rank_round(
        rank_entries,
        epsilon=float(config.OVERTAKE_EPSILON),
        drift=drift_f,
        drift_ceiling=float(config.BASELINE_DRIFT_CEILING),
        leader_score=incumbent_score,
    )
    if decision.void:
        raise RoundInfraError(str(decision.void_reason))

    evidence_url = None
    if not mock_bench:
        from storage.s3 import upload_round_evidence

        up = upload_evidence_fn or upload_round_evidence
        evidence_url, _digest, _size = up(round_id, task_id, output_dir)

    landed = complete_round(
        round_id=round_id,
        campaign_id=campaign_id,
        ordinal=ordinal,
        decision=decision,
        entries=results,
        baseline_drift=drift_f,
        epsilon=float(config.OVERTAKE_EPSILON),
        evidence_s3_url=evidence_url,
    )
    if not landed:
        logger.info("round %s lost the complete race to a concurrent void", round_id)
        return "lost_race"
    logger.info(
        "round %s complete leader_changed=%s winner=%s",
        ordinal,
        decision.leader_changed,
        decision.leader_submission_id,
    )
    return "ok"
