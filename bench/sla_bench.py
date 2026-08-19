"""Module C — SLA benchmark (open-loop trace replay, streaming).

Per engine, per repetition: fire each trace request on its own thread after
``arrival_offset_ms`` (open-loop: arrivals never block on capacity). Streaming
/v1/completions yields TTFT/ITL/e2e per request. ITL percentiles are computed
over the pooled set of inter-token gaps across all requests in a repetition
(decision: pooled ITL). Median across repetitions -> EngineSlaMetrics.

Percentiles use linear interpolation (numpy-default definition) in pure Python
so the independent recompute check matches the online aggregation exactly.
Stdlib only; failures raise EngineError (CLI exit 3).
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from pathlib import Path

from bench.http import post_completion_stream
from bench.lifecycle import EngineError
from bench.schemas import (
    EngineSlaMetrics,
    LatencyPercentiles,
    SlaBenchConfig,
    SlaBenchReport,
    TraceRequest,
    WorkloadTrace,
)
from bench.validate import RequestValidationError

logger = logging.getLogger(__name__)

REQUESTS_FILENAME = "requests.jsonl"
WARMUP_DIRNAME = "warmup"
# Measured on B7 Hopper 2026-08-03: outer p99 TTFT rel_range ~0.167. The bar
# is that maximum doubled.
REPRO_BAR_MAX_REL_RANGE = 0.335  # p99 TTFT relative range ceiling


def _require_text_prompt(req: TraceRequest) -> str:
    if req.prompt is None or req.prompt == "":
        raise RequestValidationError(
            f"trace request {req.id!r}: Module C requires a text prompt "
            f"(prompt_token_ids-only entries are not supported yet)"
        )
    return req.prompt


def percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy 'linear'/default definition).

    pct in [0, 100]. Position = (pct/100) * (n - 1); interpolate between the
    bracketing order statistics.
    """
    if not samples:
        raise ValueError("percentile of empty sample set")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct out of range: {pct}")
    s = sorted(float(x) for x in samples)
    n = len(s)
    if n == 1:
        return s[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _percentiles(samples: list[float]) -> LatencyPercentiles:
    return LatencyPercentiles(
        p50=percentile(samples, 50),
        p95=percentile(samples, 95),
        p99=percentile(samples, 99),
    )


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def aggregate_rep_metrics(
    rows: list[dict],
    *,
    wall_s: float,
    p99_ttft_ms: float,
    p99_itl_ms: float,
) -> dict:
    """One repetition's metrics from measured request rows (warmup excluded).

    TTFT/e2e are per-request samples; ITL is the pooled concatenation of every
    request's inter-token gaps. Goodput = fraction of requests whose TTFT <=
    p99_ttft_ms AND whose per-request ITL p99 <= p99_itl_ms.
    """
    ttft = [float(r["ttft_ms"]) for r in rows]
    e2e = [float(r["e2e_ms"]) for r in rows]
    pooled_itl: list[float] = []
    for r in rows:
        pooled_itl.extend(float(x) for x in (r.get("itl_ms") or []))

    total_tokens = sum(int(r.get("completion_tokens") or 0) for r in rows)
    good = 0
    for r in rows:
        req_itl = [float(x) for x in (r.get("itl_ms") or [])]
        n_tok = int(r.get("completion_tokens") or 0)
        # Multi-token replies must expose inter-token gaps; empty ITL would
        # otherwise vacuous-pass the ITL gate and report p99 ITL as 0.
        if n_tok >= 2 and not req_itl:
            raise EngineError(
                f"sla_bench: request {r.get('request_id')!r} has {n_tok} "
                f"completion tokens but no inter-token latency samples"
            )
        itl_ok = True if not req_itl else percentile(req_itl, 99) <= p99_itl_ms
        if float(r["ttft_ms"]) <= p99_ttft_ms and itl_ok:
            good += 1

    return {
        "ttft_ms": _percentiles(ttft),
        "itl_ms": _percentiles(pooled_itl)
        if pooled_itl
        else LatencyPercentiles(0.0, 0.0, 0.0),
        "e2e_ms": _percentiles(e2e),
        "output_tokens_per_s": (total_tokens / wall_s) if wall_s > 0 else 0.0,
        "requests_per_s": (len(rows) / wall_s) if wall_s > 0 else 0.0,
        "sla_goodput_ratio": (good / len(rows)) if rows else 0.0,
    }


def _engine_metrics_from_reps(rep_metrics: list[dict]) -> EngineSlaMetrics:
    """Median across repetitions -> EngineSlaMetrics."""

    def med_pct(key: str) -> LatencyPercentiles:
        return LatencyPercentiles(
            p50=_median([m[key].p50 for m in rep_metrics]),
            p95=_median([m[key].p95 for m in rep_metrics]),
            p99=_median([m[key].p99 for m in rep_metrics]),
        )

    return EngineSlaMetrics(
        ttft_ms=med_pct("ttft_ms"),
        itl_ms=med_pct("itl_ms"),
        e2e_ms=med_pct("e2e_ms"),
        output_tokens_per_s=_median([m["output_tokens_per_s"] for m in rep_metrics]),
        requests_per_s=_median([m["requests_per_s"] for m in rep_metrics]),
        sla_goodput_ratio=_median([m["sla_goodput_ratio"] for m in rep_metrics]),
    )


def _relative_range(values: list[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    if med == 0:
        return 0.0
    return float((max(values) - min(values)) / med)


def _fire(
    base_url: str,
    req: TraceRequest,
    *,
    role: str,
    rep: int,
    is_warmup: bool,
    t0: float,
    timeout_s: float,
    out: list,
    lock: threading.Lock,
    errs: list,
) -> None:
    delay = req.arrival_offset_ms / 1000.0 - (time.monotonic() - t0)
    if delay > 0:
        time.sleep(delay)
    try:
        res = post_completion_stream(
            base_url,
            prompt=_require_text_prompt(req),
            max_tokens=req.max_tokens,
            temperature=req.sampling.temperature,
            top_p=req.sampling.top_p,
            seed=0,
            timeout=timeout_s,
        )
        row = {
            "rep": rep,
            "engine_role": role,
            "request_id": req.id,
            "arrival_offset_ms": req.arrival_offset_ms,
            "warmup": is_warmup,
            "ttft_ms": round(res.ttft_s * 1000.0, 3),
            "itl_ms": [round(x * 1000.0, 3) for x in res.itl_s],
            "e2e_ms": round(res.e2e_s * 1000.0, 3),
            "completion_tokens": res.completion_tokens,
            "finish_reason": res.finish_reason,
            "error": None,
        }
    except EngineError as exc:
        row = {
            "rep": rep,
            "engine_role": role,
            "request_id": req.id,
            "arrival_offset_ms": req.arrival_offset_ms,
            "warmup": is_warmup,
            "ttft_ms": None,
            "itl_ms": [],
            "e2e_ms": None,
            "completion_tokens": None,
            "finish_reason": None,
            "error": str(exc),
        }
        if not is_warmup:
            errs.append(f"{role}/rep{rep}/{req.id}: {exc}")
    with lock:
        out.append(row)


def _replay(
    base_url: str,
    requests: list[TraceRequest],
    *,
    role: str,
    rep: int,
    is_warmup: bool,
    timeout_s: float,
) -> tuple[list[dict], float, list[str]]:
    """Open-loop replay of one request set. Returns (rows, wall_s, errors)."""
    t0 = time.monotonic()
    lock = threading.Lock()
    errs: list[str] = []
    rows: list[dict] = []
    threads = [
        threading.Thread(
            target=_fire,
            args=(base_url, r),
            kwargs=dict(
                role=role,
                rep=rep,
                is_warmup=is_warmup,
                t0=t0,
                timeout_s=timeout_s,
                out=rows,
                lock=lock,
                errs=errs,
            ),
            daemon=True,
        )
        for r in requests
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return rows, time.monotonic() - t0, errs


def _write_rep(rep_dir: Path, rows: list[dict], wall_s: float) -> None:
    rep_dir.mkdir(parents=True, exist_ok=True)
    path = rep_dir / REQUESTS_FILENAME
    tmp = rep_dir / f"{REQUESTS_FILENAME}.partial"
    with tmp.open("w", encoding="utf-8") as ef:
        for row in rows:
            ef.write(json.dumps(row, sort_keys=True) + "\n")
        ef.write(json.dumps({"_rep_meta": True, "wall_s": wall_s}) + "\n")
    tmp.replace(path)


def _run_engine(
    base_url: str,
    *,
    role: str,
    requests: list[TraceRequest],
    cfg: SlaBenchConfig,
    engine_evidence_dir: Path,
    timeout_s: float,
) -> list[dict]:
    """Warmup + N measured reps for one engine. Returns per-rep metric dicts."""
    if cfg.warmup_requests > 0:
        warm_rows, _, _ = _replay(
            base_url,
            requests[: cfg.warmup_requests],
            role=role,
            rep=0,
            is_warmup=True,
            timeout_s=timeout_s,
        )
        _write_rep(engine_evidence_dir / WARMUP_DIRNAME, warm_rows, 0.0)

    rep_metrics: list[dict] = []
    for rep in range(1, cfg.repetitions + 1):
        rep_rows, wall_s, errs = _replay(
            base_url, requests, role=role, rep=rep, is_warmup=False, timeout_s=timeout_s
        )
        _write_rep(engine_evidence_dir / f"rep_{rep}", rep_rows, wall_s)
        if errs:
            raise EngineError(
                f"sla_bench {role} rep {rep} had {len(errs)} failed request(s): {errs[0]}"
            )
        rep_metrics.append(
            aggregate_rep_metrics(
                rep_rows,
                wall_s=wall_s,
                p99_ttft_ms=cfg.thresholds.p99_ttft_ms,
                p99_itl_ms=cfg.thresholds.p99_itl_ms,
            )
        )
    return rep_metrics


def _rep_dir_paths(base: Path, repetitions: int) -> list[Path]:
    return [base / f"rep_{n}" / REQUESTS_FILENAME for n in range(1, repetitions + 1)]


def run_sla_bench(
    baseline_url: str,
    candidate_url: str,
    *,
    trace: WorkloadTrace,
    cfg: SlaBenchConfig,
    evidence_dir: Path,
    request_timeout_s: float = 120.0,
) -> SlaBenchReport:
    """Run Module C against two healthy base URLs (baseline then candidate)."""
    requests = list(trace.requests)
    base_reps = run_sla_engine(
        baseline_url,
        role="baseline",
        requests=requests,
        cfg=cfg,
        evidence_dir=evidence_dir,
        request_timeout_s=request_timeout_s,
    )
    return finish_sla_bench(
        candidate_url,
        baseline_reps=base_reps,
        requests=requests,
        cfg=cfg,
        evidence_dir=evidence_dir,
        request_timeout_s=request_timeout_s,
    )


def run_sla_engine(
    base_url: str,
    *,
    role: str,
    requests: list[TraceRequest],
    cfg: SlaBenchConfig,
    evidence_dir: Path,
    request_timeout_s: float = 120.0,
) -> list[dict]:
    """Run one engine's warmup + measured reps; persist evidence; return rep metrics."""
    if not requests:
        raise EngineError("sla_bench: empty workload trace")
    for req in requests:
        _require_text_prompt(req)
    return _run_engine(
        base_url,
        role=role,
        requests=requests,
        cfg=cfg,
        engine_evidence_dir=evidence_dir / role,
        timeout_s=request_timeout_s,
    )


def finish_sla_bench(
    candidate_url: str,
    *,
    baseline_reps: list[dict],
    requests: list[TraceRequest],
    cfg: SlaBenchConfig,
    evidence_dir: Path,
    request_timeout_s: float = 120.0,
) -> SlaBenchReport:
    """Run candidate, combine with baseline rep metrics, emit report."""
    cand_reps = run_sla_engine(
        candidate_url,
        role="candidate",
        requests=requests,
        cfg=cfg,
        evidence_dir=evidence_dir,
        request_timeout_s=request_timeout_s,
    )

    baseline = _engine_metrics_from_reps(baseline_reps)
    candidate = _engine_metrics_from_reps(cand_reps)

    def rel_range(reps: list[dict], key: str, stat: str) -> float:
        return _relative_range([getattr(m[key], stat) for m in reps])

    cross_rep_variance = {
        "p99_ttft_ms_rel_range": rel_range(cand_reps, "ttft_ms", "p99"),
        "p99_itl_ms_rel_range": rel_range(cand_reps, "itl_ms", "p99"),
        "p99_e2e_ms_rel_range": rel_range(cand_reps, "e2e_ms", "p99"),
    }
    repro_ok = cross_rep_variance["p99_ttft_ms_rel_range"] <= REPRO_BAR_MAX_REL_RANGE

    sla_pass = (
        candidate.ttft_ms.p99 <= cfg.thresholds.p99_ttft_ms
        and candidate.itl_ms.p99 <= cfg.thresholds.p99_itl_ms
    )

    def ratio(num: float, den: float) -> float:
        return (num / den) if den else 0.0

    speedup = {
        "output_tokens_per_s_ratio": ratio(
            candidate.output_tokens_per_s, baseline.output_tokens_per_s
        ),
        "requests_per_s_ratio": ratio(
            candidate.requests_per_s, baseline.requests_per_s
        ),
        "p99_ttft_ratio": ratio(baseline.ttft_ms.p99, candidate.ttft_ms.p99),
        "p99_itl_ratio": ratio(baseline.itl_ms.p99, candidate.itl_ms.p99),
        "p99_e2e_ratio": ratio(baseline.e2e_ms.p99, candidate.e2e_ms.p99),
    }

    if not repro_ok:
        verdict = "error"
    elif sla_pass:
        verdict = "pass"
    else:
        verdict = "fail_sla"

    # Hard requirement: independently recompute the candidate metrics from the
    # persisted rep_<n>/requests.jsonl via a separate code path and confirm it
    # matches the online aggregation before trusting the report.
    recomputed = recompute_sla_metrics(
        evidence_dir / "candidate",
        p99_ttft_ms=cfg.thresholds.p99_ttft_ms,
        p99_itl_ms=cfg.thresholds.p99_itl_ms,
        repetitions=cfg.repetitions,
    )
    _assert_metrics_close(recomputed, candidate)

    return SlaBenchReport(
        verdict=verdict,
        repetitions=cfg.repetitions,
        candidate=candidate,
        baseline=baseline,
        speedup=speedup,
        cross_rep_variance=cross_rep_variance,
        evidence="evidence/sla_bench",
    )


def _assert_metrics_close(
    a: EngineSlaMetrics, b: EngineSlaMetrics, *, tol: float = 1e-6
) -> None:
    def pct_close(x: LatencyPercentiles, y: LatencyPercentiles, name: str) -> None:
        for stat in ("p50", "p95", "p99"):
            if abs(getattr(x, stat) - getattr(y, stat)) > tol:
                raise EngineError(
                    f"sla recompute mismatch on {name}.{stat}: "
                    f"{getattr(x, stat)} != {getattr(y, stat)}"
                )

    pct_close(a.ttft_ms, b.ttft_ms, "ttft_ms")
    pct_close(a.itl_ms, b.itl_ms, "itl_ms")
    pct_close(a.e2e_ms, b.e2e_ms, "e2e_ms")


def recompute_sla_metrics(
    engine_evidence_dir: Path,
    *,
    p99_ttft_ms: float,
    p99_itl_ms: float,
    repetitions: int,
) -> EngineSlaMetrics:
    """Independent recompute from persisted rep_<n>/requests.jsonl files.

    Separate code path: reads per-rep rows + recorded wall time, rebuilds the
    median-of-reps metrics, and verifies the online aggregation.
    """
    rep_metrics: list[dict] = []
    for path in _rep_dir_paths(engine_evidence_dir, repetitions):
        if not path.is_file():
            raise EngineError(f"recompute: missing evidence file {path}")
        rows: list[dict] = []
        wall_s = 0.0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("_rep_meta"):
                wall_s = float(obj["wall_s"])
                continue
            if obj.get("warmup"):
                continue
            if obj.get("error") is not None:
                raise EngineError(
                    f"recompute: request {obj.get('request_id')} had error"
                )
            rows.append(obj)
        if not rows:
            raise EngineError(f"recompute: no measured rows in {path}")
        rep_metrics.append(
            aggregate_rep_metrics(
                rows,
                wall_s=wall_s,
                p99_ttft_ms=p99_ttft_ms,
                p99_itl_ms=p99_itl_ms,
            )
        )
    return _engine_metrics_from_reps(rep_metrics)
