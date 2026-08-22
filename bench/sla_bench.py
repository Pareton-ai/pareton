"""SLA benchmark (open-loop trace replay, streaming).

Per engine, per repetition: fire each trace request on its own thread after
``arrival_offset_ms`` (open-loop: arrivals never block on capacity). Streaming
/v1/completions yields TTFT/ITL/e2e per request. ITL percentiles are computed
over the pooled set of inter-token gaps across all requests in a repetition
(decision: pooled ITL). Median across repetitions -> EngineSlaMetrics.

The replay also captures each request's output text. That text is what the
correctness gate grades: one shared scorer reads it once every candidate has
run. Each engine is replayed exactly once, in production configuration.

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
from dataclasses import dataclass
from pathlib import Path

from bench.http import post_completion_stream
from bench.lifecycle import EngineError
from bench.schemas import (
    EngineSlaMetrics,
    EngineSlaResult,
    LatencyPercentiles,
    SlaBenchConfig,
    TraceRequest,
)
from bench.score import PromptTiming
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
            f"trace request {req.id!r}: sla_bench requires a text prompt "
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
            "text": res.text,
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
            "text": "",
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


@dataclass(frozen=True)
class EngineReplay:
    """One engine's finished replay: the report payload plus its output text.

    ``outputs`` maps request id to the text that engine produced on the same
    repetition ``result.timings`` was taken from, so a candidate is graded on
    the very run its speed was measured on.
    """

    result: EngineSlaResult
    outputs: dict[str, str]


def _run_engine(
    base_url: str,
    *,
    role: str,
    requests: list[TraceRequest],
    cfg: SlaBenchConfig,
    engine_evidence_dir: Path,
    timeout_s: float,
) -> tuple[list[dict], list[dict]]:
    """Warmup + N measured reps for one engine.

    Returns (per-rep metric dicts, every measured row across reps).
    """
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
    measured: list[dict] = []
    for rep in range(1, cfg.repetitions + 1):
        rep_rows, wall_s, errs = _replay(
            base_url, requests, role=role, rep=rep, is_warmup=False, timeout_s=timeout_s
        )
        _write_rep(engine_evidence_dir / f"rep_{rep}", rep_rows, wall_s)
        if errs:
            raise EngineError(
                f"sla_bench {role} rep {rep} had {len(errs)} failed request(s): {errs[0]}"
            )
        measured.extend(rep_rows)
        rep_metrics.append(
            aggregate_rep_metrics(
                rep_rows,
                wall_s=wall_s,
                p99_ttft_ms=cfg.thresholds.p99_ttft_ms,
                p99_itl_ms=cfg.thresholds.p99_itl_ms,
            )
        )
    return rep_metrics, measured


def _median_rep_row(rows: list[dict]) -> dict:
    """The repetition whose end-to-end latency is the median for this request.

    Picking one real repetition keeps ``(ttft, itl, tokens, text)`` a
    self-consistent set. Averaging would blend ITL vectors of different lengths
    and pair timings with text that never occurred together.
    """
    ordered = sorted(rows, key=lambda r: float(r["e2e_ms"]))
    return ordered[(len(ordered) - 1) // 2]


def _per_request(rows: list[dict]) -> tuple[dict[str, PromptTiming], dict[str, str]]:
    by_id: dict[str, list[dict]] = {}
    for row in rows:
        by_id.setdefault(str(row["request_id"]), []).append(row)
    timings: dict[str, PromptTiming] = {}
    outputs: dict[str, str] = {}
    for rid, rid_rows in by_id.items():
        row = _median_rep_row(rid_rows)
        timings[rid] = PromptTiming(
            ttft_s=float(row["ttft_ms"]) / 1000.0,
            itl_s=[float(x) / 1000.0 for x in (row.get("itl_ms") or [])],
            completion_tokens=int(row.get("completion_tokens") or 0),
        )
        outputs[rid] = str(row.get("text") or "")
    return timings, outputs


def _rep_dir_paths(base: Path, repetitions: int) -> list[Path]:
    return [base / f"rep_{n}" / REQUESTS_FILENAME for n in range(1, repetitions + 1)]


def run_sla_engine(
    base_url: str,
    *,
    role: str,
    requests: list[TraceRequest],
    cfg: SlaBenchConfig,
    evidence_dir: Path,
    request_timeout_s: float = 120.0,
) -> EngineReplay:
    """Replay the trace against one healthy engine and persist its evidence.

    Every engine in a round goes through this one path: the baseline, each
    candidate, and the closing drift baseline. Nothing here varies by engine
    or by role, so every image in the round is measured the same way.
    """
    if not requests:
        raise EngineError("sla_bench: empty workload trace")
    for req in requests:
        _require_text_prompt(req)

    engine_evidence_dir = evidence_dir / role
    rep_metrics, measured = _run_engine(
        base_url,
        role=role,
        requests=requests,
        cfg=cfg,
        engine_evidence_dir=engine_evidence_dir,
        timeout_s=request_timeout_s,
    )
    metrics = _engine_metrics_from_reps(rep_metrics)

    def rel_range(key: str, stat: str) -> float:
        return _relative_range([getattr(m[key], stat) for m in rep_metrics])

    cross_rep_variance = {
        "p99_ttft_ms_rel_range": rel_range("ttft_ms", "p99"),
        "p99_itl_ms_rel_range": rel_range("itl_ms", "p99"),
        "p99_e2e_ms_rel_range": rel_range("e2e_ms", "p99"),
    }
    if cross_rep_variance["p99_ttft_ms_rel_range"] > REPRO_BAR_MAX_REL_RANGE:
        logger.warning(
            "sla_bench %s p99 TTFT rel_range %.4f exceeds the reproducibility "
            "bar %.4f; the round's drift check is the backstop",
            role,
            cross_rep_variance["p99_ttft_ms_rel_range"],
            REPRO_BAR_MAX_REL_RANGE,
        )

    # Hard requirement: independently recompute this engine's metrics from the
    # persisted rep_<n>/requests.jsonl via a separate code path and confirm it
    # matches the online aggregation before trusting the numbers.
    recomputed = recompute_sla_metrics(
        engine_evidence_dir,
        p99_ttft_ms=cfg.thresholds.p99_ttft_ms,
        p99_itl_ms=cfg.thresholds.p99_itl_ms,
        repetitions=cfg.repetitions,
    )
    _assert_metrics_close(recomputed, metrics)

    timings, outputs = _per_request(measured)
    result = EngineSlaResult(
        role=role,
        metrics=metrics,
        cross_rep_variance=cross_rep_variance,
        timings=timings,
        evidence=f"evidence/sla_bench/{role}",
    )
    return EngineReplay(result=result, outputs=outputs)


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
