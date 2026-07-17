"""Module B — closed-loop perf screen (throughput smoke test).

Fixed-concurrency, non-streaming /v1/completions. Baseline then candidate run
sequentially (one engine at a time). Throughput = total server-reported
completion tokens / wall time. verdict pass iff candidate/baseline ratio >=
min_throughput_ratio. Stdlib only; failures raise EngineError (CLI exit 3).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

from bench.http import post_completion
from bench.lifecycle import EngineError
from bench.schemas import PerfScreenConfig, PerfScreenReport, TraceRequest

logger = logging.getLogger(__name__)

EVIDENCE_FILENAME = "perf_screen.jsonl"
SUMMARY_FILENAME = "summary.json"


def _run_engine(
    base_url: str,
    *,
    role: str,
    requests: list[TraceRequest],
    concurrency: int,
    timeout_s: float,
) -> tuple[float, int, float, list[dict]]:
    """Closed-loop run for one engine. Returns (wall_s, total_tokens, tok/s, rows)."""
    work: queue.Queue[int] = queue.Queue()
    for i in range(len(requests)):
        work.put(i)

    rows: list[dict] = []
    rows_lock = threading.Lock()
    errors: list[str] = []

    def worker() -> None:
        while True:
            try:
                idx = work.get_nowait()
            except queue.Empty:
                return
            req = requests[idx]
            start = time.monotonic()
            try:
                resp = post_completion(
                    base_url,
                    prompt=req.prompt or "",
                    max_tokens=req.max_tokens,
                    temperature=req.sampling.temperature,
                    top_p=req.sampling.top_p,
                    seed=0,
                    timeout=timeout_s,
                )
                latency = time.monotonic() - start
                usage = resp.get("usage") or {}
                row = {
                    "engine_role": role,
                    "request_id": req.id,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "latency_s": round(latency, 6),
                    "status": "ok",
                    "error": None,
                }
            except EngineError as exc:
                latency = time.monotonic() - start
                row = {
                    "engine_role": role,
                    "request_id": req.id,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "latency_s": round(latency, 6),
                    "status": "error",
                    "error": str(exc),
                }
                errors.append(f"{role}/{req.id}: {exc}")
            with rows_lock:
                rows.append(row)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise EngineError(
            f"perf_screen {role} had {len(errors)} failed request(s): {errors[0]}"
        )

    # Throughput from measured per-request latency, not wall time: at
    # concurrency=1 the busy window equals the sum of latencies, and wall time
    # also absorbs thread startup/HTTP overhead that swamps sub-ms requests and
    # makes the ratio non-deterministic on fast (mock) engines.
    busy_s = sum(float(r["latency_s"]) for r in rows)
    total_tokens = sum(int(r["completion_tokens"] or 0) for r in rows)
    if busy_s <= 0:
        raise EngineError(f"perf_screen {role}: non-positive cumulative latency")
    tps = total_tokens / busy_s
    logger.info(
        "perf_screen %s: %d requests, %d tokens, %.3fs busy, %.1f tok/s",
        role,
        len(rows),
        total_tokens,
        busy_s,
        tps,
    )
    return busy_s, total_tokens, tps, rows


def run_perf_screen(
    baseline_url: str,
    candidate_url: str,
    *,
    requests: list[TraceRequest],
    cfg: PerfScreenConfig,
    evidence_dir: Path,
    request_timeout_s: float = 120.0,
) -> PerfScreenReport:
    """Run Module B against two healthy base URLs (baseline then candidate)."""
    subset = list(requests)[: cfg.num_requests]
    base = run_perf_screen_engine(
        baseline_url,
        role="baseline",
        requests=subset,
        cfg=cfg,
        evidence_dir=evidence_dir,
        request_timeout_s=request_timeout_s,
    )
    return finish_perf_screen(
        candidate_url,
        baseline=base,
        requests=subset,
        cfg=cfg,
        evidence_dir=evidence_dir,
        request_timeout_s=request_timeout_s,
    )


def run_perf_screen_engine(
    base_url: str,
    *,
    role: str,
    requests: list[TraceRequest],
    cfg: PerfScreenConfig,
    evidence_dir: Path,
    request_timeout_s: float = 120.0,
) -> dict:
    """Run one engine's closed-loop screen; persist rows; return engine metrics."""
    if not requests:
        raise EngineError("perf_screen: no requests to run")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    wall, tokens, tps, rows = _run_engine(
        base_url,
        role=role,
        requests=requests,
        concurrency=cfg.concurrency,
        timeout_s=request_timeout_s,
    )
    rows_path = evidence_dir / f"{role}_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as ef:
        for row in rows:
            ef.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "role": role,
        "wall_s": wall,
        "completion_tokens": tokens,
        "output_tokens_per_s": tps,
        "num_requests": len(rows),
        "rows_file": rows_path.name,
    }


def finish_perf_screen(
    candidate_url: str,
    *,
    baseline: dict,
    requests: list[TraceRequest],
    cfg: PerfScreenConfig,
    evidence_dir: Path,
    request_timeout_s: float = 120.0,
) -> PerfScreenReport:
    """Run candidate, combine with baseline metrics, emit report + evidence."""
    cand = run_perf_screen_engine(
        candidate_url,
        role="candidate",
        requests=requests,
        cfg=cfg,
        evidence_dir=evidence_dir,
        request_timeout_s=request_timeout_s,
    )
    base_tps = float(baseline["output_tokens_per_s"])
    cand_tps = float(cand["output_tokens_per_s"])
    ratio = cand_tps / base_tps if base_tps > 0 else 0.0
    verdict = "pass" if ratio >= cfg.min_throughput_ratio else "fail_perf_screen"

    # Merge per-role rows into the canonical evidence file (.partial -> rename).
    evidence_path = evidence_dir / EVIDENCE_FILENAME
    partial_path = evidence_dir / f"{EVIDENCE_FILENAME}.partial"
    with partial_path.open("w", encoding="utf-8") as ef:
        for role_metrics in (baseline, cand):
            for line in (
                (evidence_dir / role_metrics["rows_file"])
                .read_text(encoding="utf-8")
                .splitlines()
            ):
                if line.strip():
                    ef.write(line + "\n")
    partial_path.replace(evidence_path)

    summary = {
        "baseline_output_tokens_per_s": base_tps,
        "candidate_output_tokens_per_s": cand_tps,
        "throughput_ratio": ratio,
        "baseline_wall_s": baseline["wall_s"],
        "candidate_wall_s": cand["wall_s"],
        "baseline_completion_tokens": baseline["completion_tokens"],
        "candidate_completion_tokens": cand["completion_tokens"],
        "num_requests": cand["num_requests"],
        "concurrency": cfg.concurrency,
        "min_throughput_ratio": cfg.min_throughput_ratio,
        "verdict": verdict,
    }
    (evidence_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for role_metrics in (baseline, cand):
        (evidence_dir / role_metrics["rows_file"]).unlink(missing_ok=True)

    return PerfScreenReport(
        verdict=verdict,
        baseline_output_tokens_per_s=base_tps,
        candidate_output_tokens_per_s=cand_tps,
        throughput_ratio=ratio,
        evidence=f"evidence/perf_screen/{EVIDENCE_FILENAME}",
    )
