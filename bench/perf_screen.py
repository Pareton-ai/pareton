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
from bench.validate import RequestValidationError

logger = logging.getLogger(__name__)

EVIDENCE_FILENAME = "perf_screen.jsonl"
SUMMARY_FILENAME = "summary.json"


def _require_text_prompt(req: TraceRequest) -> str:
    if req.prompt is None or req.prompt == "":
        raise RequestValidationError(
            f"trace request {req.id!r}: Module B requires a text prompt "
            f"(prompt_token_ids-only entries are not supported yet)"
        )
    return req.prompt


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
    # Closed-loop wall = first request send → last response. Summing per-request
    # latencies is wrong when concurrency > 1 (parallel work would be double-counted).
    state = {"first_send": None, "last_done": None}
    state_lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                idx = work.get_nowait()
            except queue.Empty:
                return
            req = requests[idx]
            start = time.monotonic()
            with state_lock:
                if state["first_send"] is None:
                    state["first_send"] = start
            try:
                resp = post_completion(
                    base_url,
                    prompt=_require_text_prompt(req),
                    max_tokens=req.max_tokens,
                    temperature=req.sampling.temperature,
                    top_p=req.sampling.top_p,
                    logprobs=None,
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
            finally:
                with state_lock:
                    state["last_done"] = time.monotonic()
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

    wall = (state["last_done"] - state["first_send"]) if state["first_send"] else 0.0
    total_tokens = sum(int(r["completion_tokens"] or 0) for r in rows)
    if wall <= 0:
        raise EngineError(f"perf_screen {role}: non-positive wall time")
    tps = total_tokens / wall
    logger.info(
        "perf_screen %s: %d requests, %d tokens, %.3fs wall, %.1f tok/s",
        role,
        len(rows),
        total_tokens,
        wall,
        tps,
    )
    return wall, total_tokens, tps, rows


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
    base = run_perf_screen_engine(
        baseline_url,
        role="baseline",
        requests=requests,
        cfg=cfg,
        evidence_dir=evidence_dir,
        request_timeout_s=request_timeout_s,
    )
    return finish_perf_screen(
        candidate_url,
        baseline=base,
        requests=requests,
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
    """Run one engine's closed-loop screen; persist rows; return engine metrics.

    Always applies ``cfg.num_requests`` so mock and Docker paths share the same
    deterministic subset (first N in arrival order).
    """
    subset = list(requests)[: cfg.num_requests]
    if not subset:
        raise EngineError("perf_screen: no requests to run")
    for req in subset:
        _require_text_prompt(req)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    wall, tokens, tps, rows = _run_engine(
        base_url,
        role=role,
        requests=subset,
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
