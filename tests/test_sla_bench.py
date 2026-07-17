"""Module C SLA bench: percentile helper, aggregation, recompute consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.lifecycle import EngineError
from bench.sla_bench import (
    aggregate_rep_metrics,
    percentile,
    recompute_sla_metrics,
    run_sla_bench,
    _engine_metrics_from_reps,
)
from bench.schemas import (
    SlaBenchConfig,
    SlaThresholds,
    TraceMeta,
    WorkloadTrace,
    TraceRequest,
    TraceSampling,
)
from bench.mock_engine import MockEngine, MockEngineConfig
from bench.validate import RequestValidationError


# --- percentile helper -------------------------------------------------------


def test_percentile_single_sample():
    assert percentile([5.0], 50) == 5.0
    assert percentile([5.0], 99) == 5.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_linear_interpolation():
    # numpy-default: position = (pct/100)*(n-1), interpolate.
    data = [0.0, 10.0, 20.0, 30.0]
    assert percentile(data, 0) == 0.0
    assert percentile(data, 50) == 15.0
    assert percentile(data, 100) == 30.0
    # p25 -> rank 0.75 -> 0 + (10-0)*0.75 = 7.5
    assert percentile(data, 25) == 7.5


def test_percentile_unsorted_input():
    assert percentile([30.0, 0.0, 20.0, 10.0], 50) == 15.0


# --- aggregation -------------------------------------------------------------


def _row(rid, ttft, itl, e2e, tokens=8):
    return {
        "request_id": rid,
        "ttft_ms": ttft,
        "itl_ms": itl,
        "e2e_ms": e2e,
        "completion_tokens": tokens,
        "error": None,
    }


def test_aggregate_pooled_itl():
    rows = [
        _row("a", 10.0, [1.0, 2.0], 20.0),
        _row("b", 30.0, [3.0], 40.0),
    ]
    m = aggregate_rep_metrics(rows, wall_s=1.0, p99_ttft_ms=100.0, p99_itl_ms=100.0)
    # Pooled ITL = [1,2,3]; p50=2.
    assert m["itl_ms"].p50 == 2.0
    assert m["ttft_ms"].p50 == 20.0
    assert m["e2e_ms"].p50 == 30.0
    assert m["requests_per_s"] == 2.0
    assert m["output_tokens_per_s"] == 16.0


def test_goodput_ratio_threshold():
    rows = [
        _row("a", 10.0, [1.0], 20.0),  # ttft ok
        _row("b", 500.0, [1.0], 600.0),  # ttft too high
    ]
    m = aggregate_rep_metrics(rows, wall_s=1.0, p99_ttft_ms=100.0, p99_itl_ms=100.0)
    assert m["sla_goodput_ratio"] == 0.5


def test_single_token_empty_itl_vacuous_ok():
    rows = [_row("a", 10.0, [], 20.0, tokens=1)]
    m = aggregate_rep_metrics(rows, wall_s=1.0, p99_ttft_ms=100.0, p99_itl_ms=100.0)
    assert m["itl_ms"].p99 == 0.0
    assert m["sla_goodput_ratio"] == 1.0


def test_multi_token_empty_itl_raises():
    rows = [_row("a", 10.0, [], 20.0, tokens=2)]
    with pytest.raises(EngineError, match="inter-token latency"):
        aggregate_rep_metrics(rows, wall_s=1.0, p99_ttft_ms=100.0, p99_itl_ms=100.0)


def test_engine_metrics_median_of_reps():
    rep1 = aggregate_rep_metrics(
        [_row("a", 10.0, [1.0], 20.0)], wall_s=1.0, p99_ttft_ms=1e9, p99_itl_ms=1e9
    )
    rep2 = aggregate_rep_metrics(
        [_row("a", 20.0, [1.0], 30.0)], wall_s=1.0, p99_ttft_ms=1e9, p99_itl_ms=1e9
    )
    rep3 = aggregate_rep_metrics(
        [_row("a", 30.0, [1.0], 40.0)], wall_s=1.0, p99_ttft_ms=1e9, p99_itl_ms=1e9
    )
    em = _engine_metrics_from_reps([rep1, rep2, rep3])
    assert em.ttft_ms.p50 == 20.0  # median of [10,20,30]


def test_rejects_token_ids_only(tmp_path: Path):
    trace = WorkloadTrace(
        schema_version=1,
        meta=TraceMeta(name="t"),
        requests=[
            TraceRequest(
                id="r-ids",
                arrival_offset_ms=0,
                max_tokens=4,
                sampling=TraceSampling(0.0, 1.0),
                prompt_token_ids=[1, 2, 3],
            ),
        ],
    )
    cfg = SlaBenchConfig(
        repetitions=1,
        warmup_requests=0,
        thresholds=SlaThresholds(p99_ttft_ms=1e9, p99_itl_ms=1e9),
    )
    with pytest.raises(RequestValidationError, match="text prompt"):
        run_sla_bench(
            "http://baseline",
            "http://candidate",
            trace=trace,
            cfg=cfg,
            evidence_dir=tmp_path,
        )


# --- recompute from requests.jsonl matches report ----------------------------


def test_recompute_matches_report(tmp_path: Path):
    trace = WorkloadTrace(
        schema_version=1,
        meta=TraceMeta(name="t"),
        requests=[
            TraceRequest(
                id="r1",
                arrival_offset_ms=0,
                max_tokens=6,
                sampling=TraceSampling(0.0, 1.0),
                prompt="hello",
            ),
            TraceRequest(
                id="r2",
                arrival_offset_ms=10,
                max_tokens=6,
                sampling=TraceSampling(0.0, 1.0),
                prompt="world",
            ),
        ],
    )
    cfg = SlaBenchConfig(
        repetitions=3,
        warmup_requests=0,
        thresholds=SlaThresholds(p99_ttft_ms=1e9, p99_itl_ms=1e9),
    )
    with (
        MockEngine(MockEngineConfig(model="b", token_latency_s=0.01)) as b,
        MockEngine(MockEngineConfig(model="c", token_latency_s=0.005)) as c,
    ):
        rep = run_sla_bench(
            b.base_url, c.base_url, trace=trace, cfg=cfg, evidence_dir=tmp_path
        )

    recomputed = recompute_sla_metrics(
        tmp_path / "candidate",
        p99_ttft_ms=cfg.thresholds.p99_ttft_ms,
        p99_itl_ms=cfg.thresholds.p99_itl_ms,
        repetitions=cfg.repetitions,
    )
    assert recomputed.ttft_ms.p50 == pytest.approx(rep.candidate.ttft_ms.p50)
    assert recomputed.ttft_ms.p99 == pytest.approx(rep.candidate.ttft_ms.p99)
    assert recomputed.itl_ms.p99 == pytest.approx(rep.candidate.itl_ms.p99)
    assert recomputed.e2e_ms.p99 == pytest.approx(rep.candidate.e2e_ms.p99)


def test_warmup_excluded_from_metrics(tmp_path: Path):
    trace = WorkloadTrace(
        schema_version=1,
        meta=TraceMeta(name="t"),
        requests=[
            TraceRequest(
                id="r1",
                arrival_offset_ms=0,
                max_tokens=4,
                sampling=TraceSampling(0.0, 1.0),
                prompt="hi",
            ),
        ],
    )
    cfg = SlaBenchConfig(
        repetitions=2,
        warmup_requests=1,
        thresholds=SlaThresholds(p99_ttft_ms=1e9, p99_itl_ms=1e9),
    )
    with (
        MockEngine(MockEngineConfig(model="b")) as b,
        MockEngine(MockEngineConfig(model="c")) as c,
    ):
        run_sla_bench(
            b.base_url, c.base_url, trace=trace, cfg=cfg, evidence_dir=tmp_path
        )
    # Warmup rows exist but flagged; measured reps are rep_1/rep_2.
    warm = tmp_path / "candidate" / "warmup" / "requests.jsonl"
    assert warm.is_file()
    warm_rows = [
        json.loads(line)
        for line in warm.read_text().splitlines()
        if line.strip() and '"_rep_meta"' not in line
    ]
    assert all(r["warmup"] for r in warm_rows)
    for n in (1, 2):
        rep_rows = [
            json.loads(line)
            for line in (tmp_path / "candidate" / f"rep_{n}" / "requests.jsonl")
            .read_text()
            .splitlines()
            if line.strip() and '"_rep_meta"' not in line
        ]
        assert all(not r["warmup"] for r in rep_rows)


def test_arrival_offsets_respected(tmp_path: Path):
    # A late-arriving request must not start before its offset.
    trace = WorkloadTrace(
        schema_version=1,
        meta=TraceMeta(name="t"),
        requests=[
            TraceRequest(
                id="early",
                arrival_offset_ms=0,
                max_tokens=2,
                sampling=TraceSampling(0.0, 1.0),
                prompt="a",
            ),
            TraceRequest(
                id="late",
                arrival_offset_ms=80,
                max_tokens=2,
                sampling=TraceSampling(0.0, 1.0),
                prompt="b",
            ),
        ],
    )
    cfg = SlaBenchConfig(
        repetitions=1, warmup_requests=0, thresholds=SlaThresholds(1e9, 1e9)
    )
    import time

    with (
        MockEngine(MockEngineConfig(model="b")) as b,
        MockEngine(MockEngineConfig(model="c")) as c,
    ):
        t0 = time.monotonic()
        run_sla_bench(
            b.base_url, c.base_url, trace=trace, cfg=cfg, evidence_dir=tmp_path
        )
        elapsed = time.monotonic() - t0
    # The 80ms-offset request means each engine's rep takes >= ~80ms.
    assert elapsed >= 0.08
