"""Module B perf screen: throughput math, verdict boundary, evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.lifecycle import EngineError
from bench.mock_engine import MockEngine, MockEngineConfig
from bench.perf_screen import run_perf_screen
from bench.schemas import PerfScreenConfig, TraceRequest, TraceSampling


def _req(i: int, max_tokens: int = 8) -> TraceRequest:
    return TraceRequest(
        id=f"r-{i}",
        arrival_offset_ms=0,
        max_tokens=max_tokens,
        sampling=TraceSampling(temperature=0.0, top_p=1.0),
        prompt=f"prompt {i}",
    )


def test_all_requests_issued_and_throughput_math(tmp_path: Path):
    reqs = [_req(1), _req(2), _req(3)]
    cfg = PerfScreenConfig(num_requests=3, concurrency=1, min_throughput_ratio=0.5)
    with (
        MockEngine(MockEngineConfig(model="b", token_latency_s=0.0)) as b,
        MockEngine(MockEngineConfig(model="c", token_latency_s=0.0)) as c,
    ):
        rep = run_perf_screen(
            b.base_url, c.base_url, requests=reqs, cfg=cfg, evidence_dir=tmp_path
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "perf_screen.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # 3 requests per engine.
    assert len(rows) == 6
    assert sum(1 for r in rows if r["engine_role"] == "baseline") == 3
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["completion_tokens"] == 8 for r in rows)
    assert rep.baseline_output_tokens_per_s > 0
    assert rep.candidate_output_tokens_per_s > 0
    assert rep.throughput_ratio > 0


def test_subset_uses_first_num_requests(tmp_path: Path):
    reqs = [_req(1), _req(2), _req(3), _req(4)]
    cfg = PerfScreenConfig(num_requests=2, concurrency=1, min_throughput_ratio=0.5)
    with (
        MockEngine(MockEngineConfig(model="b")) as b,
        MockEngine(MockEngineConfig(model="c")) as c,
    ):
        run_perf_screen(
            b.base_url, c.base_url, requests=reqs, cfg=cfg, evidence_dir=tmp_path
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "perf_screen.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ids = {r["request_id"] for r in rows}
    assert ids == {"r-1", "r-2"}


def test_verdict_pass_at_ratio_boundary(tmp_path: Path):
    # Candidate strictly faster -> ratio > 1 -> pass at min_throughput_ratio=1.0.
    cfg = PerfScreenConfig(num_requests=2, concurrency=1, min_throughput_ratio=1.0)
    with (
        MockEngine(MockEngineConfig(model="b", token_latency_s=0.02)) as b,
        MockEngine(MockEngineConfig(model="c", token_latency_s=0.005)) as c,
    ):
        rep = run_perf_screen(
            b.base_url,
            c.base_url,
            requests=[_req(1), _req(2)],
            cfg=cfg,
            evidence_dir=tmp_path,
        )
    assert rep.throughput_ratio >= 1.0
    assert rep.verdict == "pass"


def test_verdict_fail_when_candidate_slower(tmp_path: Path):
    cfg = PerfScreenConfig(num_requests=2, concurrency=1, min_throughput_ratio=1.0)
    with (
        MockEngine(MockEngineConfig(model="b", token_latency_s=0.005)) as b,
        MockEngine(MockEngineConfig(model="c", token_latency_s=0.02)) as c,
    ):
        rep = run_perf_screen(
            b.base_url,
            c.base_url,
            requests=[_req(1), _req(2)],
            cfg=cfg,
            evidence_dir=tmp_path,
        )
    assert rep.throughput_ratio < 1.0
    assert rep.verdict == "fail_perf_screen"


def test_http_error_raises_engine_error(tmp_path: Path):
    cfg = PerfScreenConfig(num_requests=1, concurrency=1, min_throughput_ratio=0.5)
    # Unreachable engine -> EngineError.
    with pytest.raises(EngineError):
        run_perf_screen(
            "http://127.0.0.1:9",
            "http://127.0.0.1:9",
            requests=[_req(1)],
            cfg=cfg,
            evidence_dir=tmp_path,
            request_timeout_s=1.0,
        )
