"""Module B perf screen: throughput math, verdict boundary, evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.lifecycle import EngineError
from bench.mock_engine import MockEngine, MockEngineConfig
from bench.perf_screen import run_perf_screen
from bench.schemas import PerfScreenConfig, TraceRequest, TraceSampling
from bench.validate import RequestValidationError


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


def test_engine_path_applies_num_requests_subset(tmp_path: Path):
    """Docker path calls run_perf_screen_engine directly; it must still subset."""
    from bench.perf_screen import run_perf_screen_engine

    reqs = [_req(1), _req(2), _req(3), _req(4)]
    cfg = PerfScreenConfig(num_requests=2, concurrency=1, min_throughput_ratio=0.5)
    with MockEngine(MockEngineConfig(model="b")) as eng:
        metrics = run_perf_screen_engine(
            eng.base_url,
            role="baseline",
            requests=reqs,
            cfg=cfg,
            evidence_dir=tmp_path,
        )
    assert metrics["num_requests"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "baseline_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert {r["request_id"] for r in rows} == {"r-1", "r-2"}


def test_perf_screen_disables_logprobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import time

    seen: list[object] = []

    def capture_slow(url, **kwargs):
        seen.append(kwargs.get("logprobs", "MISSING"))
        time.sleep(0.002)
        return {
            "choices": [{"text": "x", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("bench.perf_screen.post_completion", capture_slow)
    cfg = PerfScreenConfig(num_requests=1, concurrency=1, min_throughput_ratio=0.5)
    run_perf_screen(
        "http://baseline",
        "http://candidate",
        requests=[_req(1)],
        cfg=cfg,
        evidence_dir=tmp_path,
    )
    assert seen
    assert all(v is None for v in seen)


def test_wall_bounds_keep_earliest_send_and_latest_done():
    """Lock order must not replace an earlier send or a later done timestamp."""
    from bench.perf_screen import _earliest, _latest

    assert _earliest(None, 5.0) == 5.0
    assert _earliest(5.0, 3.0) == 3.0
    assert _earliest(3.0, 4.0) == 3.0
    assert _latest(None, 5.0) == 5.0
    assert _latest(5.0, 3.0) == 5.0
    assert _latest(5.0, 7.0) == 7.0


def test_rejects_token_ids_only(tmp_path: Path):
    reqs = [
        TraceRequest(
            id="r-ids",
            arrival_offset_ms=0,
            max_tokens=4,
            sampling=TraceSampling(temperature=0.0, top_p=1.0),
            prompt_token_ids=[1, 2, 3],
        )
    ]
    cfg = PerfScreenConfig(num_requests=1, concurrency=1, min_throughput_ratio=0.5)
    with pytest.raises(RequestValidationError, match="text prompt"):
        run_perf_screen(
            "http://baseline",
            "http://candidate",
            requests=reqs,
            cfg=cfg,
            evidence_dir=tmp_path,
        )


def test_concurrent_throughput_uses_wall_not_latency_sum(tmp_path: Path):
    """With concurrency > 1, tok/s must use first-send→last-done wall time.

    Summing per-request latencies double-counts parallel work and understates
    throughput. Four equal requests at concurrency=2 should finish in ~2 waves,
    so wall ≈ half the latency sum (within slack for thread/HTTP overhead).
    """
    from bench.perf_screen import run_perf_screen_engine

    reqs = [_req(i, max_tokens=4) for i in range(4)]
    cfg = PerfScreenConfig(num_requests=4, concurrency=2, min_throughput_ratio=0.5)
    with MockEngine(MockEngineConfig(model="b", token_latency_s=0.02)) as eng:
        metrics = run_perf_screen_engine(
            eng.base_url,
            role="baseline",
            requests=reqs,
            cfg=cfg,
            evidence_dir=tmp_path,
        )
    wall = float(metrics["wall_s"])
    latency_sum = sum(
        float(json.loads(line)["latency_s"])
        for line in (tmp_path / "baseline_rows.jsonl").read_text().splitlines()
        if line.strip()
    )
    assert latency_sum > 0
    # Parallelism must make wall materially shorter than the latency sum.
    assert wall < 0.75 * latency_sum
    expected_tps = metrics["completion_tokens"] / wall
    assert metrics["output_tokens_per_s"] == pytest.approx(expected_tps)


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
