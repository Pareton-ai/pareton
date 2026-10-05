"""Incentive, scheduling, historical-contract, and eligibility regression tests."""

import copy
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from bench.concurrency import (
    TIERS,
    WEIGHTED_RULE,
    request_groups,
    tier_completion_metrics,
)
from bench.http import StreamResult, post_completion_stream
from bench.lifecycle import EngineError
from bench.sampler import SamplerError, parse_sampling_rule
from bench.schemas import TraceRequest, TraceSampling
from bench.score import PromptTiming, score_candidate
from bench.sla_bench import _replay_concurrent
from campaign.models import validate_scoring_rule

pytestmark = pytest.mark.unit


def requests(n=8):
    return [
        TraceRequest(
            f"{tier}-{i}",
            0,
            2,
            TraceSampling(0, 1),
            prompt=f"{tier}-{i}",
            input_length_group=tier,
        )
        for tier in TIERS
        for i in range(n)
    ]


def tier_stats(seconds=10, n=8):
    return {
        tier: {
            "completion_s": seconds,
            "request_ids": [f"{tier}-{i}" for i in range(n)],
        }
        for tier in TIERS
    }


def timings(n=8, seconds=1):
    return {
        r.id: PromptTiming(seconds / 2, [seconds / 2], 2, "length") for r in requests(n)
    }


def test_weighted_score_and_failure_penalty():
    rule = {"name": WEIGHTED_RULE, "failure_penalty": 0.4}
    baseline, candidate = timings(), timings(seconds=0.5)
    kwargs = dict(
        baseline=baseline,
        candidate=candidate,
        baseline_tiers=tier_stats(),
        candidate_tiers=tier_stats(5),
    )
    assert score_candidate(rule, **kwargs).score == 0.5
    del candidate["2k-0"]
    result = score_candidate(rule, **kwargs)
    assert result.breakdown["failed_requests"] == 1
    assert result.breakdown["failure_rate"] == 1 / 32
    assert result.breakdown["penalty"] == 0.4 / 32
    assert result.score == -0.4 / 32  # failures cannot buy positive capacity credit
    assert score_candidate({"name": WEIGHTED_RULE}, **kwargs).score == 0


@pytest.mark.parametrize("batch_baseline", [False, True])
@pytest.mark.parametrize("gaps", [[], [0.1]])
def test_batched_tokens_earn_tier_speedup_without_failure(batch_baseline, gaps):
    baseline = {
        rid: PromptTiming(1, [] if batch_baseline else [1] * 7, 8, "length")
        for rid in timings()
    }
    candidate = {rid: PromptTiming(0.5, gaps, 8, "length") for rid in baseline}
    result = score_candidate(
        {"name": WEIGHTED_RULE, "failure_penalty": 0.1},
        baseline=baseline,
        candidate=candidate,
        baseline_tiers=tier_stats(),
        candidate_tiers=tier_stats(5),
    )
    assert result.score == 0.5
    assert result.breakdown["failed_requests"] == 0
    assert result.breakdown["penalty"] == 0
    assert all(
        p.reason == "insufficient timing" and not p.candidate_failed
        for p in result.per_prompt
    )


@pytest.mark.parametrize(
    "tokens,finish", [(1, "length"), (3, "length"), (2, None), (2, "error")]
)
def test_v5_completion_failures_still_cap_and_penalize(tokens, finish):
    candidate = timings()
    candidate["2k-0"] = PromptTiming(0.1, [], tokens, finish)
    result = score_candidate(
        {"name": WEIGHTED_RULE, "failure_penalty": 0.1},
        baseline=timings(),
        candidate=candidate,
        baseline_tiers=tier_stats(),
        candidate_tiers=tier_stats(5),
    )
    assert result.breakdown["failed_requests"] == 1
    assert result.score == -0.1 / 32


@pytest.mark.parametrize(
    "timeout", [0, -1, True, "600", None, float("inf"), float("nan")]
)
def test_v5_rejects_invalid_request_timeout(timeout):
    from test_longform_sampling import rule

    v5 = rule(algo_version=5, request_concurrency=4, request_timeout_s=timeout)
    del v5["request_interval_ms"]
    with pytest.raises(SamplerError, match="request_timeout_s"):
        parse_sampling_rule(v5)


def test_parked_requests_lose_despite_better_median_latency():
    base = timings(seconds=10)
    candidate = timings(seconds=4)
    candidate["2k-0"] = PromptTiming(60, [60], 2, "length")
    legacy = score_candidate(
        {"name": "median_e2e_speedup"}, baseline=base, candidate=candidate
    )
    new = score_candidate(
        {"name": WEIGHTED_RULE},
        baseline=base,
        candidate=candidate,
        baseline_tiers=tier_stats(80),
        candidate_tiers=tier_stats(120),
    )
    assert legacy.score == 0.6
    assert new.score == -0.5


def test_weights_and_longer_durations_are_monotonic():
    rule = validate_scoring_rule(
        {"name": WEIGHTED_RULE, "tier_weights": dict(zip(TIERS, [0.1, 0.2, 0.3, 0.4]))}
    )
    b, c = tier_stats(), tier_stats(5)
    c["16k"]["completion_s"] = 20
    result = score_candidate(
        rule,
        baseline=timings(),
        candidate=timings(),
        baseline_tiers=b,
        candidate_tiers=c,
    )
    assert result.score == pytest.approx(-0.1)
    for tier in TIERS:
        delayed = copy.deepcopy(c)
        delayed[tier]["completion_s"] += 1
        assert (
            score_candidate(
                rule,
                baseline=timings(),
                candidate=timings(),
                baseline_tiers=b,
                candidate_tiers=delayed,
            ).score
            < result.score
        )


@pytest.mark.parametrize(
    "weights",
    [
        [0.25] * 4,
        {"2k": 1},
        dict.fromkeys(TIERS, 0.3),
        dict.fromkeys(TIERS, float("nan")),
        dict(zip(TIERS, [True, 0, 0, 0])),
    ],
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError, match="tier_weights"):
        validate_scoring_rule({"name": WEIGHTED_RULE, "tier_weights": weights})


@pytest.mark.parametrize("coefficient", [-1, True, float("inf")])
def test_invalid_failure_penalty(coefficient):
    with pytest.raises(ValueError, match="failure_penalty"):
        validate_scoring_rule({"name": WEIGHTED_RULE, "failure_penalty": coefficient})


@pytest.mark.parametrize(
    "c,sizes", [(1, [8] * 4), (4, [8] * 4), (8, [8] * 4), (16, [16] * 2), (32, [32])]
)
def test_group_membership_frozen(c, sizes):
    groups = request_groups(requests(), c)
    assert list(map(len, groups)) == sizes
    assert [r.input_length_group for r in groups[0][: max(1, c // 8)]] == list(
        TIERS[: max(1, c // 8)]
    )
    filtered = request_groups([r for r in requests() if r.id != "2k-0"], c)
    assert len(filtered[0]) == sizes[0] - 1
    with pytest.raises(EngineError, match="nonempty"):
        request_groups([r for r in requests() if r.input_length_group != "2k"], c)


def test_immediate_refill_and_full_protocol_occupancy(monkeypatch):
    parked = threading.Event()
    refill = threading.Event()
    lock = threading.Lock()
    active, peak = 0, 0

    def post(url, *, prompt, **kwargs):
        nonlocal active, peak
        started = time.monotonic()
        with lock:
            active += 1
            peak = max(peak, active)
        if prompt == "2k-0":
            assert parked.wait(2)
        elif prompt == "2k-4":
            refill.set()
            parked.set()
        time.sleep(0.002)
        finished = time.monotonic()
        with lock:
            active -= 1
        return StreamResult(
            "ok",
            "length",
            2,
            0.001,
            [0.001],
            0.002,
            dispatch_monotonic_s=started,
            completion_monotonic_s=started + 0.001,
            protocol_completion_monotonic_s=finished,
        )

    monkeypatch.setattr("bench.sla_bench.post_completion_stream", post)
    rows, _, errs = _replay_concurrent(
        "unused",
        requests(),
        role="test",
        rep=1,
        is_warmup=False,
        timeout_s=3,
        request_concurrency=4,
    )
    assert not errs and refill.is_set()
    assert peak <= 4
    assert len(rows) == 32
    metrics = tier_completion_metrics(rows, 1)
    assert all(metrics[t]["completion_s"] > 0 for t in TIERS)
    first_group_end = max(
        r["protocol_completion_offset_ms"]
        for r in rows
        if r["input_length_group"] == "2k"
    )
    assert (
        min(r["admission_offset_ms"] for r in rows if r["input_length_group"] == "4k")
        >= first_group_end
    )
    assert all(
        r["protocol_completion_offset_ms"] > r["completion_offset_ms"] for r in rows
    )


def test_hung_headers_hit_absolute_deadline_without_refill(monkeypatch):
    release = threading.Event()
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["prompt"])
        release.wait(2)
        raise EngineError("closed")

    monkeypatch.setattr("bench.sla_bench.post_completion_stream", post)
    try:
        rows, _, errs = _replay_concurrent(
            "unused",
            requests(),
            role="test",
            rep=1,
            is_warmup=False,
            timeout_s=0.02,
            request_concurrency=1,
        )
        assert errs and len(calls) == 1
        assert "deadline" in rows[0]["error"]
    finally:
        release.set()


def test_replay_deadline_stops_parked_worker_before_refill(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls, workers = [], []

    class ParkedWorker(threading.Thread):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            workers.append(self)

        def join(self, timeout=None):
            # Model a timed-out join while _fire is still in progress.
            assert started.wait(2)

    def fire(url, req, **kwargs):
        calls.append(req.id)
        started.set()
        assert release.wait(2)

    monkeypatch.setattr("bench.sla_bench.threading.Thread", ParkedWorker)
    monkeypatch.setattr("bench.sla_bench._fire", fire)
    try:
        with pytest.raises(EngineError, match="absolute replay deadline"):
            _replay_concurrent(
                "unused",
                requests(),
                role="test",
                rep=1,
                is_warmup=False,
                timeout_s=1,
                request_concurrency=1,
            )
    finally:
        release.set()
        for worker in workers:
            super(ParkedWorker, worker).join(2)
            assert not worker.is_alive()
    assert calls == ["2k-0"]


def test_done_timestamp_is_separate_from_last_token(monkeypatch):
    from test_http import _FakeResp, _sse

    class DelayedDone(_FakeResp):
        def __iter__(self):
            for line in super().__iter__():
                if b"[DONE]" in line:
                    time.sleep(0.02)
                yield line

    monkeypatch.setattr(
        "bench.http.urlopen",
        lambda *a, **kw: DelayedDone(
            _sse(
                {
                    "choices": [{"text": "x", "finish_reason": "length"}],
                    "usage": {"completion_tokens": 1},
                }
            )
        ),
    )
    result = post_completion_stream("http://test", prompt="test", max_tokens=1)
    assert (
        result.protocol_completion_monotonic_s - result.completion_monotonic_s >= 0.02
    )


def test_v5_receipt_roundtrip_and_old_rule_unchanged(tmp_path):
    from test_longform_sampling import fields, formatter, row, rule, sample

    from bench.longform import qualification_contract
    from bench.validate import validate_workload_trace_dict
    from worker.round_job import materialize_round_trace

    old = rule()
    assert parse_sampling_rule(old)["request_interval_ms"] == 2
    v5 = {k: v for k, v in old.items() if k != "request_interval_ms"}
    v5.update(algo_version=5, request_concurrency=4, output_tokens=6)
    assert parse_sampling_rule(v5)["request_timeout_s"] == 600
    v5["request_timeout_s"] = 480
    sampled = sample(rule=v5, prompt_formatter=formatter(v5))
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert trace.meta.sampling["request_concurrency"] == 4
    assert trace.meta.sampling["request_timeout_s"] == 480
    assert sampled.receipt["request_timeout_s"] == 480
    assert qualification_contract(
        parse_sampling_rule(v5), {}, {}
    ) != qualification_contract(
        parse_sampling_rule({**v5, "request_timeout_s": 600}), {}, {}
    )
    path = materialize_round_trace(
        {"sampled_trace_sha256": sampled.sha256, "sampling_receipt": sampled.receipt},
        SimpleNamespace(bench=fields()["bench"], engine=fields()["engine"]),
        tmp_path,
        row_fetcher=row,
        prompt_formatter=formatter(v5),
    )
    assert path.read_bytes() == sampled.body
    for invalid in (0, float("inf"), None):
        broken = json.loads(sampled.body)
        broken["meta"]["sampling"]["request_timeout_s"] = invalid
        with pytest.raises(ValueError, match="request_timeout_s"):
            validate_workload_trace_dict(broken)
    assert "request_interval_ms" not in sampled.receipt
    replay = sample(
        rule=v5, prompt_formatter=formatter(v5), sampling_receipt=sampled.receipt
    )
    assert replay.body == sampled.body
    with pytest.raises(SamplerError, match="retired"):
        parse_sampling_rule({**v5, "request_interval_ms": 0})
    with pytest.raises(SamplerError, match="require algo_version 5"):
        parse_sampling_rule({**old, "request_concurrency": 4})


def test_baseline_exclusions_frozen_before_scored_runs(monkeypatch, tmp_path):
    from test_bench_cli import SAMPLE_REQUEST

    from bench import main as bm
    from bench.output import OutputLayout
    from bench.schemas import BenchRequest, TraceMeta, WorkloadTrace

    raw = json.loads(SAMPLE_REQUEST.read_text())
    raw["mode"] = "sla_bench"
    raw["scoring_rule"] = {"name": WEIGHTED_RULE}
    req = BenchRequest.from_dict(raw)
    source = requests()
    trace = WorkloadTrace(
        1,
        TraceMeta(
            "v5",
            sampling={
                "algo_version": 5,
                "request_concurrency": 4,
                "request_timeout_s": 480,
                "output_tokens": 2,
                "min_output_tokens": 2,
            },
        ),
        source,
    )
    calls = []

    class Provider:
        @contextmanager
        def start(self, start, phase):
            yield start.role

    def replay(url, *, requests, **kwargs):
        assert kwargs["request_timeout_s"] == 480
        calls.append((url, list(requests)))
        return SimpleNamespace(
            result=SimpleNamespace(role=url, timings={}, cross_rep_variance={}),
            excluded_prompts={},
        )

    def drops(trace, replay, *, dropped):
        result = dict(dropped)
        if replay.result.role == "qualification-1":
            result["2k-0"] = "degenerate"
        if replay.result.role == "qualification-2":
            result["4k-0"] = "degenerate"
        return result

    monkeypatch.setattr(bm, "run_sla_engine", replay)
    monkeypatch.setattr(bm, "baseline_prompt_drops", drops)
    monkeypatch.setattr(
        "bench.workload_preflight.validate_engine_workload", lambda *a, **kw: None
    )
    # run_round replaces the baseline dataclass to attach exclusion evidence.
    from bench.sla_bench import EngineReplay

    def replay_dataclass(url, **kwargs):
        result = replay(url, **kwargs)
        return EngineReplay(result.result, {}, {})

    monkeypatch.setattr(bm, "run_sla_engine", replay_dataclass)
    layout = OutputLayout(tmp_path)
    layout.prepare()
    bm.run_round(req=req, provider=Provider(), prompts=[], trace=trace, layout=layout)
    assert [role for role, _ in calls[:4]] == [
        "qualification-1",
        "qualification-2",
        "baseline",
        "baseline-drift",
    ]
    assert all(len(rs) == 32 for _, rs in calls[:2])
    assert all(len(rs) == 30 for _, rs in calls[2:])
    assert all(
        r.id not in {"2k-0", "4k-0"} and r.sampling.ignore_eos and r.max_tokens == 2
        for _, rs in calls[2:]
        for r in rs
    )
    frozen = json.loads((layout.sla_bench_dir / "eligible_workload.json").read_text())
    assert set(frozen["excluded_prompts"]) == {"2k-0", "4k-0"}


@pytest.mark.parametrize("c", [1, 4, 8, 16, 32])
def test_real_http_replay_persists_complete_tier_evidence(tmp_path, c):
    from bench.mock_engine import MockEngine, MockEngineConfig
    from bench.schemas import SlaBenchConfig, SlaThresholds
    from bench.sla_bench import run_sla_engine

    with MockEngine(MockEngineConfig(model="c", token_latency_s=0.001)) as engine:
        replay = run_sla_engine(
            engine.base_url,
            role="candidate",
            requests=requests(),
            cfg=SlaBenchConfig(2, SlaThresholds(1e9, 1e9)),
            evidence_dir=tmp_path,
            request_concurrency=c,
        )
    assert set(replay.result.tier_completion) == set(TIERS)
    assert all(
        len(t["request_ids"]) == 8 and len(t["repetitions_s"]) == 2
        for t in replay.result.tier_completion.values()
    )
    assert all(o["observed_peak"] <= c for o in replay.result.concurrency_observations)
    data = [
        json.loads(line)
        for line in (tmp_path / "candidate/rep_1/requests.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(data) == 33
    assert all(
        "protocol_completion_offset_ms" in r for r in data if not r.get("_rep_meta")
    )


def test_failed_concurrent_warmup_preserves_error_evidence(monkeypatch, tmp_path):
    from bench.schemas import SlaBenchConfig, SlaThresholds
    from bench.sla_bench import run_sla_engine

    error = "connection reset by peer"
    row = {"request_id": "2k-0", "error": error}
    monkeypatch.setattr(
        "bench.sla_bench._replay_concurrent", lambda *a, **kw: ([row], 0.5, [error])
    )
    with pytest.raises(EngineError, match=f"concurrency warmup failed: {error}"):
        run_sla_engine(
            "unused",
            role="candidate",
            requests=requests(),
            cfg=SlaBenchConfig(2, SlaThresholds(1e9, 1e9)),
            evidence_dir=tmp_path,
            request_concurrency=32,
        )
    evidence = (tmp_path / "candidate/warmup/requests.jsonl").read_text().splitlines()
    assert json.loads(evidence[0]) == row


def test_reference_drift_cannot_cancel_between_tiers():
    from bench.main import baseline_drift

    b, c = tier_stats(), tier_stats()
    c["2k"]["completion_s"] = 5
    c["16k"]["completion_s"] = 15
    baseline = SimpleNamespace(
        result=SimpleNamespace(timings=timings(), tier_completion=b),
        excluded_prompts={},
    )
    drift = SimpleNamespace(
        result=SimpleNamespace(timings=timings(), tier_completion=c)
    )
    req = SimpleNamespace(scoring_rule={"name": WEIGHTED_RULE, "failure_penalty": 0.8})
    assert baseline_drift(req, baseline, drift) == 0.5


@pytest.mark.parametrize("batched", [False, True])
def test_full_v5_round_scores_fixed_work_and_verifies_baseline(
    monkeypatch, tmp_path, batched
):
    from test_bench_cli import SAMPLE_REQUEST

    from bench import main as bm
    from bench.mock_engine import MockEngine, MockEngineConfig
    from bench.output import OutputLayout
    from bench.schemas import BenchRequest, TraceMeta, WorkloadTrace

    if batched:
        from bench.mock_engine import _Handler

        write_sse = _Handler._write_sse

        def write_batch(handler, chunk):
            handler.pending_text = (
                getattr(handler, "pending_text", "") + chunk["choices"][0]["text"]
            )
            if "usage" in chunk:
                chunk["choices"][0]["text"] = handler.pending_text
                write_sse(handler, chunk)

        monkeypatch.setattr(_Handler, "_write_sse", write_batch)

    raw = json.loads(SAMPLE_REQUEST.read_text())
    raw["scoring_rule"] = {"name": WEIGHTED_RULE, "failure_penalty": 0.1}
    raw["sla_bench"]["repetitions"] = 1
    req = BenchRequest.from_dict(raw)
    trace = WorkloadTrace(
        1,
        TraceMeta(
            "v5",
            sampling={
                "algo_version": 5,
                "request_concurrency": 4,
                "request_timeout_s": 480,
                "output_tokens": 2,
                "min_output_tokens": 2,
            },
        ),
        requests(),
    )
    starts = []

    class Provider:
        @contextmanager
        def start(self, start, phase):
            starts.append(start.role)
            with MockEngine(
                MockEngineConfig(model=start.role, token_latency_s=0.001)
            ) as engine:
                yield engine.base_url

    monkeypatch.setattr(
        "bench.workload_preflight.validate_engine_workload", lambda *a, **kw: None
    )
    layout = OutputLayout(tmp_path)
    layout.prepare()
    baseline, drift, runs, correctness = bm.run_round(
        req=req, provider=Provider(), prompts=[], trace=trace, layout=layout
    )
    assert starts[:4] == [
        "qualification-1",
        "qualification-2",
        "baseline",
        "baseline-drift",
    ]
    assert starts[-1] == "scorer"
    assert (layout.correctness_dir / "baseline.jsonl").is_file()
    assert all(report.verdict == "pass" for report in correctness.values())
    entries = bm._build_entries(
        req=req,
        baseline=baseline,
        runs=runs,
        correctness=correctness,
        digests=[],
        mock_engine=True,
    )
    assert entries and all(entry.status == "scored" for entry in entries)
    assert entries[0].score_report["score_breakdown"]["failed_requests"] == 0
    if batched:
        assert all(not t.itl_s for t in baseline.result.timings.values())
        assert all(
            o["token_timing_unavailable_requests"]
            for o in baseline.result.concurrency_observations
        )
    assert entries[0].score_report["score_breakdown"]["failure_penalty"] == 0.1
    assert (
        entries[0].sla.eligible_workload_sha256
        == baseline.result.eligible_workload_sha256
    )
    assert (
        drift.result.eligible_workload_sha256
        == baseline.result.eligible_workload_sha256
    )
    assert all(t.completion_tokens == 2 for t in baseline.result.timings.values())


def test_fixed_budget_checks_degenerate_sibling_repetitions(tmp_path):
    from dataclasses import replace

    from test_correctness import LOOP_TEXT, PROSE_TEXT, _captured, _cfg

    from bench.correctness import grade_candidate
    from bench.mock_engine import MockEngine, MockEngineConfig

    output = replace(
        _captured("r1", "Hello world", PROSE_TEXT, ignore_eos=True),
        output_samples=(PROSE_TEXT, LOOP_TEXT),
    )
    with MockEngine(MockEngineConfig(model="scorer")) as scorer:
        report = grade_candidate(
            scorer.base_url,
            [output],
            cfg=_cfg(num_prompts=1),
            evidence_path=tmp_path / "fixed.jsonl",
            strict_fixed_output=True,
        )
    assert report.verdict == "fail_correctness"
    evidence = json.loads((tmp_path / "fixed.jsonl").read_text())
    assert evidence["ignore_eos"] is True
    assert len(evidence["repetition_degeneracy"]) == 2
    assert evidence["repetition_degeneracy"][1]["degenerate"]


@pytest.mark.parametrize("budget, succeeds", [(120, False), (600, True)])
def test_long_completion_uses_configured_absolute_deadline(
    monkeypatch, budget, succeeds
):
    from test_http import _FakeResp, _sse

    clock = [0.0]

    class LongResponse(_FakeResp):
        fp = SimpleNamespace(
            raw=SimpleNamespace(_sock=SimpleNamespace(shutdown=lambda *_: None))
        )

        def __iter__(self):
            clock[0] = 150.0
            yield from super().__iter__()

    def open_response(req, timeout):
        assert timeout == budget
        return LongResponse(
            _sse(
                {
                    "choices": [{"text": "two tokens", "finish_reason": "length"}],
                    "usage": {"completion_tokens": 2},
                }
            )
        )

    monkeypatch.setattr("bench.http.time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr("bench.http.urlopen", open_response)

    def complete():
        return post_completion_stream(
            "http://test",
            prompt="prompt",
            max_tokens=2,
            timeout=budget,
            absolute_deadline_s=budget,
            require_token_timing=False,
        )

    if succeeds:
        assert complete().e2e_s == 150
    else:
        with pytest.raises(EngineError, match="absolute deadline"):
            complete()
