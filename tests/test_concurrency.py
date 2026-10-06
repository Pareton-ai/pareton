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


@pytest.mark.parametrize("tokens,finish", [(1, "length"), (2, None), (2, "error")])
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


@pytest.mark.parametrize("tokens", [89, 90, 91, 100, 110])
def test_natural_output_minimum_and_penalty(tokens):
    baseline = {rid: PromptTiming(1, [], 100, "stop") for rid in timings()}
    candidate = dict(baseline)
    candidate["2k-0"] = PromptTiming(0.5, [], tokens, "stop")
    result = score_candidate(
        {"name": WEIGHTED_RULE, "failure_penalty": 0.4},
        baseline=baseline,
        candidate=candidate,
        baseline_tiers=tier_stats(),
        candidate_tiers=tier_stats(5),
    )
    failed = tokens < 90
    assert result.breakdown["failed_requests"] == int(failed)
    assert result.breakdown["penalty"] == (0.4 / 32 if failed else 0)
    assert result.score == (-0.4 / 32 if failed else 0.5)


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
        {"32k": 1},
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
    v5.update(algo_version=5, request_concurrency=4)
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


def test_baseline_exclusions_union_before_candidates(monkeypatch, tmp_path):
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
        from bench.sla_bench import EngineReplay

        assert kwargs["request_timeout_s"] == 480
        calls.append((url, list(requests)))
        rows = []
        for rep in range(1, req.sla_bench.repetitions + 1):
            rep_rows = [
                {
                    "request_id": r.id,
                    "rep": rep,
                    "input_length_group": r.input_length_group,
                    "group_start_offset_ms": 20000 * TIERS.index(r.input_length_group),
                    "protocol_completion_offset_ms": 20000
                    * TIERS.index(r.input_length_group)
                    + (90000 if r.id in {"2k-0", "4k-0"} else 10000),
                }
                for r in requests
            ]
            path = layout.sla_bench_dir / url / f"rep_{rep}" / "requests.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(row) for row in rep_rows))
            rows.extend(rep_rows)
        return EngineReplay(
            SimpleNamespace(
                role=url,
                timings={r.id: timings()[r.id] for r in requests},
                tier_completion=tier_completion_metrics(
                    rows, req.sla_bench.repetitions
                ),
                cross_rep_variance={},
            ),
            {},
            {},
        )

    def drops(trace, replay, *, dropped):
        result = dict(dropped)
        if replay.result.role == "baseline":
            result["2k-0"] = "degenerate"
        if replay.result.role == "baseline-drift":
            result["4k-0"] = "degenerate"
        return result

    monkeypatch.setattr(bm, "run_sla_engine", replay)
    monkeypatch.setattr(bm, "baseline_prompt_drops", drops)
    monkeypatch.setattr(
        "bench.workload_preflight.validate_engine_workload", lambda *a, **kw: None
    )
    layout = OutputLayout(tmp_path)
    layout.prepare()
    baseline, drift, _, _ = bm.run_round(
        req=req, provider=Provider(), prompts=[], trace=trace, layout=layout
    )
    assert [role for role, _ in calls[:2]] == [
        "baseline",
        "baseline-drift",
    ]
    assert all(len(rs) == 32 for _, rs in calls[:2])
    assert all(len(rs) == 30 for _, rs in calls[2:])
    assert all(
        r.id not in {"2k-0", "4k-0"} and not r.sampling.ignore_eos and r.max_tokens == 2
        for _, rs in calls[2:]
        for r in rs
    )
    assert baseline.excluded_prompts == {"2k-0": "degenerate", "4k-0": "degenerate"}
    assert bm.baseline_drift(req, baseline, drift) == 0
    for reference in (baseline, drift):
        assert len(reference.result.timings) == 30
        assert all(
            t["completion_s"] == 10 for t in reference.result.tier_completion.values()
        )
        assert all(
            not set(t["request_ids"]) & {"2k-0", "4k-0"}
            for t in reference.result.tier_completion.values()
        )


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
@pytest.mark.parametrize("selected_tiers", [None, ["8k", "16k"]])
def test_full_v5_round_scores_natural_outputs_and_verifies_baseline(
    monkeypatch, tmp_path, batched, selected_tiers
):
    from dataclasses import replace

    from test_bench_cli import SAMPLE_REQUEST

    from bench import main as bm
    from bench.mock_engine import MockEngine, MockEngineConfig, _Handler
    from bench.output import OutputLayout
    from bench.schemas import BenchRequest, TraceMeta, WorkloadTrace

    stream = _Handler._stream_completion
    write_sse = _Handler._write_sse

    def natural_stop(handler, *, max_tokens, **kwargs):
        assert max_tokens == 8  # Preserve the ceiling, stop naturally at two.
        stream(handler, max_tokens=2, **kwargs)

    def write_output(handler, chunk):
        if "usage" in chunk:
            chunk["choices"][0]["finish_reason"] = "stop"
        if batched:
            handler.pending_text = (
                getattr(handler, "pending_text", "") + chunk["choices"][0]["text"]
            )
            if "usage" not in chunk:
                return
            chunk["choices"][0]["text"] = handler.pending_text
        write_sse(handler, chunk)

    monkeypatch.setattr(_Handler, "_stream_completion", natural_stop)
    monkeypatch.setattr(_Handler, "_write_sse", write_output)

    raw = json.loads(SAMPLE_REQUEST.read_text())
    raw["engines"]["candidates"] = [raw["engines"]["candidates"][0]] * 6
    raw["leader_candidate_index"] = 0
    raw["scoring_rule"] = {"name": WEIGHTED_RULE, "failure_penalty": 0.1}
    raw["sla_bench"]["repetitions"] = 1
    if selected_tiers:
        raw["scoring_rule"]["tier_weights"] = dict.fromkeys(selected_tiers, 0.5)
    req = BenchRequest.from_dict(raw)
    trace = WorkloadTrace(
        1,
        TraceMeta(
            "v5",
            sampling={
                "algo_version": 5,
                "request_concurrency": 4,
                "request_timeout_s": 480,
                "min_output_tokens": 2,
                **(
                    {"input_tiers": selected_tiers, "max_baseline_prompt_drops": 4}
                    if selected_tiers
                    else {}
                ),
            },
        ),
        [
            replace(r, max_tokens=8)
            for r in requests()
            if r.input_length_group in (selected_tiers or TIERS)
        ],
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
    assert starts == [
        "baseline",
        "baseline-drift",
        *[f"candidate-{i}" for i in range(6)],
        "scorer",
    ]
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
    expected_count = 8 * len(selected_tiers or TIERS)
    assert all(report.num_prompts == expected_count for report in correctness.values())
    assert len(baseline.result.timings) == expected_count
    assert set(baseline.result.tier_completion) == set(selected_tiers or TIERS)
    for role in starts[:-1]:
        for replay_pass in ("warmup", "rep_1"):
            rows = [
                json.loads(line)
                for line in (
                    layout.sla_bench_dir / role / replay_pass / "requests.jsonl"
                )
                .read_text()
                .splitlines()
            ]
            assert len([r for r in rows if not r.get("_rep_meta")]) == expected_count
    assert entries and all(entry.status == "scored" for entry in entries)
    assert entries[0].score_report["score_breakdown"]["failed_requests"] == 0
    if batched:
        assert all(not t.itl_s for t in baseline.result.timings.values())
        assert all(
            o["token_timing_unavailable_requests"]
            for o in baseline.result.concurrency_observations
        )
    assert entries[0].score_report["score_breakdown"]["failure_penalty"] == 0.1
    assert all(t.completion_tokens == 2 for t in baseline.result.timings.values())


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
