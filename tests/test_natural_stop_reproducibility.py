import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_bench_cli import SAMPLE_REQUEST
from test_concurrency import requests, tier_stats, timings

from bench import main as bm
from bench.concurrency import TIERS, WEIGHTED_RULE, tier_completion_metrics
from bench.lifecycle import EngineError
from bench.output import OutputLayout
from bench.schemas import BenchRequest, TraceMeta, WorkloadTrace
from bench.sla_bench import EngineReplay, REPRO_BAR_MAX_REL_RANGE

pytestmark = pytest.mark.unit


def workload(ignore_eos=False):
    raw = json.loads(SAMPLE_REQUEST.read_text())
    raw["mode"] = "sla_bench"
    raw["engines"]["candidates"] = [raw["engines"]["candidates"][0]]
    raw["leader_candidate_index"] = 0
    raw["sla_bench"]["repetitions"] = 3
    raw["scoring_rule"] = {"name": WEIGHTED_RULE, "failure_penalty": 0.1}
    trace = WorkloadTrace(
        1,
        TraceMeta(
            "v5",
            sampling={
                "algo_version": 5,
                "request_concurrency": 4,
                "request_timeout_s": 600,
                "min_output_tokens": 2,
            },
        ),
        [
            replace(r, sampling=replace(r.sampling, ignore_eos=ignore_eos))
            for r in requests()
        ],
    )
    return BenchRequest.from_dict(raw), trace


@pytest.mark.parametrize("ignore_eos", [False, True])
@pytest.mark.parametrize("unstable_role", ["baseline", "baseline-drift"])
def test_baseline_tier_variance_depends_on_output_contract(
    monkeypatch, tmp_path, caplog, ignore_eos, unstable_role
):
    req, trace = workload(ignore_eos)
    layout = OutputLayout(tmp_path)
    layout.prepare()
    calls = []

    class Provider:
        @contextmanager
        def start(self, start, phase):
            yield start.role

    def replay(url, *, requests, **kwargs):
        calls.append(url)
        rows = []
        for rep in range(1, 4):
            duration = [84.501, 37.021, 64.739][rep - 1] if url == unstable_role else 80
            rep_rows = [
                {
                    "request_id": r.id,
                    "rep": rep,
                    "input_length_group": r.input_length_group,
                    "group_start_offset_ms": 0,
                    "admission_offset_ms": 0,
                    "protocol_completion_offset_ms": 1000
                    * (duration if r.input_length_group == "2k" else 80),
                    "completion_tokens": 2,
                    "finish_reason": "length" if ignore_eos else "stop",
                }
                for r in requests
            ]
            path = layout.sla_bench_dir / url / f"rep_{rep}" / "requests.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(row) for row in rep_rows))
            rows.extend(rep_rows)
        tiers = tier_completion_metrics(rows, 3)
        result = SimpleNamespace(
            role=url,
            timings=timings(),
            tier_completion=tiers,
            cross_rep_variance={
                "tier_completion_max_rel_range": max(
                    t["relative_range"] for t in tiers.values()
                )
            },
        )
        return EngineReplay(result, {}, {})

    monkeypatch.setattr(bm, "run_sla_engine", replay)
    monkeypatch.setattr(
        bm, "baseline_prompt_drops", lambda trace, replay, *, dropped: dropped
    )
    monkeypatch.setattr(
        "bench.workload_preflight.validate_engine_workload", lambda *a, **kw: None
    )
    if ignore_eos:
        with pytest.raises(
            EngineError, match="baseline tier completion repetitions are unstable"
        ):
            bm.run_round(
                req=req, provider=Provider(), prompts=[], trace=trace, layout=layout
            )
        assert not any(role.startswith("candidate") for role in calls)
    else:
        baseline, drift, runs, _ = bm.run_round(
            req=req, provider=Provider(), prompts=[], trace=trace, layout=layout
        )
        reference = baseline if unstable_role == "baseline" else drift
        assert reference.result.cross_rep_variance[
            "tier_completion_max_rel_range"
        ] == pytest.approx(0.7334, abs=0.0001)
        assert reference.result.tier_completion["2k"]["repetitions_s"] == pytest.approx(
            [84.501, 37.021, 64.739]
        )
        assert len(runs) == 1
        assert "diagnostic only" in caplog.text


def candidate_replays():
    baseline = EngineReplay(
        SimpleNamespace(timings=timings(), tier_completion=tier_stats()), {}, {}
    )
    result = SimpleNamespace(
        role="candidate-0",
        timings=timings(),
        tier_completion=tier_stats(8),
        cross_rep_variance={"tier_completion_max_rel_range": 0.7334},
    )
    return baseline, [
        bm._CandidateRun(index=0, status="scored", replay=EngineReplay(result, {}, {}))
    ]


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_candidate_and_incumbent_keep_raw_variance_without_changing_score(
    caplog, ignore_eos
):
    req, trace = workload(ignore_eos)
    baseline, runs = candidate_replays()
    kwargs = dict(
        req=req,
        baseline=baseline,
        runs=runs,
        correctness={},
        digests=["sha256:" + "b" * 64],
        trace=trace,
    )
    entry = bm._build_entries(**kwargs)[0]
    assert entry.sla.cross_rep_variance["tier_completion_max_rel_range"] == 0.7334
    if ignore_eos:
        assert entry.status == "infra_failed"
        assert entry.score is None
    else:
        assert entry.status == "scored"
        assert entry.score == pytest.approx(0.2)
        assert "diagnostic only" in caplog.text
        before = entry.score_report
        runs[0].replay.result.cross_rep_variance["tier_completion_max_rel_range"] = 0.1
        assert bm._build_entries(**kwargs)[0].score_report == before


def test_missing_trace_does_not_disable_candidate_gate():
    req, _ = workload()
    baseline, runs = candidate_replays()
    entry = bm._build_entries(
        req=req, baseline=baseline, runs=runs, correctness={}, digests=[]
    )[0]
    assert entry.status == "infra_failed"
    assert "tier_completion_max_rel_range" in entry.reason


@pytest.mark.parametrize(
    "verdict,status",
    [("fail_correctness", "disqualified"), ("infra_failed", "infra_failed")],
)
def test_natural_stop_variance_does_not_bypass_correctness(verdict, status):
    req, trace = workload()
    baseline, runs = candidate_replays()
    entry = bm._build_entries(
        req=req,
        baseline=baseline,
        runs=runs,
        correctness={0: SimpleNamespace(verdict=verdict, reason="correctness failure")},
        digests=[],
        trace=trace,
    )[0]
    assert entry.status == status
    assert entry.score is None
    assert entry.reason == "correctness failure"


def test_natural_stop_variance_does_not_bypass_engine_failure():
    req, trace = workload()
    baseline, _ = candidate_replays()
    entry = bm._build_entries(
        req=req,
        baseline=baseline,
        runs=[
            bm._CandidateRun(index=0, status="infra_failed", reason="request timeout")
        ],
        correctness={},
        digests=[],
        trace=trace,
    )[0]
    assert entry.status == "infra_failed"
    assert entry.reason == "request timeout"


@pytest.mark.parametrize(
    "algo_version,rule", [(4, WEIGHTED_RULE), (5, "median_e2e_speedup")]
)
def test_diagnostic_scope_requires_v5_and_weighted_rule(algo_version, rule):
    req, trace = workload()
    req.scoring_rule = {"name": rule}
    trace.meta.sampling["algo_version"] = algo_version
    assert not bm._natural_stop_tier_workload(req, trace)


def test_mixed_output_contract_remains_variable_length():
    req, trace = workload(True)
    trace.requests[0].sampling.ignore_eos = False
    assert bm._natural_stop_tier_workload(req, trace)
    assert REPRO_BAR_MAX_REL_RANGE == 0.335
    assert set(tier_stats()) == set(TIERS)
