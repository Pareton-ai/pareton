"""Shared baseline-owned correctness and score exclusions, without GPU or DB."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.correctness import PromptCase, baseline_prompt_drops
from bench.lifecycle import EngineError
from bench.main import _build_entries, baseline_drift, run_round
from bench.output import OutputLayout
from bench.schemas import EngineSlaMetrics, EngineSlaResult, LatencyPercentiles
from bench.score import PromptTiming
from bench.sla_bench import EngineReplay
from bench.validate import validate_bench_request_dict

pytestmark = pytest.mark.unit
CLEAN = "A clean answer."
LOOP = " apple" * 200


def trace(count=12):
    return SimpleNamespace(
        requests=[
            SimpleNamespace(id=f"r{i}", sampling=SimpleNamespace(ignore_eos=False))
            for i in range(count)
        ],
        meta=SimpleNamespace(sampling={"algo_version": 4, "min_output_tokens": 3000}),
    )


def replay(role, workload, *, loops=(), short=()):
    ids = [request.id for request in workload.requests]
    timings = {
        rid: PromptTiming(ttft_s=0.1, itl_s=[0.01] * 9, completion_tokens=10)
        for rid in ids
    }
    return EngineReplay(
        result=EngineSlaResult(
            role,
            EngineSlaMetrics(
                LatencyPercentiles(0, 0, 0),
                LatencyPercentiles(0, 0, 0),
                LatencyPercentiles(0, 0, 0),
                0,
                0,
                0,
            ),
            {},
            timings,
            "evidence",
        ),
        outputs={rid: CLEAN for rid in ids},
        output_samples={
            rid: (CLEAN, LOOP if rid in loops else CLEAN, CLEAN) for rid in ids
        },
        completion_token_samples={
            rid: (3000, 2999 if rid in short else 3000, 3000) for rid in ids
        },
    )


def test_union_counts_prompts_once_and_voids_only_above_eight():
    workload = trace()
    first = replay("baseline", workload, loops={"r0", "r1", "r2", "r3"})
    dropped = baseline_prompt_drops(workload, first, dropped={})
    second = replay("baseline-drift", workload, loops={"r3", "r4", "r5", "r6", "r7"})
    union = baseline_prompt_drops(workload, second, dropped=dropped)
    assert len(union) == 8
    assert union["r3"].startswith("baseline:")
    assert union["r7"].startswith("baseline-drift:")
    second.output_samples["r8"] = (CLEAN, CLEAN, LOOP)
    with pytest.raises(EngineError, match="9 prompts.*limit of 8"):
        baseline_prompt_drops(workload, second, dropped=dropped)


def test_short_output_and_repetition_share_the_baseline_exclusion_set():
    workload = trace()
    first = replay("baseline", workload, short={"r0"})
    dropped = baseline_prompt_drops(workload, first, dropped={})
    second = replay("baseline-drift", workload, loops={"r0", "r1"})
    union = baseline_prompt_drops(workload, second, dropped=dropped)
    assert set(union) == {"r0", "r1"}
    assert "2999" in union["r0"]


@pytest.mark.parametrize("bad_count", [8, 9])
def test_both_controls_precede_candidates_and_share_grading_and_score_mask(
    monkeypatch, tmp_path, bad_count
):
    import bench.main as bm

    workload = trace()
    raw = json.loads(Path("fixtures/bench/sample_request.json").read_text())
    raw["correctness"]["thresholds"]["max_mean_logprob_drop"] = 1.5
    raw["scoring_rule"] = {"name": "median_e2e_speedup"}
    request = validate_bench_request_dict(raw)
    first = replay("baseline", workload, short={"r0"})
    second = replay(
        "baseline-drift", workload, loops={f"r{i}" for i in range(1, bad_count)}
    )
    candidate = replay("candidate-0", workload)
    # Garbage in excluded candidate outputs must never reach the scorer.
    candidate.outputs.update({f"r{i}": LOOP for i in range(bad_count)})
    candidate.output_samples.update(
        {f"r{i}": (CLEAN, CLEAN, LOOP) for i in range(bad_count)}
    )
    # These extreme timings must not enter candidate or baseline-comparison scores.
    for rid in [f"r{i}" for i in range(bad_count)]:
        second.result.timings[rid] = PromptTiming(
            ttft_s=100, itl_s=[10] * 9, completion_tokens=10
        )
        candidate.result.timings[rid] = second.result.timings[rid]
    replays = {"baseline": first, "baseline-drift": second, "candidate-0": candidate}
    starts = []

    @contextmanager
    def start(engine, **_):
        starts.append(engine.role)
        yield "unused"

    monkeypatch.setattr(
        "bench.workload_preflight.validate_engine_workload", lambda *a, **k: None
    )
    monkeypatch.setattr(bm, "run_sla_engine", lambda *a, role, **k: replays[role])
    monkeypatch.setattr(
        "bench.correctness.probe_logprob_capability", lambda *a, **k: {}
    )
    scored = []

    def score(_url, captured, **_):
        assert captured.request_id not in {f"r{i}" for i in range(bad_count)}
        scored.append(captured.request_id)
        return [SimpleNamespace(logprob=-0.1)] * 10, 10, captured.output_text

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    layout = OutputLayout(tmp_path)
    layout.prepare()
    prompts = [PromptCase(r.id, "Write") for r in workload.requests]
    args = {
        "req": request,
        "provider": SimpleNamespace(start=start),
        "prompts": prompts,
        "trace": workload,
        "layout": layout,
    }
    if bad_count == 9:
        with pytest.raises(EngineError, match="9 prompts.*limit of 8") as caught:
            run_round(**args)
        assert caught.value.error_role == "baseline"
        assert starts == ["baseline", "baseline-drift"]
        assert not scored
        return
    baseline, drift, runs, correctness = run_round(**args)
    assert starts == ["baseline", "baseline-drift", "candidate-0", "scorer"]
    assert correctness[0].verdict == "pass"
    assert correctness[0].num_prompts == 4
    assert set(scored) == {"r8", "r9", "r10", "r11"}
    evidence = json.loads(
        (layout.correctness_dir / "baseline_exclusions.json").read_text()
    )
    assert evidence == baseline.excluded_prompts
    entries = _build_entries(
        req=request,
        baseline=baseline,
        runs=runs,
        correctness=correctness,
        digests=["sha256:" + "b" * 64],
    )
    assert entries[0].score == 0
    assert len(entries[0].score_report["prompts"]) == 4
    assert entries[0].score_report["excluded_prompts"] == evidence
    assert baseline_drift(request, baseline, drift) == 0


@pytest.mark.parametrize("failing_role", ["baseline", "baseline-drift"])
@pytest.mark.parametrize("relative_bar", [None, 1.5])
def test_empty_correctness_subset_aborts_before_candidates(
    monkeypatch, tmp_path, failing_role, relative_bar
):
    workload = trace()
    raw = json.loads(Path("fixtures/bench/sample_request.json").read_text())
    raw["correctness"]["thresholds"]["max_mean_logprob_drop"] = relative_bar
    request = validate_bench_request_dict(raw)
    starts = []

    @contextmanager
    def start(engine, **_):
        starts.append(engine.role)
        assert engine.role in ("baseline", "baseline-drift")
        yield "unused"

    monkeypatch.setattr(
        "bench.workload_preflight.validate_engine_workload", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "bench.main.run_sla_engine",
        lambda *a, role, **k: replay(
            role, workload, loops={"r0", "r1"} if role == failing_role else ()
        ),
    )
    layout = OutputLayout(tmp_path)
    layout.prepare()
    # Ten timing prompts remain stable, but none of the correctness sample do.
    with pytest.raises(EngineError, match="no stable correctness prompts") as caught:
        run_round(
            req=request,
            provider=SimpleNamespace(start=start),
            prompts=[PromptCase(rid, "Write") for rid in ("r0", "r1")],
            trace=workload,
            layout=layout,
        )
    assert caught.value.error_role == "baseline"
    assert starts == (
        ["baseline"] if failing_role == "baseline" else ["baseline", "baseline-drift"]
    )


def test_round_plan_marker_survives_teardown_without_stale_engine_fields(monkeypatch):
    from worker.round_job import _round_phase_writer

    writes = []
    monkeypatch.setattr(
        "worker.round_job.set_round_phase", lambda **kw: writes.append(kw) or True
    )
    write = _round_phase_writer("round")
    write(
        job_id=1,
        attempt=1,
        phase="sla_bench",
        progress={"plan_version": 2, "role": "baseline-drift", "step": 2},
    )
    write(job_id=1, attempt=1, phase="teardown")
    assert writes[-1]["progress"] == {"plan_version": 2}
