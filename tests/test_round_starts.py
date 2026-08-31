"""Round start count and cache isolation. No GPU, no Docker daemon.

A round of one leader plus five challengers costs exactly 9 engine starts:
the baseline, six candidates, one scorer, and the closing drift baseline. The
count is the same whether the campaign pins vLLM or SGLang, because only the
scorer is started with correctness-specific serve args.

These tests also hold the cache rule: the compile cache is mounted for the
baseline runs and never for a candidate, so no candidate inherits a warm
cache the previous one left behind.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from bench.main import _EngineProvider, plan_round_starts, run_round
from bench.output import OutputLayout
from bench.phases import BenchPhase
from bench.validate import (
    load_workload_trace,
    validate_bench_request_dict,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "fixtures" / "bench" / "sample_request.json"
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"

# One leader plus five challengers, the cohort size a round is built around.
ROUND_CANDIDATES = 6
EXPECTED_STARTS = 9

VLLM_SERVE_ARGS = ["--model", "/model", "--enable-prefix-caching"]
SGLANG_SERVE_ARGS = ["--model-path", "/model", "--tp-size", "8"]
VLLM_CACHE_DIR = "/root/.cache/vllm"
SGLANG_CACHE_DIR = "/root/.cache/sglang"


def _request(
    serve_args: list[str],
    *,
    candidates: int = ROUND_CANDIDATES,
    cache_dir: str = VLLM_CACHE_DIR,
) -> dict:
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    raw["workload_trace"]["path"] = str(SAMPLE_TRACE)
    raw["engines"]["baseline"] = {
        "image": "ghcr.io/example/engine@sha256:" + ("a" * 64),
        "serve_args": list(serve_args),
        "env": {},
        "cache_dir": cache_dir,
    }
    raw["engines"]["candidates"] = [
        {
            "image": f"ghcr.io/example/engine@sha256:{'0123456789abcdef'[i] * 64}",
            "serve_args": list(serve_args),
            "env": {},
            "cache_dir": cache_dir,
        }
        for i in range(candidates)
    ]
    return raw


@pytest.mark.parametrize(
    "engine,serve_args",
    [("vllm", VLLM_SERVE_ARGS), ("sglang", SGLANG_SERVE_ARGS)],
)
def test_five_challenger_round_is_nine_starts_on_either_engine(
    engine: str, serve_args: list[str]
):
    req = validate_bench_request_dict(_request(serve_args))
    plan = plan_round_starts(req.engines)
    assert len(plan) == EXPECTED_STARTS, engine
    assert [s.kind for s in plan] == (
        ["baseline"] + ["candidate"] * ROUND_CANDIDATES + ["scorer", "drift"]
    )


def test_the_two_engines_produce_the_identical_plan_shape():
    """A round costs the same number of starts whichever engine is pinned."""
    vllm = plan_round_starts(
        validate_bench_request_dict(_request(VLLM_SERVE_ARGS)).engines
    )
    sglang = plan_round_starts(
        validate_bench_request_dict(_request(SGLANG_SERVE_ARGS)).engines
    )
    assert [s.role for s in vllm] == [s.role for s in sglang]
    assert len(vllm) == len(sglang) == EXPECTED_STARTS


def test_only_a_baseline_start_mounts_the_engine_cache():
    """Each candidate begins in the same cache state as every other one."""
    plan = plan_round_starts(
        validate_bench_request_dict(_request(VLLM_SERVE_ARGS)).engines
    )
    mounted = {s.kind for s in plan if s.mount_engine_cache}
    assert mounted == {"baseline", "drift"}
    assert not any(s.mount_engine_cache for s in plan if s.kind == "candidate")


def test_no_candidate_runs_with_correctness_serve_args():
    """Exactly one container per round carries the scorer flags."""
    req = validate_bench_request_dict(_request(VLLM_SERVE_ARGS))
    plan = plan_round_starts(req.engines)
    scorers = [s for s in plan if s.kind == "scorer"]
    assert len(scorers) == 1
    assert "--no-enable-prefix-caching" in scorers[0].spec.serve_args
    for start in plan:
        if start.kind == "candidate":
            assert start.spec.serve_args == VLLM_SERVE_ARGS


def test_sglang_scorer_skips_the_vllm_only_flags_and_still_costs_one_start():
    req = validate_bench_request_dict(_request(SGLANG_SERVE_ARGS))
    plan = plan_round_starts(req.engines)
    scorers = [s for s in plan if s.kind == "scorer"]
    assert len(scorers) == 1
    assert scorers[0].spec.serve_args == SGLANG_SERVE_ARGS


def test_every_start_carries_its_position_in_the_plan():
    """PAR-98: nine starts share two phase names, so position is the only signal."""
    plan = plan_round_starts(
        validate_bench_request_dict(_request(VLLM_SERVE_ARGS)).engines
    )
    assert [s.step for s in plan] == list(range(1, EXPECTED_STARTS + 1))
    assert {s.steps for s in plan} == {EXPECTED_STARTS}


def test_position_counts_the_plan_actually_returned():
    """sla_bench mode drops the scorer, so `steps` must drop with it."""
    plan = plan_round_starts(
        validate_bench_request_dict(_request(VLLM_SERVE_ARGS)).engines,
        mode="sla_bench",
    )
    assert {s.steps for s in plan} == {len(plan)} == {EXPECTED_STARTS - 1}


def test_sla_bench_mode_skips_the_scorer():
    req = validate_bench_request_dict(_request(VLLM_SERVE_ARGS))
    plan = plan_round_starts(req.engines, mode="sla_bench")
    assert len(plan) == EXPECTED_STARTS - 1
    assert not any(s.kind == "scorer" for s in plan)


def test_the_runner_performs_exactly_the_planned_starts(tmp_path: Path):
    """The runner starts what the plan lists, and nothing else."""
    req = validate_bench_request_dict(_request(VLLM_SERVE_ARGS, candidates=2))
    trace = load_workload_trace(SAMPLE_TRACE, expected_sha256=req.workload_trace.sha256)
    layout = OutputLayout(tmp_path / "out")
    layout.prepare()
    provider = _EngineProvider(req=req, mock=True, logs_dir=tmp_path / "logs")
    from bench.correctness import select_correctness_prompts
    from bench.validate import sha256_file

    prompts = select_correctness_prompts(
        trace_path=SAMPLE_TRACE,
        expected_sha256=sha256_file(SAMPLE_TRACE),
        num_prompts=req.correctness.num_prompts,
    )
    run_round(req=req, provider=provider, prompts=prompts, trace=trace, layout=layout)
    assert provider.starts == [
        "baseline",
        "candidate-0",
        "candidate-1",
        "scorer",
        "baseline-drift",
    ]
    # The sample request predates PAR-108's relative bar. It must retain the
    # original candidate-only correctness path rather than grading a baseline.
    assert not (layout.correctness_dir / "baseline.jsonl").exists()


def test_empty_baseline_output_reaches_candidate_parity_grading(
    tmp_path: Path, monkeypatch
):
    raw = _request(VLLM_SERVE_ARGS, candidates=1)
    raw["correctness"]["thresholds"]["max_mean_logprob_drop"] = 1.5
    req = validate_bench_request_dict(raw)
    trace = load_workload_trace(SAMPLE_TRACE, expected_sha256=req.workload_trace.sha256)
    layout = OutputLayout(tmp_path / "out")
    layout.prepare()
    provider = _EngineProvider(req=req, mock=True, logs_dir=tmp_path / "logs")
    from bench.correctness import select_correctness_prompts
    from bench.sla_bench import EngineReplay
    from bench.validate import sha256_file

    prompts = select_correctness_prompts(
        trace_path=SAMPLE_TRACE,
        expected_sha256=sha256_file(SAMPLE_TRACE),
        num_prompts=req.correctness.num_prompts,
    )
    outputs = {request.id: "" for request in trace.requests}
    replay = EngineReplay(
        result=SimpleNamespace(
            timings={
                request.id: SimpleNamespace(completion_tokens=1)
                for request in trace.requests
            }
        ),
        outputs=outputs,
        output_samples={request.id: ("",) for request in trace.requests},
    )

    monkeypatch.setattr("bench.main.run_sla_engine", lambda *_args, **_kwargs: replay)

    _, _, _, correctness = run_round(
        req=req,
        provider=provider,
        prompts=prompts,
        trace=trace,
        layout=layout,
    )

    assert correctness[0].verdict == "pass"
    assert correctness[0].coverage_ratio == 1.0
    rows = [
        json.loads(line)
        for line in (layout.correctness_dir / "candidate_0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["baseline_empty_match"] for row in rows] == [True, True]


@pytest.mark.parametrize(
    "serve_args,cache_dir",
    [(VLLM_SERVE_ARGS, VLLM_CACHE_DIR), (SGLANG_SERVE_ARGS, SGLANG_CACHE_DIR)],
)
def test_a_candidate_container_is_never_launched_with_a_cache_mount(
    tmp_path: Path, monkeypatch, serve_args: list[str], cache_dir: str
):
    """End to end through the real docker argv builder, against a fake daemon."""
    from tests.test_lifecycle import FakeDocker

    cache = tmp_path / "engine-cache"
    monkeypatch.setenv("PARETON_BENCH_ENGINE_CACHE_DIR", str(cache))
    req = validate_bench_request_dict(
        _request(serve_args, candidates=2, cache_dir=cache_dir)
    )
    fake = FakeDocker()
    for spec in [req.engines.baseline, *req.engines.candidates]:
        fake.image_digests[spec.image] = [f"{spec.image}"]
    provider = _EngineProvider(
        req=req, mock=False, logs_dir=tmp_path / "logs", docker_runner=fake
    )

    import bench.lifecycle as life

    original_wait = life.wait_until_healthy
    life.wait_until_healthy = lambda *_a, **_k: None  # type: ignore[assignment]
    try:
        for start in plan_round_starts(req.engines):
            with provider.start(start, phase=BenchPhase.SLA_BENCH):
                pass
    finally:
        life.wait_until_healthy = original_wait  # type: ignore[assignment]

    mount = f"{cache.resolve()}:{cache_dir}"
    runs = [c for c, _ in fake.calls if c[:2] == ["docker", "run"]]
    assert len(runs) == 5
    by_name = {c[c.index("--name") + 1]: c for c in runs}
    for name, argv in by_name.items():
        if "candidate" in name or "scorer" in name:
            assert mount not in argv, name
        else:
            assert mount in argv, name
