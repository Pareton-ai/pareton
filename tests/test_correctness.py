"""Module A correctness gate — in-process mock engines, no Docker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.correctness import (
    PromptCase,
    extract_output_logprobs,
    post_completion,
    probe_logprob_capability,
    run_correctness,
    scoring_order,
    select_correctness_prompts,
)
from bench.lifecycle import EngineError
from bench.mock_engine import MockEngine, MockEngineConfig, build_completion_response
from bench.schemas import CorrectnessConfig, CorrectnessThresholds
from bench.validate import RequestValidationError, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"


def _cfg(
    *,
    mean: float = 0.005,
    max_d: float = 0.05,
    argmax: float = 0.001,
    num_prompts: int = 2,
    max_new: int = 2,
) -> CorrectnessConfig:
    return CorrectnessConfig(
        num_prompts=num_prompts,
        max_new_tokens=max_new,
        thresholds=CorrectnessThresholds(
            mean_abs_logprob_diff=mean,
            max_abs_logprob_diff=max_d,
            argmax_mismatch_rate=argmax,
        ),
    )


def test_select_correctness_prompts_from_trace():
    prompts = select_correctness_prompts(
        trace_path=SAMPLE_TRACE,
        expected_sha256=sha256_file(SAMPLE_TRACE),
        num_prompts=2,
    )
    assert len(prompts) == 2
    assert prompts[0].id == "r-000001"
    assert prompts[0].prompt == "Hello world"


def test_select_rejects_token_ids_only(tmp_path: Path):
    trace = {
        "schema_version": 1,
        "meta": {"name": "t"},
        "requests": [
            {
                "id": "r1",
                "arrival_offset_ms": 0,
                "prompt_token_ids": [1, 2, 3],
                "max_tokens": 4,
                "sampling": {"temperature": 0.0, "top_p": 1.0},
            }
        ],
    }
    path = tmp_path / "t.json"
    raw = json.dumps(trace).encode("utf-8")
    path.write_bytes(raw)
    from bench.validate import sha256_bytes

    with pytest.raises(RequestValidationError, match="text prompt"):
        select_correctness_prompts(
            trace_path=path,
            expected_sha256=sha256_bytes(raw),
            num_prompts=1,
        )


def test_scoring_order_deterministic_by_task_id():
    ids = ["a", "b", "c", "d", "e"]
    t1 = "550e8400-e29b-41d4-a716-446655440000"
    t2 = "550e8400-e29b-41d4-a716-446655440001"
    assert scoring_order(ids, t1) == scoring_order(ids, t1)
    assert scoring_order(ids, t1) != scoring_order(ids, t2)


def test_extract_skips_null_and_prompt_portion():
    cfg = MockEngineConfig()
    prompt = "Hello world"
    cont = " OK"
    full = prompt + cont
    resp = build_completion_response(
        cfg=cfg,
        prompt=full,
        max_tokens=0,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    scores = extract_output_logprobs(resp, original_prompt=prompt)
    assert scores, "expected at least one output position"
    assert all(s.text_offset >= len(prompt) for s in scores)
    # First prompt token null must not appear
    assert all(s.position > 0 for s in scores)


def test_probe_logprob_capability_ok():
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as eng:
        probe_logprob_capability(eng.base_url)


def test_probe_raises_when_logprobs_missing(monkeypatch):
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as eng:
        original = post_completion

        def no_lp(*args, **kwargs):
            resp = original(*args, **kwargs)
            resp["choices"][0]["logprobs"] = None
            return resp

        monkeypatch.setattr("bench.correctness.post_completion", no_lp)
        with pytest.raises(EngineError, match="missing logprob capability"):
            probe_logprob_capability(eng.base_url)


def test_baseline_vs_baseline_passes(tmp_path: Path):
    prompts = select_correctness_prompts(
        trace_path=SAMPLE_TRACE,
        expected_sha256=sha256_file(SAMPLE_TRACE),
        num_prompts=2,
    )
    with (
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as base,
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as cand,
    ):
        report = run_correctness(
            base.base_url,
            cand.base_url,
            prompts=prompts,
            cfg=_cfg(),
            task_id="550e8400-e29b-41d4-a716-446655440000",
            evidence_dir=tmp_path / "correctness",
        )
    assert report.verdict == "pass"
    assert report.num_prompts == 2
    assert report.num_positions_compared > 0
    assert report.mean_abs_logprob_diff == 0.0
    assert report.max_abs_logprob_diff == 0.0
    assert report.argmax_mismatch_rate == 0.0
    evidence = tmp_path / "correctness" / "logprob_diffs.jsonl"
    assert evidence.is_file()
    lines = evidence.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == report.num_positions_compared


def test_tampered_candidate_fails(tmp_path: Path):
    prompts = [
        PromptCase(id="p1", prompt="Hello world"),
        PromptCase(id="p2", prompt="Second prompt"),
    ]
    with (
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as base,
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0, tampered=True)) as cand,
    ):
        report = run_correctness(
            base.base_url,
            cand.base_url,
            prompts=prompts,
            cfg=_cfg(),
            task_id="550e8400-e29b-41d4-a716-446655440000",
            evidence_dir=tmp_path / "correctness",
        )
    assert report.verdict == "fail_correctness"
    # Tamper offset is +1.0 — well above thresholds.
    assert report.mean_abs_logprob_diff >= 0.9
    assert report.max_abs_logprob_diff >= 0.9


def test_forced_token_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Logprob diffs are undefined when engines report different forced tokens."""
    from bench.correctness import _PositionScore, extract_output_logprobs

    real_extract = extract_output_logprobs
    calls = {"n": 0}

    def mutate_candidate_tokens(resp, *, original_prompt: str):
        scores = real_extract(resp, original_prompt=original_prompt)
        calls["n"] += 1
        # Even calls are candidate scores in run_correctness (base then cand).
        if calls["n"] % 2 == 0 and scores:
            return [
                _PositionScore(
                    position=s.position,
                    token=s.token + "_X",
                    text_offset=s.text_offset,
                    logprob=s.logprob,
                    top1=s.top1,
                )
                for s in scores
            ]
        return scores

    monkeypatch.setattr(
        "bench.correctness.extract_output_logprobs", mutate_candidate_tokens
    )
    prompts = [PromptCase(id="p1", prompt="Hello world")]
    with (
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as base,
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as cand,
    ):
        with pytest.raises(EngineError, match="forced token mismatch"):
            run_correctness(
                base.base_url,
                cand.base_url,
                prompts=prompts,
                cfg=_cfg(num_prompts=1),
                task_id="550e8400-e29b-41d4-a716-446655440000",
                evidence_dir=tmp_path / "correctness",
            )
