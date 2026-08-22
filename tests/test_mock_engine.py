"""Mock engine: response shape, echo+logprobs quirks, round scripting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.mock_engine import (
    GARBAGE_TEXT,
    MockEngine,
    MockEngineConfig,
    _latency_at,
    build_completion_response,
    mock_tokenize,
    post_completion,
    response_shape_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
SHAPE_FIXTURE = ROOT / "fixtures" / "bench" / "vllm_completion_response_shape.json"


def test_tokenize_roundtrip_basic():
    text = "Hello world"
    tokens = mock_tokenize(text)
    assert "".join(tokens) == text
    assert len(tokens) >= 2


def test_echo_first_prompt_logprob_is_null():
    cfg = MockEngineConfig(model="mock-model")
    resp = build_completion_response(
        cfg=cfg,
        prompt="Hello world",
        max_tokens=2,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    lp = resp["choices"][0]["logprobs"]
    assert lp is not None
    assert lp["token_logprobs"][0] is None
    assert lp["top_logprobs"][0] is None
    # Later tokens have numeric logprobs
    assert any(isinstance(x, float) for x in lp["token_logprobs"][1:])
    assert resp["object"] == "text_completion"
    assert "usage" in resp
    assert resp["usage"]["prompt_tokens"] > 0
    assert resp["usage"]["completion_tokens"] == 2


def test_response_shape_matches_checked_in_fixture():
    fixture = json.loads(SHAPE_FIXTURE.read_text(encoding="utf-8"))
    cfg = MockEngineConfig(model="mock-model")
    resp = build_completion_response(
        cfg=cfg,
        prompt="Hello world",
        max_tokens=2,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    assert response_shape_fingerprint(resp) == fixture["fingerprint"]
    # Example in fixture has the required top-level keys
    for key in ("id", "object", "created", "model", "choices", "usage"):
        assert key in fixture["example"]
    choice = fixture["example"]["choices"][0]
    assert "logprobs" in choice
    assert choice["logprobs"]["token_logprobs"][0] is None


def test_http_server_completions():
    with MockEngine(MockEngineConfig(model="mock-http", token_latency_s=0.0)) as eng:
        resp = post_completion(
            eng.completions_url,
            prompt="Hi there",
            max_tokens=1,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
    assert resp["model"] == "mock-http"
    assert resp["choices"][0]["logprobs"]["token_logprobs"][0] is None


def test_stream_chunks_match_token_count_and_done():
    from bench.http import post_completion_stream

    with MockEngine(MockEngineConfig(model="s", token_latency_s=0.0)) as eng:
        res = post_completion_stream(
            eng.base_url, prompt="hi", max_tokens=5, temperature=0.0
        )
    assert res.completion_tokens == 5
    assert res.finish_reason == "length"
    # 5 chunks -> 4 inter-token gaps.
    assert len(res.itl_s) == 4
    assert res.ttft_s >= 0
    assert res.e2e_s >= res.ttft_s
    assert res.text != ""


def test_stream_per_token_delay_observable():
    from bench.http import post_completion_stream

    with MockEngine(MockEngineConfig(model="s2", token_latency_s=0.02)) as eng:
        res = post_completion_stream(
            eng.base_url, prompt="hi", max_tokens=3, temperature=0.0
        )
    # First-chunk delay ~ token_latency; ITL gaps ~ token_latency.
    assert res.ttft_s >= 0.015
    assert all(g >= 0.015 for g in res.itl_s)


def test_repeat_to_max_tokens():
    cfg = MockEngineConfig(model="r")
    resp = build_completion_response(
        cfg=cfg,
        prompt="hi",
        max_tokens=6,
        echo=False,
        temperature=0.0,
        logprobs_requested=None,
    )
    # greedy " OK" is 2 tokens; cycled to fill 6.
    assert resp["usage"]["completion_tokens"] == 6
    assert resp["choices"][0]["finish_reason"] == "length"


def test_garbage_output_scores_far_below_a_clean_one():
    """A scorer judges text, not who produced it, which is what lets one
    shared scorer grade output captured from any number of candidates."""
    scorer = MockEngineConfig(logprob_base=-0.5)
    clean = build_completion_response(
        cfg=scorer,
        prompt="Hello world" + " OK OK",
        max_tokens=0,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    dirty = build_completion_response(
        cfg=scorer,
        prompt="Hello world" + GARBAGE_TEXT,
        max_tokens=0,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    clean_lps = [x for x in clean["choices"][0]["logprobs"]["token_logprobs"] if x]
    dirty_lps = [x for x in dirty["choices"][0]["logprobs"]["token_logprobs"] if x]
    assert min(clean_lps) > -5.0
    assert min(dirty_lps) <= -30.0


def test_speed_factor_divides_per_token_latency():
    base = MockEngineConfig(token_latency_s=0.02)
    fast = MockEngineConfig(token_latency_s=0.02, speed_factor=2.0)
    assert _latency_at(base, 0) == pytest.approx(0.02)
    assert _latency_at(fast, 0) == pytest.approx(0.01)
    # A zero or negative factor is ignored rather than dividing by zero.
    assert _latency_at(MockEngineConfig(token_latency_s=0.02, speed_factor=0.0), 0) == (
        pytest.approx(0.02)
    )


def test_configurable_explicit_logprobs_list():
    cfg = MockEngineConfig(logprobs=[-1.0, -2.0, -3.0])
    resp = build_completion_response(
        cfg=cfg,
        prompt="A B C",
        max_tokens=0,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    lps = resp["choices"][0]["logprobs"]["token_logprobs"]
    # index 0 null; index 1 uses logprobs[1], etc.
    assert lps[0] is None
    assert lps[1] == -2.0


def test_top_logprobs_respects_requested_k():
    cfg = MockEngineConfig()
    one = build_completion_response(
        cfg=cfg,
        prompt="Hello world",
        max_tokens=0,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    three = build_completion_response(
        cfg=cfg,
        prompt="Hello world",
        max_tokens=0,
        echo=True,
        temperature=0.0,
        logprobs_requested=3,
    )
    top1 = one["choices"][0]["logprobs"]["top_logprobs"][1]
    top3 = three["choices"][0]["logprobs"]["top_logprobs"][1]
    assert top1 is not None and len(top1) == 1
    assert top3 is not None and len(top3) == 3


def test_run_stub_does_not_accumulate_root_handlers(tmp_path: Path):
    import logging

    from bench.main import main

    root = logging.getLogger()
    before = len(root.handlers)
    sample = ROOT / "fixtures" / "bench" / "sample_request.json"
    argv_a = [
        "--request",
        str(sample),
        "--output-dir",
        str(tmp_path / "a"),
        "--mock-engine",
    ]
    argv_b = [
        "--request",
        str(sample),
        "--output-dir",
        str(tmp_path / "b"),
        "--mock-engine",
    ]
    assert main(argv_a) == 0
    assert main(argv_b) == 0
    assert len(root.handlers) == before
