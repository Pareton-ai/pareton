"""Mock engine: response shape, echo+logprobs quirks, tampered mode."""

from __future__ import annotations

import json
from pathlib import Path

from bench.mock_engine import (
    MockEngine,
    MockEngineConfig,
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


def test_tampered_mode_alters_logprobs():
    clean_cfg = MockEngineConfig(
        tampered=False, logprob_base=-0.5, tamper_logprob_offset=1.25
    )
    dirty_cfg = MockEngineConfig(
        tampered=True, logprob_base=-0.5, tamper_logprob_offset=1.25
    )
    prompt = "Hello world"
    clean = build_completion_response(
        cfg=clean_cfg,
        prompt=prompt,
        max_tokens=0,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    dirty = build_completion_response(
        cfg=dirty_cfg,
        prompt=prompt,
        max_tokens=0,
        echo=True,
        temperature=0.0,
        logprobs_requested=1,
    )
    clean_lps = clean["choices"][0]["logprobs"]["token_logprobs"]
    dirty_lps = dirty["choices"][0]["logprobs"]["token_logprobs"]
    # First stays null in both; at least one later position differs by the offset.
    assert clean_lps[0] is None and dirty_lps[0] is None
    diffs = [
        abs(a - b)
        for a, b in zip(clean_lps[1:], dirty_lps[1:])
        if isinstance(a, float) and isinstance(b, float)
    ]
    assert diffs
    assert all(abs(d - 1.25) < 1e-9 for d in diffs)


def test_configurable_explicit_logprobs_list():
    cfg = MockEngineConfig(logprobs=[-1.0, -2.0, -3.0], tampered=False)
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
