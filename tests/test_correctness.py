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
    resolve_trace_path,
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


def test_resolve_trace_path_prefers_request_dir(tmp_path: Path, monkeypatch):
    """Relative path: request directory wins over a same-named CWD file."""
    req_dir = tmp_path / "req"
    cwd = tmp_path / "cwd"
    req_dir.mkdir()
    cwd.mkdir()
    (req_dir / "trace.json").write_text('{"from":"request"}', encoding="utf-8")
    (cwd / "trace.json").write_text('{"from":"cwd"}', encoding="utf-8")
    monkeypatch.chdir(cwd)
    resolved = resolve_trace_path(
        "trace.json", request_path=req_dir / "bench_request.json"
    )
    assert resolved == (req_dir / "trace.json").resolve()
    assert resolved.read_text(encoding="utf-8") == '{"from":"request"}'


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
    scores = extract_output_logprobs(resp, original_prompt=prompt, continuation=cont)
    assert scores, "expected at least one output position"
    assert all(s.text_offset >= len(prompt) for s in scores)
    assert all(s.text_offset < len(full) for s in scores)
    # First prompt token null must not appear
    assert all(s.position > 0 for s in scores)


def test_extract_sglang_completion_only_offsets():
    """SGLang echo logprobs offset into choices[0].text, not the prompt."""
    prompt = "Hello world" * 20
    resp = {
        "choices": [
            {
                "logprobs": {
                    "tokens": [" OK"],
                    "token_logprobs": [-0.2],
                    "top_logprobs": [{" OK": -0.2}],
                    "text_offset": [0],
                }
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 1},
    }
    scores = extract_output_logprobs(resp, original_prompt=prompt, continuation=" OK")
    assert len(scores) == 1
    assert scores[0].token == " OK"


def test_extract_offset_cut_drops_token_past_continuation():
    """vLLM-style offsets: keep C, drop the clamp extra after P+C."""
    prompt = "Hello"
    cont = " world"
    extra = " !"
    tokens = ["Hello", " world", extra]
    resp = {
        "choices": [
            {
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": [None, -0.2, -0.9],
                    "top_logprobs": [None, {" world": -0.2}, {extra: -0.9}],
                    "text_offset": [0, 5, 11],
                }
            }
        ]
    }
    scores = extract_output_logprobs(resp, original_prompt=prompt, continuation=cont)
    assert [s.token for s in scores] == [" world"]


def test_extract_sglang_full_echo_drops_clamp_extra():
    """Live SGLang shape: echo of P+C plus 1 clamp token, offsets all -1."""
    prompt = "Hello world"
    cont = " OK then"
    extra = " !"
    tokens = ["Hello", " world", " OK", " then", extra]
    n = len(tokens)
    resp = {
        "choices": [
            {
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": [-0.1] * n,
                    "top_logprobs": [{t: -0.1} for t in tokens],
                    "text_offset": [-1] * n,
                }
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
    }
    scores = extract_output_logprobs(resp, original_prompt=prompt, continuation=cont)
    assert [s.token for s in scores] == [" OK", " then"]


def test_extract_non_numeric_logprobs_is_engine_error():
    resp = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["a", "b"],
                    "token_logprobs": [None, "not-a-float"],
                    "top_logprobs": [None, {"b": -0.1}],
                    "text_offset": [0, 1],
                }
            }
        ]
    }
    with pytest.raises(EngineError, match="malformed logprobs.token_logprobs"):
        extract_output_logprobs(resp, original_prompt="", continuation="ab")


def test_empty_baseline_continuation_is_engine_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A prompt with empty greedy output must not be silently skipped."""
    from bench.correctness import post_completion

    real_post = post_completion

    def empty_generate(url, *, prompt, max_tokens=16, echo=False, **kwargs):
        if not echo and max_tokens > 0:
            return {"choices": [{"text": ""}]}
        return real_post(url, prompt=prompt, max_tokens=max_tokens, echo=echo, **kwargs)

    monkeypatch.setattr("bench.correctness.post_completion", empty_generate)
    prompts = [PromptCase(id="p1", prompt="Hello world")]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as base:
        with pytest.raises(EngineError, match="empty continuation"):
            run_correctness(
                base.base_url,
                base.base_url,
                prompts=prompts,
                cfg=_cfg(num_prompts=1),
                task_id="550e8400-e29b-41d4-a716-446655440000",
                evidence_dir=tmp_path / "correctness",
            )


def test_probe_logprob_capability_ok():
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as eng:
        resp = probe_logprob_capability(eng.base_url)
        assert "choices" in resp


def test_run_correctness_writes_response_shape(tmp_path: Path):
    from bench.correctness import SHAPE_EVIDENCE_FILENAME

    prompts = [PromptCase(id="a", prompt="Hello "), PromptCase(id="b", prompt="Hi ")]
    evidence = tmp_path / "correctness"
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as base:
        with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as cand:
            run_correctness(
                base.base_url,
                cand.base_url,
                prompts=prompts,
                cfg=_cfg(),
                task_id="t",
                evidence_dir=evidence,
            )
    shape_path = evidence / SHAPE_EVIDENCE_FILENAME
    assert shape_path.is_file()
    payload = json.loads(shape_path.read_text(encoding="utf-8"))
    assert "fingerprint" in payload
    assert "choices" in payload["fingerprint"]


def test_post_completion_invalid_json_is_engine_error(monkeypatch: pytest.MonkeyPatch):
    """HTTP 200 with non-JSON body must become EngineError (CLI exit 3), not crash."""
    from io import BytesIO
    from urllib.response import addinfourl

    class _FakeHeaders(dict):
        def get_content_charset(self, failobj=None):  # noqa: ANN001
            return "utf-8"

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        fp = BytesIO(b"not-json{{{")
        return addinfourl(fp, _FakeHeaders(), req.full_url, code=200)

    monkeypatch.setattr("bench.http.urlopen", fake_urlopen)
    with pytest.raises(EngineError, match="invalid JSON"):
        post_completion("http://127.0.0.1:9", prompt="hi", max_tokens=1)


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
    assert report.argmax_mismatches == 0
    evidence = tmp_path / "correctness" / "logprob_diffs.jsonl"
    assert evidence.is_file()
    assert not (tmp_path / "correctness" / "logprob_diffs.jsonl.partial").exists()
    lines = evidence.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == report.num_positions_compared


def test_threshold_equality_is_fail(tmp_path: Path):
    """Thresholds are exclusive: equality fails."""
    prompts = [PromptCase(id="p1", prompt="Hello world")]
    with (
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as base,
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as cand,
    ):
        report = run_correctness(
            base.base_url,
            cand.base_url,
            prompts=prompts,
            cfg=_cfg(mean=0.0, max_d=0.0, argmax=0.0, num_prompts=1),
            task_id="550e8400-e29b-41d4-a716-446655440000",
            evidence_dir=tmp_path / "correctness",
        )
    assert report.mean_abs_logprob_diff == 0.0
    assert report.verdict == "fail_correctness"


def test_min_positions_compared_raises_engine_error(tmp_path: Path):
    prompts = [PromptCase(id="p1", prompt="Hello world")]
    cfg = CorrectnessConfig(
        num_prompts=1,
        max_new_tokens=2,
        thresholds=CorrectnessThresholds(
            mean_abs_logprob_diff=1.0,
            max_abs_logprob_diff=1.0,
            argmax_mismatch_rate=1.0,
        ),
        min_positions_compared=10_000,
    )
    with (
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as base,
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as cand,
    ):
        with pytest.raises(EngineError, match="min_positions_compared"):
            run_correctness(
                base.base_url,
                cand.base_url,
                prompts=prompts,
                cfg=cfg,
                task_id="550e8400-e29b-41d4-a716-446655440000",
                evidence_dir=tmp_path / "correctness",
            )


def test_partial_evidence_left_on_engine_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Mid-run EngineError must not promote .partial to the final evidence name."""
    from bench.correctness import extract_output_logprobs

    real_extract = extract_output_logprobs
    calls = {"n": 0}

    def fail_on_candidate(resp, *, original_prompt: str, continuation: str):
        scores = real_extract(
            resp, original_prompt=original_prompt, continuation=continuation
        )
        calls["n"] += 1
        if calls["n"] > 1:
            raise EngineError("simulated candidate score failure")
        return scores

    monkeypatch.setattr("bench.correctness.extract_output_logprobs", fail_on_candidate)
    prompts = [PromptCase(id="p1", prompt="Hello world")]
    evidence_dir = tmp_path / "correctness"
    evidence_dir.mkdir(parents=True)
    prior = evidence_dir / "logprob_diffs.jsonl"
    prior.write_text('{"prior":true}\n', encoding="utf-8")
    with (
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as base,
        MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as cand,
    ):
        with pytest.raises(EngineError, match="simulated candidate"):
            run_correctness(
                base.base_url,
                cand.base_url,
                prompts=prompts,
                cfg=_cfg(num_prompts=1),
                task_id="550e8400-e29b-41d4-a716-446655440000",
                evidence_dir=evidence_dir,
            )
    assert (evidence_dir / "logprob_diffs.jsonl.partial").is_file()
    # Failed retry must not erase a prior successful evidence file.
    assert prior.read_text(encoding="utf-8") == '{"prior":true}\n'


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

    def mutate_candidate_tokens(resp, *, original_prompt: str, continuation: str):
        scores = real_extract(
            resp, original_prompt=original_prompt, continuation=continuation
        )
        calls["n"] += 1
        # Baseline phase extracts first; candidate phase extracts afterward.
        if calls["n"] > 1 and scores:
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
