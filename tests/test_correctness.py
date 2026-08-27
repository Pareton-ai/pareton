"""Correctness gate: the shared scorer grades captured output. No Docker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.correctness import (
    BASELINE_INDEX,
    CapturedOutput,
    PendingCorrectness,
    PromptCase,
    capture_outputs,
    distinct_ngram_ratio,
    distinct_word_ratio,
    extract_output_logprobs,
    grade_all,
    grade_candidate,
    post_completion,
    probe_logprob_capability,
    quantile_low,
    resolve_trace_path,
    select_correctness_prompts,
)
from bench.lifecycle import EngineError
from bench.mock_engine import (
    GARBAGE_MARKER,
    GARBAGE_TEXT,
    MockEngine,
    MockEngineConfig,
    build_completion_response,
)
from bench.schemas import CorrectnessConfig, CorrectnessThresholds
from bench.score import PromptTiming
from bench.validate import RequestValidationError, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"


def _cfg(
    *,
    min_mean: float = -4.0,
    min_token: float = -12.0,
    quantile: float = 0.0,
    coverage: float = 0.5,
    num_prompts: int = 2,
    min_distinct: float | None = None,
    min_distinct_ngram: float | None = None,
    max_drop: float | None = None,
) -> CorrectnessConfig:
    """Default quantile is 0, so existing cases still gate on the plain min.

    The PAR-108 bars default to None, matching a campaign whose manifest
    predates them: cases that want them pass them explicitly.
    """
    return CorrectnessConfig(
        num_prompts=num_prompts,
        thresholds=CorrectnessThresholds(
            min_mean_logprob=min_mean,
            min_token_logprob=min_token,
            min_token_quantile=quantile,
            min_coverage_ratio=coverage,
            min_distinct_ratio=min_distinct,
            min_distinct_ngram_ratio=min_distinct_ngram,
            max_mean_logprob_drop=max_drop,
        ),
    )


# A 500-token-style repeat loop: the exploit PAR-108 is about. Nothing here is
# nonsense at the token level, so the scorer rates every token near its own
# greedy path.
LOOP_TEXT = " apple" * 200
# The same attack with a wide vocabulary, to prove the n-gram bar does not
# depend on the loop being short: 150 distinct words repeated 4 times clears
# the word bar comfortably and still reads as a loop.
WIDE_LOOP_TEXT = (" " + " ".join(f"w{i}" for i in range(150))) * 4
PROSE_TEXT = (
    " The function reads the manifest, verifies its hash against the campaign "
    "row, and refuses to continue when the two disagree. That check runs "
    "before any container starts, so a bad pin costs nothing but a log line "
    "and an early exit rather than a wasted round on rented hardware."
)


def _captured(request_id: str, prompt: str, text: str, tokens: int = 2):
    return CapturedOutput(
        request_id=request_id,
        prompt=prompt,
        output_text=text,
        completion_tokens=tokens,
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


def test_probe_logprob_capability_ok():
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as eng:
        resp = probe_logprob_capability(eng.base_url)
        assert "choices" in resp


def test_grade_all_writes_response_shape(tmp_path: Path):
    from bench.correctness import SHAPE_EVIDENCE_FILENAME

    pending = [
        PendingCorrectness(
            candidate_index=0, outputs=[_captured("a", "Hello ", "there now")]
        )
    ]
    evidence = tmp_path / "correctness"
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        grade_all(scorer.base_url, pending, cfg=_cfg(), evidence_dir=evidence)
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


def test_capture_outputs_pairs_prompts_with_what_the_engine_emitted():
    prompts = [PromptCase(id="r1", prompt="Hello "), PromptCase(id="r2", prompt="Hi ")]
    timings = {"r1": PromptTiming(ttft_s=0.1, itl_s=[0.01], completion_tokens=2)}
    captured = capture_outputs(prompts, timings=timings, outputs={"r1": "there now"})
    # r2 was never answered, so there is nothing to grade for it.
    assert [c.request_id for c in captured] == ["r1"]
    assert captured[0].prompt == "Hello "
    assert captured[0].output_text == "there now"
    assert captured[0].completion_tokens == 2


def test_plausible_output_passes(tmp_path: Path):
    outputs = [
        _captured("r1", "Hello world", " OK OK"),
        _captured("r2", "Second prompt", " OK OK"),
    ]
    evidence = tmp_path / "correctness" / "candidate_0.jsonl"
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url, outputs, cfg=_cfg(), evidence_path=evidence
        )
    assert report.verdict == "pass"
    assert report.num_prompts == 2
    assert report.num_positions_scored > 0
    assert report.mean_logprob > -4.0
    assert report.reason is None
    assert evidence.is_file()
    assert not evidence.with_suffix(".jsonl.partial").exists()
    lines = evidence.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_garbage_output_is_disqualified(tmp_path: Path):
    """Correctness is a hard gate, and the scorer judges the text itself."""
    outputs = [_captured("r1", "Hello world", GARBAGE_TEXT, tokens=3)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "fail_correctness"
    assert report.min_logprob <= -30.0
    assert "logprob" in (report.reason or "")


def test_quantile_low_picks_the_kth_lowest():
    """k = ceil(quantile * n), so 0.001 of 4097 positions ignores four."""
    values = [-3.0] * 4096 + [-15.979]
    assert quantile_low(values, 0.001) == -3.0
    assert quantile_low(values, 0.0) == -15.979
    assert quantile_low([-9.0, -1.0, -5.0], 0.0) == -9.0
    # A sample too small for the quantile to reach a second position, and a
    # quantile large enough to reach past the last one, both stay in range.
    assert quantile_low([-9.0, -1.0], 0.001) == -9.0
    assert quantile_low([-9.0, -1.0], 0.99) == -1.0


def test_quantile_low_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="at least one value"):
        quantile_low([], 0.001)


def _long_output(n_tokens: int) -> str:
    return " " + " ".join(f"w{i}" for i in range(n_tokens))


def test_one_outlier_token_no_longer_disqualifies(tmp_path: Path):
    """PAR-94: round 7 disqualified a noop on one token in 4097.

    The scorer rates position 5 at -15.979 and everything else at -3.0, the
    shape of the real evidence. At the default quantile that single position
    is ignored; at quantile 0 it fails, which is what shipped and what round 7
    hit.
    """
    prompt = "Hello world "
    text = _long_output(1400)
    n_positions = len(prompt.split()) + 1400 + 8  # generous upper bound
    logprobs = [-3.0] * n_positions
    logprobs[5] = -15.979
    outputs = [_captured("r1", prompt, text, tokens=1400)]
    cfg = MockEngineConfig(host="127.0.0.1", port=0, logprobs=logprobs)
    with MockEngine(cfg) as scorer:
        passing = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, quantile=0.001),
            evidence_path=tmp_path / "pass" / "candidate_0.jsonl",
        )
        failing = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, quantile=0.0),
            evidence_path=tmp_path / "fail" / "candidate_0.jsonl",
        )
    assert passing.num_positions_scored >= 1001  # or the quantile ignores nothing
    assert passing.verdict == "pass"
    assert passing.min_logprob == -15.979
    assert passing.quantile_logprob == -3.0
    assert failing.verdict == "fail_correctness"
    assert "-15.979" in (failing.reason or "")


def test_garbage_still_fails_at_the_default_quantile(tmp_path: Path):
    """The quantile ignores a few positions, not a wrong answer."""
    outputs = [_captured("r1", "Hello world", GARBAGE_TEXT, tokens=3)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, quantile=0.001),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "fail_correctness"
    assert report.quantile_logprob <= -30.0


def test_a_different_but_plausible_output_still_passes(tmp_path: Path):
    """The gate asks for plausibility, not for a specific answer.

    The scorer's own greedy continuation is " OK"; this candidate emitted
    something else entirely and still passes, because the bar is whether the
    pinned model finds the output plausible, not whether it matches a
    reference run token for token.
    """
    outputs = [_captured("r1", "Hello world", " something else entirely", tokens=3)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "pass"


def test_empty_output_fails_correctness(tmp_path: Path):
    outputs = [_captured("r1", "Hello world", "", tokens=0)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "fail_correctness"
    assert "no output" in (report.reason or "")


def test_no_logprobs_at_all_is_infra_not_disqualification(tmp_path: Path):
    """With nothing scored there is no evidence either way, so the entry is
    requeued rather than disqualified."""

    def nothing(resp, *, original_prompt: str, continuation: str):
        return []

    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        import bench.correctness as correctness

        real = correctness.extract_output_logprobs
        correctness.extract_output_logprobs = nothing
        try:
            report = grade_candidate(
                scorer.base_url,
                [_captured("r1", "Hello world", " OK OK")],
                cfg=_cfg(num_prompts=1),
                evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
            )
        finally:
            correctness.extract_output_logprobs = real
    assert report.verdict == "infra_failed"
    assert report.num_positions_scored == 0


def test_a_candidate_cannot_trade_disqualification_for_a_requeue(tmp_path: Path):
    """Coverage is measured on what the scorer saw, not on what the candidate
    says it emitted, and bad output is judged on the evidence there is.

    An engine reporting a huge token count while emitting nonsense would
    otherwise land under the coverage floor and be requeued instead of
    disqualified.
    """
    outputs = [_captured("r1", "Hello world", GARBAGE_TEXT, tokens=100_000)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "fail_correctness"
    assert report.coverage_ratio == 1.0


def test_one_candidate_the_scorer_chokes_on_does_not_stop_the_batch(tmp_path: Path):
    """The forced text is whatever a candidate engine chose to emit, so one
    entry can fail to grade on input nobody else sent."""
    import bench.correctness as correctness

    real = correctness.extract_output_logprobs

    def fail_on_garbage(resp, *, original_prompt: str, continuation: str):
        if GARBAGE_MARKER in continuation:
            raise EngineError("echo logprobs did not reconstruct the forced sequence")
        return real(resp, original_prompt=original_prompt, continuation=continuation)

    pending = [
        PendingCorrectness(
            candidate_index=0, outputs=[_captured("r1", "Hello world", " OK OK")]
        ),
        PendingCorrectness(
            candidate_index=1,
            outputs=[_captured("r1", "Hello world", GARBAGE_TEXT, tokens=3)],
        ),
        PendingCorrectness(
            candidate_index=2, outputs=[_captured("r1", "Hello world", " OK OK")]
        ),
    ]
    correctness.extract_output_logprobs = fail_on_garbage
    try:
        with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
            reports = grade_all(
                scorer.base_url,
                pending,
                cfg=_cfg(num_prompts=1),
                evidence_dir=tmp_path / "correctness",
            )
    finally:
        correctness.extract_output_logprobs = real
    assert set(reports) == {0, 1, 2}
    assert reports[0].verdict == "pass"
    assert reports[1].verdict == "infra_failed"
    assert "could not grade" in (reports[1].reason or "")
    # The entries either side of the bad one are still graded normally.
    assert reports[2].verdict == "pass"


def test_threshold_is_exclusive(tmp_path: Path):
    """A mean exactly at the bar fails: the check is strictly-below."""
    outputs = [_captured("r1", "Hello world", " OK OK")]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(min_mean=0.0, num_prompts=1),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "fail_correctness"


def test_one_scorer_grades_every_candidate(tmp_path: Path):
    """One scorer start produces a verdict for every candidate in the round."""
    pending = [
        PendingCorrectness(
            candidate_index=0, outputs=[_captured("r1", "Hello world", " OK OK")]
        ),
        PendingCorrectness(
            candidate_index=1,
            outputs=[_captured("r1", "Hello world", GARBAGE_TEXT, tokens=3)],
        ),
    ]
    evidence = tmp_path / "correctness"
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        reports = grade_all(
            scorer.base_url, pending, cfg=_cfg(num_prompts=1), evidence_dir=evidence
        )
    assert set(reports) == {0, 1}
    assert reports[0].verdict == "pass"
    assert reports[1].verdict == "fail_correctness"
    assert (evidence / "candidate_0.jsonl").is_file()
    assert (evidence / "candidate_1.jsonl").is_file()


def test_partial_evidence_left_on_engine_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Mid-run EngineError must not promote .partial to the final evidence name."""

    def boom(resp, *, original_prompt: str, continuation: str):
        raise EngineError("simulated scorer failure")

    monkeypatch.setattr("bench.correctness.extract_output_logprobs", boom)
    evidence_dir = tmp_path / "correctness"
    evidence_dir.mkdir(parents=True)
    prior = evidence_dir / "candidate_0.jsonl"
    prior.write_text('{"prior":true}\n', encoding="utf-8")
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        with pytest.raises(EngineError, match="simulated scorer"):
            grade_candidate(
                scorer.base_url,
                [_captured("r1", "Hello world", " OK OK")],
                cfg=_cfg(num_prompts=1),
                evidence_path=prior,
            )
    assert (evidence_dir / "candidate_0.jsonl.partial").is_file()
    # Failed retry must not erase a prior successful evidence file.
    assert prior.read_text(encoding="utf-8") == '{"prior":true}\n'


# ---------------------------------------------------------------------------
# PAR-108: a degenerate repeat loop outscores a real answer on absolute bars
# ---------------------------------------------------------------------------


def test_degeneracy_metrics_separate_a_loop_from_prose():
    """Both metrics are period-independent, which is the point of the n-gram one."""
    loop = LOOP_TEXT.split()
    wide = WIDE_LOOP_TEXT.split()
    prose = PROSE_TEXT.split()

    assert distinct_word_ratio(loop) < 0.10
    assert distinct_ngram_ratio(loop) < 0.35
    # Widening the loop's vocabulary clears the word bar and not the n-gram one.
    assert distinct_word_ratio(wide) > 0.10
    assert distinct_ngram_ratio(wide) < 0.35
    assert distinct_word_ratio(prose) > 0.10
    assert distinct_ngram_ratio(prose) > 0.35


def test_a_repeat_loop_clears_every_absolute_logprob_bar(tmp_path: Path):
    """The hole PAR-108 closes, kept as a test so it cannot reopen quietly.

    Repeated tokens are the most predictable text there is, so the scorer
    rates them near its own greedy path. Absolute plausibility is not monotone
    in output quality, and no logprob floor alone can catch this.
    """
    outputs = [_captured("r1", "Hello world", LOOP_TEXT, tokens=200)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "pass"
    assert report.mean_logprob > -4.0


def test_a_repeat_loop_is_disqualified_by_the_degeneracy_bars(tmp_path: Path):
    outputs = [_captured("r1", "Hello world", LOOP_TEXT, tokens=200)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, min_distinct=0.10, min_distinct_ngram=0.35),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "fail_correctness"
    assert "degenerate" in (report.reason or "")
    # Still comfortably above the absolute bar: the text bars did the work.
    assert report.mean_logprob > -4.0


def test_a_wide_loop_is_caught_by_the_ngram_bar(tmp_path: Path):
    """Widening the repeated unit evades the word bar, not the n-gram one."""
    outputs = [_captured("r1", "Hello world", WIDE_LOOP_TEXT, tokens=600)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, min_distinct=0.10, min_distinct_ngram=0.35),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "fail_correctness"
    assert "4-gram" in (report.reason or "")


def test_prose_passes_the_degeneracy_bars(tmp_path: Path):
    outputs = [_captured("r1", "Hello world", PROSE_TEXT, tokens=49)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, min_distinct=0.10, min_distinct_ngram=0.35),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "pass"


def test_a_short_output_is_never_read_as_degenerate(tmp_path: Path):
    """Below DEGENERACY_MIN_WORDS there is nothing to repeat, and an honest
    short answer must not be disqualified for having few words."""
    outputs = [_captured("r1", "Hello world", " yes yes yes", tokens=3)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, min_distinct=0.10, min_distinct_ngram=0.35),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "pass"


def test_degeneracy_evidence_records_both_metrics(tmp_path: Path):
    evidence = tmp_path / "correctness" / "candidate_0.jsonl"
    outputs = [_captured("r1", "Hello world", LOOP_TEXT, tokens=200)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, min_distinct=0.10, min_distinct_ngram=0.35),
            evidence_path=evidence,
        )
    line = json.loads(evidence.read_text(encoding="utf-8").strip().splitlines()[0])
    assert line["distinct_word_ratio"] < 0.10
    assert line["distinct_ngram_ratio"] < 0.35
    assert "distinct word ratio" in line["degenerate"]


# ---------------------------------------------------------------------------
# PAR-108: the relative bar, for the opposite failure
# ---------------------------------------------------------------------------


def test_a_candidate_far_below_the_baseline_is_disqualified(tmp_path: Path):
    """A candidate that degrades the model clears the absolute floor and still
    scores well below the baseline on the same prompts."""
    outputs = [_captured("r1", "Hello world", " something else entirely", tokens=3)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, max_drop=0.1),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
            baseline_mean_logprob=-0.2,
        )
    assert report.verdict == "fail_correctness"
    assert "below the baseline" in (report.reason or "")


def test_the_relative_bar_is_skipped_without_a_baseline(tmp_path: Path):
    """Grading the baseline itself, and any campaign that predates the bar,
    both arrive here with no reference and must be graded on the absolute
    bars alone."""
    outputs = [_captured("r1", "Hello world", " something else entirely", tokens=3)]
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        report = grade_candidate(
            scorer.base_url,
            outputs,
            cfg=_cfg(num_prompts=1, max_drop=0.1),
            evidence_path=tmp_path / "correctness" / "candidate_0.jsonl",
        )
    assert report.verdict == "pass"


def test_grade_all_grades_the_baseline_first_and_keys_it_separately(tmp_path: Path):
    """The baseline is queued like any other engine and becomes the reference."""
    pending = [
        PendingCorrectness(
            candidate_index=0, outputs=[_captured("a", "Hello ", PROSE_TEXT)]
        ),
        PendingCorrectness(
            candidate_index=BASELINE_INDEX,
            outputs=[_captured("a", "Hello ", PROSE_TEXT)],
        ),
    ]
    evidence = tmp_path / "correctness"
    with MockEngine(MockEngineConfig(host="127.0.0.1", port=0)) as scorer:
        reports = grade_all(
            scorer.base_url,
            pending,
            cfg=_cfg(num_prompts=1, min_distinct=0.10, min_distinct_ngram=0.35),
            evidence_dir=evidence,
        )
    assert set(reports) == {BASELINE_INDEX, 0}
    assert reports[BASELINE_INDEX].verdict == "pass"
    assert reports[0].verdict == "pass"
    assert (evidence / "baseline.jsonl").is_file()
    assert (evidence / "candidate_0.jsonl").is_file()
