"""Natural-output repetition enforcement, without GPU or network access."""

import json
import random
import string
from types import SimpleNamespace

import pytest
from test_correctness import SAMPLE_TRACE, _cfg
from test_http import _FakeResp, _sse
from test_longform_sampling import rule, sample

from bench.correctness import (
    BaselineDegeneracyReference,
    PromptCase,
    build_baseline_degeneracy_references,
    capture_outputs,
    degeneracy_reason,
    grade_candidate,
    graded_ratios,
    relative_degeneracy_reason,
    select_correctness_prompts,
)
from bench.http import post_completion_stream
from bench.lifecycle import EngineError
from bench.sla_bench import NaturalStopReference, _output_samples, _per_request
from bench.validate import sha256_file


def _clean_long():
    rng = random.Random(42)
    return " ".join(
        "".join(rng.choices(string.ascii_lowercase, k=9)) for _ in range(5120)
    )


def _template_repetition():
    return "\n\n".join(
        f"In district {i}, the sound belongs to {word}. "
        "It is a sound that is always present. "
        "It is a sound that is part of everyday life. "
        "It is a sound that shapes the neighborhood."
        for i, word in enumerate(_clean_long().split()[:10])
    )


@pytest.mark.parametrize(
    "baseline,candidate,fails",
    [
        (0.8953, 0.6808, True),
        (0.9, 0.8, False),
        (0.9, 0.79999, True),
        (0.7, 0.6, False),
        (0.8, 0.9, False),
        (0.14, 0.13, True),  # Existing relative diagnostic below the absolute floor.
    ],
)
def test_relative_distinct_ratio_allows_at_most_ten_percentage_points(
    baseline, candidate, fails
):
    reason = relative_degeneracy_reason(
        _template_repetition(),
        baseline_distinct_ratio=baseline,
        baseline_repeated_span_ratio=0.01,
        distinct_ratio=candidate,
        repeated_span_ratio=0.0153,
    )
    assert (reason is not None) is fails


@pytest.mark.parametrize("scorer_error", [False, True])
def test_sentence_template_repetition_fails_even_with_high_logprobs(
    monkeypatch, tmp_path, scorer_error
):
    text = _template_repetition()
    assert degeneracy_reason(text) is None  # Clears both absolute bars.
    clean = _clean_long()[: len(text)]
    baseline = capture_outputs(
        [PromptCase("r1", "Write")], timings={}, outputs={"r1": clean}
    )
    references = build_baseline_degeneracy_references(
        baseline,
        {"r1": NaturalStopReference("r1", 200, "stop", clean)},
        {"r1": (clean,) * 3},
    )
    captured = capture_outputs(
        [PromptCase("r1", "Write")], timings={}, outputs={"r1": text}
    )

    def score(*a, **kw):
        if scorer_error:
            raise EngineError("scorer unavailable")
        return [SimpleNamespace(logprob=-0.01)] * 200, 200, text

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    path = tmp_path / "candidate.jsonl"
    result = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=1),
        evidence_path=path,
        baseline_degeneracy=references,
    )
    assert result.verdict == "fail_correctness"
    assert "0.10 below baseline" in result.reason
    evidence = json.loads(path.read_text())
    assert evidence["relative_degenerate"]
    assert evidence["distinct_ngram_ratio_drop"] > 0.10
    assert evidence["max_distinct_ngram_ratio_drop"] == 0.10
    assert evidence["output_selection"] == "latency_median"
    assert ("scorer_error" in evidence) is scorer_error
    public = result.to_dict()["prompt_checks"][0]
    assert public["request_id"] == "r1"
    assert public["distinct_ngram_ratio_drop"] == evidence["distinct_ngram_ratio_drop"]
    assert "output_text" not in public
    assert "scorer_error" not in public


def test_relative_guard_uses_least_distinct_valid_baseline_sample(
    monkeypatch, tmp_path
):
    text = _template_repetition()
    clean = _clean_long()[: len(text)]
    baseline = capture_outputs(
        [PromptCase("r1", "Write")], timings={}, outputs={"r1": clean}
    )
    references = build_baseline_degeneracy_references(
        baseline,
        {"r1": NaturalStopReference("r1", 200, "stop", clean)},
        {"r1": (clean, text, clean)},
    )
    assert not references.dropped
    assert references["r1"].full_distinct_ngram_ratio == graded_ratios(text)[0]
    captured = capture_outputs(
        [PromptCase("r1", "Write")], timings={}, outputs={"r1": text}
    )
    monkeypatch.setattr(
        "bench.correctness.score_captured_output",
        lambda *a, **kw: ([SimpleNamespace(logprob=-0.01)], 1, text),
    )
    result = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=1),
        evidence_path=tmp_path / "candidate.jsonl",
        baseline_degeneracy=references,
    )
    assert result.verdict == "pass"


def test_relative_guard_matches_each_of_32_prompts_to_its_own_baseline(
    monkeypatch, tmp_path
):
    repeated = _template_repetition()
    clean = _clean_long()[: len(repeated)]
    prompts = [PromptCase(f"hf-{i:03d}", f"Writing task {i}") for i in range(32)]
    # Identical candidate text should pass the lower reference on even prompts
    # and fail the higher reference on odd prompts. A global baseline statistic
    # would produce the same verdict for every prompt instead.
    samples = {
        prompt.id: (clean, repeated if i % 2 == 0 else clean, clean)
        for i, prompt in enumerate(prompts)
    }
    baseline = capture_outputs(
        prompts, timings={}, outputs={p.id: clean for p in prompts}
    )
    references = build_baseline_degeneracy_references(
        baseline,
        {p.id: NaturalStopReference(p.id, 200, "stop", clean) for p in prompts},
        samples,
    )
    # Reverse the response order to ensure matching is by ID, not list position.
    captured = capture_outputs(
        list(reversed(prompts)),
        timings={},
        outputs={p.id: repeated for p in prompts},
    )
    monkeypatch.setattr(
        "bench.correctness.score_captured_output",
        lambda *a, **kw: ([SimpleNamespace(logprob=-0.01)], 1, repeated),
    )
    path = tmp_path / "candidate.jsonl"
    result = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=32),
        evidence_path=path,
        baseline_degeneracy=references,
    )
    assert result.verdict == "fail_correctness"
    assert result.num_prompts == 32
    evidence = {
        r["request_id"]: r for r in map(json.loads, path.read_text().splitlines())
    }
    assert len(evidence) == 32
    for i, prompt in enumerate(prompts):
        row = evidence[prompt.id]
        assert bool(row["relative_degenerate"]) is (i % 2 == 1)
        assert bool(row["degenerate"]) is (i % 2 == 1)
        assert row["baseline_distinct_ngram_ratio"] == min(
            graded_ratios(text)[0] for text in samples[prompt.id]
        )


@pytest.mark.parametrize("bad_rep", [None, 0, 1, 2])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("loop_slowest", [False, True])
def test_all_natural_repetitions_are_enforced_even_when_runaway_is_slowest(
    monkeypatch, tmp_path, bad_rep, finish_reason, loop_slowest
):
    clean = " ".join(_clean_long().split()[:3000])
    # The first 3000 tokens are clean; the loop begins after the baseline's
    # natural stop. A 5120-token runaway can be the slowest rep, leaving a
    # clean 3000-token sibling as the latency median regardless of rep order.
    loop = clean + " apple" * 2120
    assert degeneracy_reason(clean) is None
    assert degeneracy_reason(loop) is not None
    rows = [
        {
            "request_id": "r1",
            "text": loop if rep == bad_rep else clean,
            "completion_tokens": 5120 if rep == bad_rep else 3000,
            "e2e_ms": 100_000 if loop_slowest and rep == bad_rep else rep + 1,
            "ttft_ms": 1,
            "itl_ms": [],
            "finish_reason": finish_reason,
        }
        for rep in range(3)
    ]
    timings, median = _per_request(rows)
    captured = capture_outputs(
        [PromptCase("r1", "Write a long answer")],
        timings=timings,
        outputs=median,
        output_samples=_output_samples(rows),
    )
    if loop_slowest:
        assert captured[0].output_text == clean
        assert captured[0].completion_tokens == 3000
    baseline_text = " ".join(clean.split()[:2500])
    references = build_baseline_degeneracy_references(
        captured,
        {"r1": NaturalStopReference("r1", 2500, "stop", baseline_text)},
        {"r1": (baseline_text,) * 3},
    )
    limits = []
    relative_texts = []
    from bench.correctness import relative_degeneracy_reason

    def relative_check(text, **kwargs):
        relative_texts.append(text)
        return relative_degeneracy_reason(text, **kwargs)

    monkeypatch.setattr("bench.correctness.relative_degeneracy_reason", relative_check)

    def score(_url, output, **kwargs):
        assert output.output_text == median["r1"]
        limits.append(kwargs["prefix_token_limit"])
        count = output.completion_tokens
        return [SimpleNamespace(logprob=-0.1)] * count, count, baseline_text

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    evidence = tmp_path / "candidate.jsonl"
    report = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=1),
        evidence_path=evidence,
        baseline_degeneracy=references,
    )
    assert report.verdict == ("pass" if bad_rep is None else "fail_correctness")
    assert limits == [2500]
    assert relative_texts == [median["r1"]]
    row = json.loads(evidence.read_text())
    assert row["degeneracy_scope"] == "full_output"
    assert row["degeneracy_exemptions"] == []
    assert row["output_selection"] == "latency_median"
    assert [r["rep"] for r in row["repetition_degeneracy"] if r["degenerate"]] == (
        [] if bad_rep is None else [bad_rep + 1]
    )
    assert bool(row["degenerate"]) is (bad_rep is not None)


def test_normal_eos_cannot_inherit_forced_exemptions(monkeypatch, tmp_path):
    loop = " apple" * 5120
    captured = capture_outputs(
        [PromptCase("r1", "Write")], timings={}, outputs={"r1": loop}
    )
    # Even a mismatched/stale forced reference must not exempt a normal request.
    references = {"r1": BaselineDegeneracyReference(1, 0.0, 1.0, (loop,))}
    monkeypatch.setattr(
        "bench.correctness.score_captured_output",
        lambda *a, **kw: ([SimpleNamespace(logprob=-0.1)], 1, " apple"),
    )
    evidence = tmp_path / "candidate.jsonl"
    report = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=1),
        evidence_path=evidence,
        baseline_degeneracy=references,
    )
    assert report.verdict == "fail_correctness"
    row = json.loads(evidence.read_text())
    assert row["degeneracy_exemptions"] == []
    assert row["degeneracy_scope"] == "full_output"


@pytest.mark.parametrize("chunk_chars", [1, 6, 60])
def test_streamed_speculative_text_cannot_hide_repetition(
    monkeypatch, tmp_path, chunk_chars
):
    text = " apple" * 5120
    chunks = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    # Coalesced speculative output is already rejected for insufficient ITL
    # samples. Extra empty choice chunks can satisfy timing accounting, but
    # must not change the text inspected by correctness.
    chunks.extend([""] * max(0, 5120 - len(chunks)))
    body = _sse(
        *[{"choices": [{"text": chunk, "finish_reason": None}]} for chunk in chunks],
        {"choices": [], "usage": {"completion_tokens": 5120}},
    )
    monkeypatch.setattr("bench.http.urlopen", lambda *a, **kw: _FakeResp(body))
    result = post_completion_stream("http://unused", prompt="Write", max_tokens=5120)
    assert result.text == text
    captured = capture_outputs(
        [PromptCase("r1", "Write")], timings={}, outputs={"r1": result.text}
    )
    monkeypatch.setattr(
        "bench.correctness.score_captured_output",
        lambda *a, **kw: ([SimpleNamespace(logprob=-0.1)], 1, text),
    )
    report = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=1),
        evidence_path=tmp_path / "candidate.jsonl",
    )
    assert report.verdict == "fail_correctness"


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_request_eos_policy_survives_capture(tmp_path, ignore_eos):
    trace = json.loads(SAMPLE_TRACE.read_text())
    trace["requests"][0]["sampling"]["ignore_eos"] = ignore_eos
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    prompts = select_correctness_prompts(
        trace_path=path, expected_sha256=sha256_file(path), num_prompts=1
    )
    request_id = prompts[0].id
    outputs = {request_id: "answer"}
    captured = capture_outputs(
        prompts,
        timings={},
        outputs=outputs,
        output_samples={request_id: ("first", "answer", "third")},
    )
    assert captured[0].ignore_eos is ignore_eos
    assert captured[0].output_samples == ("first", "answer", "third")
    with pytest.raises(EngineError, match="output samples missing"):
        capture_outputs(prompts, timings={}, outputs=outputs, output_samples={})


@pytest.mark.parametrize("loop_tier", [None, "2k", "4k", "8k", "16k"])
def test_tiered_followups_grade_new_outputs_not_history(
    monkeypatch, tmp_path, loop_tier
):
    sampled = sample(rule=rule(max_tokens=5120, min_output_tokens=5000))
    trace_path = tmp_path / "trace.json"
    trace_path.write_bytes(sampled.body)
    prompts = select_correctness_prompts(
        trace_path=trace_path, expected_sha256=sha256_file(trace_path), num_prompts=4
    )
    requests = json.loads(sampled.body)["requests"]
    tiers = {request["id"]: request["input_length_group"] for request in requests}
    clean = _clean_long()
    outputs = {prompt.id: clean for prompt in prompts}
    # These fixture histories repeat heavily in every tier. They must never
    # contaminate either the baseline drop policy or candidate repetition grade.
    assert all(degeneracy_reason(prompt.prompt) for prompt in prompts)
    assert all(not prompt.ignore_eos for prompt in prompts)
    baseline = capture_outputs(prompts, timings={}, outputs=outputs)
    references = build_baseline_degeneracy_references(
        baseline,
        {
            prompt.id: NaturalStopReference(prompt.id, 5120, "length", clean)
            for prompt in prompts
        },
        {prompt.id: (clean,) * 3 for prompt in prompts},
    )
    assert not references.dropped
    samples = {
        prompt.id: (
            clean,
            clean,
            " apple" * 5120 if tiers[prompt.id] == loop_tier else clean,
        )
        for prompt in prompts
    }
    outputs = {
        prompt.id: (" apple" * 5120 if tiers[prompt.id] == loop_tier else clean)
        for prompt in prompts
    }
    captured = capture_outputs(
        prompts, timings={}, outputs=outputs, output_samples=samples
    )
    scored = []

    def score(_url, output, **kwargs):
        scored.append(output.prompt)
        return [SimpleNamespace(logprob=-0.1)] * 5120, 5120, output.output_text

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    evidence = tmp_path / "candidate.jsonl"
    report = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=4),
        evidence_path=evidence,
        baseline_degeneracy=references,
    )
    assert report.verdict == ("pass" if loop_tier is None else "fail_correctness")
    assert scored == [prompt.prompt for prompt in prompts]
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        assert row["degeneracy_scope"] == "full_output"
        assert row["degeneracy_exemptions"] == []
        assert row["output_selection"] == "latency_median"
        assert bool(row["degenerate"]) is (tiers[row["request_id"]] == loop_tier)


@pytest.mark.parametrize("failure_request", ["r1", "r2"])
@pytest.mark.parametrize(
    "kind", ["natural-loop", "sibling-loop", "clean", "forced-loop"]
)
def test_known_repetition_failure_survives_scorer_error(
    monkeypatch, tmp_path, failure_request, kind
):
    from bench.correctness import PendingCorrectness, grade_all

    prompts = [
        PromptCase(rid, "Write", ignore_eos=kind == "forced-loop")
        for rid in ("r1", "r2")
    ]
    captured = capture_outputs(
        prompts,
        timings={},
        outputs={
            rid: (
                " apple" * 200
                if rid == "r1" and kind == "natural-loop"
                else "A clean answer."
            )
            for rid in ("r1", "r2")
        },
        output_samples={
            "r1": (
                "A clean answer.",
                " apple" * 200 if kind != "clean" else "Another answer.",
            ),
            "r2": ("A clean answer.",),
        },
    )
    monkeypatch.setattr(
        "bench.correctness.probe_logprob_capability", lambda *a, **kw: {}
    )

    def score(_url, output, **kwargs):
        if output.request_id == failure_request:
            raise EngineError("simulated scorer failure")
        return [SimpleNamespace(logprob=-0.1)], 1, output.output_text

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    healthy = capture_outputs(
        [PromptCase("r3", "Write")], timings={}, outputs={"r3": "A clean answer."}
    )
    reports = grade_all(
        "unused",
        [PendingCorrectness(0, captured), PendingCorrectness(1, healthy)],
        cfg=_cfg(num_prompts=2),
        evidence_dir=tmp_path,
    )
    assert reports[1].verdict == "pass"
    report = reports[0]
    expected = (
        "fail_correctness"
        if kind in ("natural-loop", "sibling-loop")
        else "infra_failed"
    )
    assert report.verdict == expected
    if kind in ("natural-loop", "sibling-loop"):
        assert "r1" in report.reason
        path = tmp_path / "candidate_0.jsonl"
        assert report.evidence.endswith(path.name)
        assert not path.with_suffix(".jsonl.partial").exists()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["degenerate"]
        assert rows[0]["output_selection"] == "latency_median"
        assert rows[0]["repetition_degeneracy"][1]["degenerate"]
        assert rows[-1]["request_id"] == failure_request
        assert rows[-1]["scorer_error"] == "simulated scorer failure"
