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


@pytest.mark.parametrize("bad_rep", [None, 0, 1, 2])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_only_latency_median_natural_repetition_is_enforced(
    monkeypatch, tmp_path, bad_rep, finish_reason
):
    clean = _clean_long()
    # The first 3000 tokens are clean; the loop begins after the baseline's
    # natural stop. The median-latency row is clean when rep 0 or 2 loops.
    loop = " ".join(clean.split()[:3000]) + " apple" * 2120
    assert degeneracy_reason(clean) is None
    assert degeneracy_reason(loop) is not None
    rows = [
        {
            "request_id": "r1",
            "text": loop if rep == bad_rep else clean,
            "completion_tokens": 5120,
            "e2e_ms": rep + 1,
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
    baseline_text = " ".join(clean.split()[:2500])
    references = build_baseline_degeneracy_references(
        captured,
        {"r1": NaturalStopReference("r1", 2500, "stop", baseline_text)},
        {"r1": (baseline_text,) * 3},
    )
    limits = []

    def score(*args, **kwargs):
        limits.append(kwargs["prefix_token_limit"])
        return [SimpleNamespace(logprob=-0.1)] * 5120, 5120, baseline_text

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    evidence = tmp_path / "candidate.jsonl"
    report = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=1),
        evidence_path=evidence,
        baseline_degeneracy=references,
    )
    assert report.verdict == ("fail_correctness" if bad_rep == 1 else "pass")
    assert limits == [2500]
    row = json.loads(evidence.read_text())
    assert row["degeneracy_scope"] == "full_output"
    assert row["degeneracy_exemptions"] == []
    assert row["output_selection"] == "latency_median"
    assert bool(row["degenerate"]) is (bad_rep == 1)


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
@pytest.mark.parametrize("kind", ["natural-loop", "clean", "forced-loop"])
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
    expected = "fail_correctness" if kind == "natural-loop" else "infra_failed"
    assert report.verdict == expected
    if kind == "natural-loop":
        assert "r1" in report.reason
        path = tmp_path / "candidate_0.jsonl"
        assert report.evidence.endswith(path.name)
        assert not path.with_suffix(".jsonl.partial").exists()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["degenerate"]
        assert rows[0]["output_selection"] == "latency_median"
        assert rows[-1]["request_id"] == failure_request
        assert rows[-1]["scorer_error"] == "simulated scorer failure"
