"""Natural-output repetition enforcement, without GPU or network access."""

import json
import random
import string
from types import SimpleNamespace

import pytest
from test_correctness import SAMPLE_TRACE, _cfg
from test_http import _FakeResp, _sse

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
def test_all_natural_repetitions_are_enforced(
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
    assert report.verdict == ("pass" if bad_rep is None else "fail_correctness")
    assert limits == [2500]
    row = json.loads(evidence.read_text())
    assert row["degeneracy_scope"] == "full_output"
    assert row["degeneracy_exemptions"] == []
    assert [r["rep"] for r in row["repetition_degeneracy"] if r["degenerate"]] == (
        [] if bad_rep is None else [bad_rep + 1]
    )


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
