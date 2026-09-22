"""Repeated-span allowance over distinct prompts, without GPU or network."""

import json
import random
import string
from types import SimpleNamespace

import pytest
from test_correctness import _cfg

from bench.correctness import (
    BaselineDegeneracyReferences,
    PendingCorrectness,
    PromptCase,
    build_baseline_degeneracy_references,
    capture_outputs,
    degeneracy_reason,
    grade_all,
    grade_candidate,
)
from bench.lifecycle import EngineError
from bench.sla_bench import NaturalStopReference

CLEAN = "".join(random.Random(42).choices(string.ascii_lowercase, k=1024))
SPAN = CLEAN[:300] * 2 + " tail"
FLOOR = " apple" * 200


def captured_cases(count, mode="median"):
    prompts = [
        PromptCase(f"r{i:02d}", "Write", ignore_eos=mode.startswith("forced"))
        for i in range(32)
    ]
    baseline = capture_outputs(
        prompts, timings={}, outputs={p.id: CLEAN for p in prompts}
    )
    refs = build_baseline_degeneracy_references(
        baseline,
        {
            p.id: NaturalStopReference(
                p.id, 200, "stop", CLEAN, probed=mode.startswith("forced")
            )
            for p in prompts
        },
        {p.id: ((SPAN if mode == "forced-match" else CLEAN),) * 3 for p in prompts},
    )
    flagged = {p.id for p in prompts[:count]}
    text = SPAN + "</think>" + CLEAN if mode == "split" else SPAN
    outputs = {
        p.id: text
        if p.id in flagged and mode not in ("sibling", "prefix", "forced-prefix")
        else CLEAN
        for p in prompts
    }
    samples = {
        p.id: (
            (CLEAN, CLEAN, SPAN)
            if mode == "sibling" and p.id in flagged
            else (outputs[p.id],) * 3
        )
        for p in prompts
    }
    return (
        capture_outputs(prompts, timings={}, outputs=outputs, output_samples=samples),
        refs,
        flagged,
    )


@pytest.mark.parametrize("count", [0, 1, 4, 5])
@pytest.mark.parametrize("mode", ["median", "sibling", "prefix", "split"])
def test_four_prompts_pass_fifth_fails(monkeypatch, tmp_path, count, mode):
    assert "longest repeated span" in degeneracy_reason(SPAN)
    assert degeneracy_reason(SPAN, check_repeated_span=False) is None
    captured, refs, flagged = captured_cases(count, mode)

    def score(_url, output, **kwargs):
        prefix = (
            SPAN
            if mode == "prefix" and output.request_id in flagged
            else output.output_text
        )
        return [SimpleNamespace(logprob=-0.1)], 1, prefix

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    report = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=32),
        baseline_degeneracy=refs,
        evidence_path=tmp_path / "checks.jsonl",
    )
    assert report.verdict == ("pass" if count <= 4 else "fail_correctness")
    assert report.repeated_span_degeneracy == {
        "max_failed_prompts": 4,
        "failed_prompts": count,
        "failed_request_ids": sorted(flagged),
    }
    assert report.num_prompts == 32
    assert report.num_positions_scored == 32
    assert sum(bool(r.get("degenerate")) for r in report.prompt_checks) == count
    if count > 4:
        assert "repeated-span failures exceed allowance of 4 prompts" in report.reason


@pytest.mark.parametrize("count", [1, 4, 5])
@pytest.mark.parametrize("error_at", ["boundary", "later"])
def test_scorer_error_requires_proven_over_budget_failure(
    monkeypatch, tmp_path, count, error_at
):
    captured, refs, _ = captured_cases(count, "sibling")
    error_id = f"r{count - 1:02d}" if error_at == "boundary" else "r31"

    def score(_url, output, **kwargs):
        if output.request_id == error_id:
            raise EngineError("scorer unavailable")
        return [SimpleNamespace(logprob=-0.1)], 1, output.output_text

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    monkeypatch.setattr(
        "bench.correctness.probe_logprob_capability", lambda *a, **kw: {}
    )
    # A second candidate has its own allowance and continues after a scorer error.
    clean, _, _ = captured_cases(0)
    clean = [o for o in clean if o.request_id != error_id]
    reports = grade_all(
        "unused",
        [PendingCorrectness(0, captured), PendingCorrectness(1, clean)],
        cfg=_cfg(num_prompts=32),
        baseline_degeneracy=refs,
        evidence_dir=tmp_path,
    )
    assert reports[0].verdict == ("infra_failed" if count <= 4 else "fail_correctness")
    assert reports[1].verdict == "pass"
    assert reports[1].repeated_span_degeneracy["failed_prompts"] == 0
    if count > 4:
        assert reports[0].repeated_span_degeneracy["failed_prompts"] == 5
        rows = [
            json.loads(line)
            for line in (tmp_path / "candidate_0.jsonl").read_text().splitlines()
        ]
        assert rows[-1]["scorer_error"] == "scorer unavailable"


@pytest.mark.parametrize(
    "mode", ["later-rep", "answer-floor", "full-floor", "empty-rep"]
)
def test_tolerated_span_cannot_mask_immediate_failure(monkeypatch, tmp_path, mode):
    text = (
        SPAN + "</think>" + FLOOR
        if mode == "answer-floor"
        else FLOOR
        if mode == "full-floor"
        else CLEAN
    )
    samples = (
        (SPAN, FLOOR if mode == "later-rep" else "")
        if mode in ("later-rep", "empty-rep")
        else (text,)
    )
    captured = capture_outputs(
        [PromptCase("r1", "Write")],
        timings={},
        outputs={"r1": text},
        output_samples={"r1": samples},
    )
    monkeypatch.setattr(
        "bench.correctness.score_captured_output",
        lambda *a, **kw: ([SimpleNamespace(logprob=-0.1)], 1, SPAN),
    )
    report = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=1),
        evidence_path=tmp_path / "checks.jsonl",
    )
    assert report.verdict == "fail_correctness"
    assert (
        "empty output" if mode == "empty-rep" else "below harness floor"
    ) in report.reason


@pytest.mark.parametrize(
    "mode,count,expected",
    [
        ("forced-tail", 5, 0),
        ("forced-match", 5, 0),
        ("forced-prefix", 4, 4),
        ("forced-prefix", 5, 5),
        ("excluded", 5, 4),
    ],
)
def test_exclusions_and_forced_tail_scope(monkeypatch, tmp_path, mode, count, expected):
    captured, refs, flagged = captured_cases(count, mode)
    if mode == "excluded":
        refs = BaselineDegeneracyReferences(
            {k: v for k, v in refs.items() if k != "r00"},
            dropped={"r00": "baseline loop"},
        )

    def score(_url, output, **kwargs):
        prefix = (
            SPAN
            if mode in ("forced-prefix", "forced-match")
            and output.request_id in flagged
            else CLEAN
        )
        return [SimpleNamespace(logprob=-0.1)], 1, prefix

    monkeypatch.setattr("bench.correctness.score_captured_output", score)
    report = grade_candidate(
        "unused",
        captured,
        cfg=_cfg(num_prompts=32),
        baseline_degeneracy=refs,
        evidence_path=tmp_path / "checks.jsonl",
    )
    assert report.verdict == ("pass" if expected <= 4 else "fail_correctness")
    assert report.repeated_span_degeneracy["failed_prompts"] == expected


def test_baseline_span_still_excludes_prompt():
    captured = capture_outputs(
        [PromptCase("r1", "Write")], timings={}, outputs={"r1": SPAN}
    )
    refs = build_baseline_degeneracy_references(
        captured, {"r1": NaturalStopReference("r1", 200, "stop", SPAN)}
    )
    assert not refs
    assert "longest repeated span" in refs.dropped["r1"]
