"""SGLang native scoring, including the byte-split Unicode seen on the H200."""

import json
from copy import deepcopy

import pytest

from bench import correctness
from bench.correctness import (
    BASELINE_INDEX,
    BaselineDegeneracyReference,
    CapturedOutput,
    PendingCorrectness,
    grade_all,
    grade_candidate,
    score_captured_output,
)
from bench.lifecycle import EngineError
from bench.schemas import CorrectnessConfig, CorrectnessThresholds

pytestmark = pytest.mark.unit

# IDs and individual decodes from the pinned Qwen tokenizer; synthetic logprobs.
PROMPT_IDS = [12675, 10838, 247, 226, 198]
OUTPUT_IDS = [3793, 25677, 244, 13]
TOKEN_TEXTS = ["Hi", " �", "�", "�", "\n", "OK", " �", "�", "."]
CAPTURED = CapturedOutput("unicode", "Hi 🙄\n", "OK 😖.", 4)
CFG = CorrectnessConfig(1, CorrectnessThresholds(-4.0, -12.0, 0.001, 0.5))


@pytest.fixture
def native(monkeypatch):
    state = {
        "tokens": [list(PROMPT_IDS), PROMPT_IDS + OUTPUT_IDS],
        "rows": [
            [None if i == 0 else -0.1, token_id, text]
            for i, (token_id, text) in enumerate(
                zip(PROMPT_IDS + OUTPUT_IDS, TOKEN_TEXTS)
            )
        ],
        "decoded": "OK 😖.",
        "context_prefix": "",
        "calls": [],
    }

    def post(_url, path, body, **_kw):
        state["calls"].append((path, body))
        if path == "/tokenize":
            return {"tokens": deepcopy(state["tokens"])}
        if path == "/detokenize":
            assert body["skip_special_tokens"] is False
            assert body["tokens"][:2] == [PROMPT_IDS, PROMPT_IDS + OUTPUT_IDS]
            context = state["context_prefix"] + CAPTURED.prompt
            prefix = body["tokens"][2][len(PROMPT_IDS) :]
            if prefix == OUTPUT_IDS:
                prefix_text = state["decoded"]
            else:
                prefix_text = {
                    tuple(OUTPUT_IDS[:2]): "OK �",
                    tuple(OUTPUT_IDS[:3]): "OK 😖",
                }[tuple(prefix)]
            return {
                "text": [context, context + state["decoded"], context + prefix_text]
            }
        assert path == "/generate"
        assert body["input_ids"] == PROMPT_IDS + OUTPUT_IDS
        assert body["logprob_start_len"] == 0
        return {
            "meta_info": {
                "input_token_logprobs": deepcopy(state["rows"]),
                "output_token_logprobs": [[-999.0, 99, "extra"]],
            }
        }

    monkeypatch.setattr(correctness, "post_json", post)
    return state


def test_native_score_handles_unicode_and_excludes_generated_token(native):
    assert "".join(TOKEN_TEXTS) != CAPTURED.prompt + CAPTURED.output_text
    scores, span, prefix = score_captured_output(
        "http://scorer", CAPTURED, engine_name="sglang"
    )
    assert span == 4
    assert [p.token_id for p in scores] == OUTPUT_IDS
    assert [p.logprob for p in scores] == [-0.1] * 4
    assert prefix == CAPTURED.output_text
    assert native["calls"][0][1]["prompt"] == [
        CAPTURED.prompt,
        CAPTURED.prompt + CAPTURED.output_text,
    ]


@pytest.mark.parametrize(
    "problem", ["wrong_id", "missing_row", "wrong_boundary", "wrong_text"]
)
def test_native_score_rejects_misaligned_sequence(native, problem):
    if problem == "wrong_id":
        native["rows"][0][1] = 999
    elif problem == "missing_row":
        native["rows"].pop()
    elif problem == "wrong_boundary":
        native["tokens"][1][0] = 999
    else:
        native["decoded"] = "different output"
    with pytest.raises(EngineError):
        score_captured_output("http://scorer", CAPTURED, engine_name="sglang")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "-0.1"])
def test_native_score_rejects_invalid_logprob(native, value):
    native["rows"][-1][0] = value
    with pytest.raises(EngineError, match="invalid input logprob"):
        score_captured_output("http://scorer", CAPTURED, engine_name="sglang")


def test_native_score_keeps_unscored_positions_in_coverage(native):
    native["rows"][-1][0] = None
    scores, span, _ = score_captured_output(
        "http://scorer", CAPTURED, engine_name="sglang"
    )
    assert len(scores) == 3
    assert span == 4


def test_native_decode_keeps_model_added_prefix_out_of_continuation(native):
    native["context_prefix"] = "<bos>"
    scores, span, prefix = score_captured_output(
        "http://scorer", CAPTURED, engine_name="sglang"
    )
    assert len(scores) == span == 4
    assert prefix == CAPTURED.output_text


@pytest.mark.parametrize("stop_tokens,prefix_chars", [(2, 3), (3, 4), (4, 5)])
def test_native_repetition_check_uses_decoded_token_prefix(
    native, tmp_path, stop_tokens, prefix_chars
):
    evidence = tmp_path / "candidate.jsonl"
    report = grade_candidate(
        "http://scorer",
        [CAPTURED],
        cfg=CFG,
        evidence_path=evidence,
        engine_name="sglang",
        baseline_degeneracy={
            "unicode": BaselineDegeneracyReference(stop_tokens, 1.0, 0.0)
        },
    )
    assert report.verdict == "pass"
    assert json.loads(evidence.read_text())["prefix_chars"] == prefix_chars


def test_grade_all_uses_native_scorer_for_baseline_and_candidate(
    native, monkeypatch, tmp_path
):
    monkeypatch.setattr(correctness, "probe_logprob_capability", lambda *a, **kw: {})
    reports = grade_all(
        "http://scorer",
        [
            PendingCorrectness(0, [CAPTURED]),
            PendingCorrectness(BASELINE_INDEX, [CAPTURED]),
        ],
        cfg=CFG,
        evidence_dir=tmp_path,
        engine_name="sglang",
    )
    assert reports[BASELINE_INDEX].verdict == reports[0].verdict == "pass"
    assert sum(path == "/generate" for path, _ in native["calls"]) == 2
