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
from bench.main import scorer_engine_spec
from bench.schemas import CorrectnessConfig, CorrectnessThresholds, EngineSpec

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
        "continuation_ids": list(OUTPUT_IDS),
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
            if isinstance(body["prompt"], str):
                assert body["add_special_tokens"] is False
                return {"tokens": deepcopy(state["continuation_ids"])}
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


@pytest.mark.parametrize("problem", ["wrong_id", "missing_row", "wrong_text"])
@pytest.mark.parametrize("merged_boundary", [False, True])
def test_native_score_rejects_misaligned_sequence(native, problem, merged_boundary):
    if merged_boundary:
        native["tokens"][1][0] = 999
    if problem == "wrong_id":
        native["rows"][0][1] = 999
    elif problem == "missing_row":
        native["rows"].pop()
    else:
        native["decoded"] = "different output"
    with pytest.raises(EngineError):
        score_captured_output("http://scorer", CAPTURED, engine_name="sglang")


@pytest.mark.parametrize("ids", [None, [], [True]])
def test_native_score_rejects_invalid_continuation_ids_after_boundary_merge(
    native, ids
):
    native["tokens"][1][0] = 999
    native["continuation_ids"] = ids
    with pytest.raises(EngineError, match="invalid continuation IDs"):
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


@pytest.mark.parametrize(
    "leading_newlines,output_token,merged_token",
    [("\n", 198, 1358), ("\n\n", 271, 987)],
)
def test_native_score_preserves_prompt_when_newlines_merge(
    monkeypatch, tmp_path, leading_newlines, output_token, merged_token
):
    # Recorded from the Qwen tokenizer shared by the pinned BF16/FP8 models:
    # SHA256 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3.
    # Encoding the concatenated text merges trailing prompt newlines with the
    # first output token. Preserve the original prompt even with a BOS prefix.
    prompt_ids = [12675, 10838, 247, 226, 271]
    output_ids = [output_token, 9419, 13]
    merged_ids = prompt_ids[:-1] + [merged_token, 9419, 13]
    captured = CapturedOutput("newlines", "Hi 🙄\n\n", leading_newlines + "Hello.", 3)
    calls = []

    def post(_url, path, body, **_kw):
        calls.append((path, body))
        if path == "/tokenize":
            if isinstance(body["prompt"], list):
                assert body["prompt"] == [
                    captured.prompt,
                    captured.prompt + captured.output_text,
                ]
                return {"tokens": [prompt_ids, merged_ids]}
            assert body["prompt"] == captured.output_text
            assert body["add_special_tokens"] is False
            return {"tokens": output_ids}
        if path == "/detokenize":
            assert body["skip_special_tokens"] is False
            assert body["tokens"] == [
                prompt_ids,
                prompt_ids + output_ids,
                prompt_ids + output_ids[:1],
            ]
            prompt = "<bos>" + captured.prompt
            return {
                "text": [
                    prompt,
                    prompt + captured.output_text,
                    prompt + leading_newlines,
                ]
            }
        assert path == "/generate"
        assert body["input_ids"] == prompt_ids + output_ids
        return {
            "meta_info": {
                "input_token_logprobs": [
                    [None if i == 0 else -0.1, token_id, "token"]
                    for i, token_id in enumerate(prompt_ids + output_ids)
                ],
                "output_token_logprobs": [[-999.0, 99, "clamp"]],
            }
        }

    monkeypatch.setattr(correctness, "post_json", post)
    monkeypatch.setattr(correctness, "probe_logprob_capability", lambda *a, **kw: {})
    cfg = CorrectnessConfig(
        1, CorrectnessThresholds(-4.0, -12.0, 0.001, 0.5, max_mean_logprob_drop=1.5)
    )
    reports = grade_all(
        "http://scorer",
        [
            PendingCorrectness(0, [captured]),
            PendingCorrectness(BASELINE_INDEX, [captured]),
        ],
        cfg=cfg,
        evidence_dir=tmp_path,
        engine_name="sglang",
        baseline_degeneracy={"newlines": BaselineDegeneracyReference(1, 1.0, 0.0)},
    )
    for report in reports.values():
        assert report.verdict == "pass"
        assert report.num_positions_scored == 3
        assert report.coverage_ratio == 1.0
        assert report.mean_logprob == pytest.approx(-0.1)
        evidence = tmp_path / report.evidence.rsplit("/", 1)[-1]
        assert json.loads(evidence.read_text())["prefix_chars"] == len(leading_newlines)
    assert sum(path == "/generate" for path, _ in calls) == 2


def test_full_replay_context_scores_every_token_with_relative_bar(
    monkeypatch, tmp_path
):
    context_len = 8192
    prompt_ids = [1] * 3072
    output_ids = list(range(100, 5220))
    full_ids = prompt_ids + output_ids
    token_text = {1: "p", **{i: chr(0x4E00 + i) for i in output_ids}}
    captured = CapturedOutput(
        "full-context",
        "p" * len(prompt_ids),
        "".join(token_text[i] for i in output_ids),
        len(output_ids),
    )
    assert len(full_ids) == 8192
    rows = [
        [None if i == 0 else -0.1, token_id, token_text[token_id]]
        for i, token_id in enumerate(full_ids)
    ]
    rows[-1][0] = -0.7  # The final captured token must contribute to the mean.

    def post(_url, path, body, **_kw):
        if path == "/tokenize":
            return {"tokens": [prompt_ids, full_ids]}
        if path == "/detokenize":
            return {
                "text": ["".join(token_text[i] for i in ids) for ids in body["tokens"]]
            }
        assert path == "/generate"
        assert body["input_ids"] == full_ids  # No truncation to make scoring fit.
        # Pinned SGLang tokenizer + scheduler guards: the worker reserves one
        # slot, then five input slots, and validate_input_length uses >=.
        if (
            len(body["input_ids"]) >= context_len - 6
            or len(body["input_ids"]) + body["sampling_params"]["max_new_tokens"]
            > context_len
        ):
            raise EngineError("SGLang input exceeds context limit")
        return {
            "meta_info": {
                "input_token_logprobs": rows,
                "output_token_logprobs": [[-999.0, 99999, "clamp"]],
            }
        }

    monkeypatch.setattr(correctness, "post_json", post)
    monkeypatch.setattr(correctness, "probe_logprob_capability", lambda *a, **kw: {})
    with pytest.raises(EngineError, match="context limit"):
        score_captured_output("http://scorer", captured, engine_name="sglang")

    scorer = scorer_engine_spec(
        EngineSpec(
            image="sha256:" + ("a" * 64),
            name="sglang",
            serve_args=["--context-length", str(context_len)],
        )
    )
    context_len = int(scorer.serve_args[1])
    cfg = CorrectnessConfig(
        1, CorrectnessThresholds(-4.0, -12.0, 0.001, 0.5, max_mean_logprob_drop=1.5)
    )
    reports = grade_all(
        "http://scorer",
        [
            PendingCorrectness(0, [captured]),
            PendingCorrectness(BASELINE_INDEX, [captured]),
        ],
        cfg=cfg,
        evidence_dir=tmp_path,
        engine_name="sglang",
    )
    for report in reports.values():
        assert report.verdict == "pass"
        assert report.num_positions_scored == 5120
        assert report.coverage_ratio == 1.0
        assert report.mean_logprob == pytest.approx((-0.1 * 5119 - 0.7) / 5120)
