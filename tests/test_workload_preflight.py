"""No-GPU checks for tokenizer agreement and pinned engine capacity bounds."""

import re
from types import SimpleNamespace

import pytest

from bench import workload_preflight
from bench.lifecycle import EngineError
from bench.validate import validate_workload_trace_dict
from bench.trajectory import token_ids_sha256
from bench.workload_preflight import validate_engine_workload
from test_trajectory_sampling import TEST_CONTEXT, context, formatter, sample
import json


def run_check(monkeypatch, tmp_path, engine="sglang", **changes):
    trace = validate_workload_trace_dict(
        json.loads(sample(sampling_context=context(engine)).body)
    )
    metadata = {
        "context_length": TEST_CONTEXT,
        "max_req_input_len": TEST_CONTEXT - 6,
        "max_total_num_tokens": 65536,
        "page_size": 16,
        "data": [{"max_model_len": TEST_CONTEXT}],
        **changes,
    }

    def get_json(_url, path, **kwargs):
        reads.append((path, kwargs))
        return metadata

    reads = []
    monkeypatch.setattr("bench.workload_preflight.get_json", get_json)
    tokenize = formatter().encode
    monkeypatch.setattr(
        "bench.workload_preflight.post_json",
        lambda _url, _path, body: {"tokens": tokenize(body["prompt"])},
    )
    validate_engine_workload(
        "http://engine",
        trace,
        engine_name=engine,
        max_model_len=TEST_CONTEXT,
        evidence_dir=tmp_path,
        verify_tokenizer=True,
    )
    return reads


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_pinned_tokenization_and_capacity_pass_before_measurement(
    monkeypatch, tmp_path, engine
):
    run_check(monkeypatch, tmp_path, engine)
    assert (
        json.loads((tmp_path / "workload_preflight.json").read_text())[
            "tokenizer_verified"
        ]
        is True
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"max_req_input_len": 32768},
        {"max_total_num_tokens": 32780},
        {"context_length": TEST_CONTEXT * 2},
        {"page_size": None},
        {
            "speculative_algorithm": "EAGLE",
            "speculative_eagle_topk": 2,
            "speculative_num_steps": 4,
            "speculative_num_draft_tokens": 8,
        },
    ],
)
def test_a_baseline_that_would_shorten_the_workload_is_rejected(
    monkeypatch, tmp_path, changes
):
    with pytest.raises(EngineError):
        run_check(monkeypatch, tmp_path, **changes)


def test_sglang_server_info_waits_out_a_busy_scheduler(monkeypatch, tmp_path):
    # The 60s default timed out on a Kimi K3 round engine and voided the round.
    assert run_check(monkeypatch, tmp_path) == [("/server_info", {"timeout": 600})]


def test_trusted_token_ids_must_match_not_just_the_token_count(monkeypatch, tmp_path):
    trace = validate_workload_trace_dict(json.loads(sample().body))
    monkeypatch.setattr(
        "bench.workload_preflight.get_json",
        lambda *_: {"data": [{"max_model_len": TEST_CONTEXT}]},
    )
    monkeypatch.setattr(
        "bench.workload_preflight.post_json",
        lambda _url, _path, body: {
            "tokens": [999] * len(formatter().encode(body["prompt"]))
        },
    )
    with pytest.raises(EngineError, match="tokenization differs"):
        validate_engine_workload(
            "http://engine",
            trace,
            engine_name="vllm",
            max_model_len=TEST_CONTEXT,
            evidence_dir=tmp_path,
            verify_tokenizer=True,
        )


def test_kimi_prompts_are_tokenized_one_generation_window_at_a_time(
    monkeypatch, tmp_path
):
    # A toy tokenizer with Kimi's two paths: generation windows the whole prompt
    # from its start; /tokenize windows each piece between special tokens.
    window = 10
    monkeypatch.setitem(
        workload_preflight.TOKENIZE_WINDOWS, "kimi-400000-25000-v1", window
    )

    def words(text):
        # "<S>" is a special token; other words are ordinary tokens.
        return [
            0 if w == "<S>" else 1 + hash(w) % 1000
            for w in re.split(r"(<S>)| ", text)
            if w
        ]

    def generation(text):
        return [
            t for i in range(0, len(text), window) for t in words(text[i : i + window])
        ]

    def tokenize_endpoint(text):
        return [
            t
            for part in re.split(r"(<S>)", text)
            for i in range(0, len(part), window)
            for t in words(part[i : i + window])
        ]

    prompt = "<S>alpha beta gamma delta epsilon"
    ids = generation(prompt)
    assert tokenize_endpoint(prompt) != ids
    trace = SimpleNamespace(
        meta=SimpleNamespace(
            sampling={
                "context": {"max_model_len": TEST_CONTEXT, "engine_name": "vllm"},
                "tokenizer": {"windowing": "kimi-400000-25000-v1"},
            }
        ),
        requests=[
            SimpleNamespace(
                id="hf-000",
                prompt=prompt,
                input_tokens=len(ids),
                input_ids_sha256=token_ids_sha256(ids),
            )
        ],
    )
    monkeypatch.setattr(
        "bench.workload_preflight.get_json",
        lambda *_: {"data": [{"max_model_len": TEST_CONTEXT}]},
    )
    monkeypatch.setattr(
        "bench.workload_preflight.post_json",
        lambda _url, _path, body: {"tokens": tokenize_endpoint(body["prompt"])},
    )
    validate_engine_workload(
        "http://engine",
        trace,
        engine_name="vllm",
        max_model_len=TEST_CONTEXT,
        evidence_dir=tmp_path,
        verify_tokenizer=True,
    )
