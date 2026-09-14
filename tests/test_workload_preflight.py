"""No-GPU checks for tokenizer agreement and pinned engine capacity bounds."""

import pytest

from bench.lifecycle import EngineError
from bench.validate import validate_workload_trace_dict
from bench.workload_preflight import validate_engine_workload
from test_trajectory_sampling import context, formatter, sample
import json


def run_check(monkeypatch, tmp_path, engine="sglang", **changes):
    trace = validate_workload_trace_dict(
        json.loads(sample(sampling_context=context(engine)).body)
    )
    metadata = {
        "context_length": 128,
        "max_req_input_len": 122,
        "max_total_num_tokens": 1024,
        "page_size": 16,
        "data": [{"max_model_len": 128}],
        **changes,
    }
    monkeypatch.setattr("bench.workload_preflight.get_json", lambda *_: metadata)
    tokenize = formatter().encode
    monkeypatch.setattr(
        "bench.workload_preflight.post_json",
        lambda _url, _path, body: {"tokens": tokenize(body["prompt"])},
    )
    validate_engine_workload(
        "http://engine",
        trace,
        engine_name=engine,
        max_model_len=128,
        evidence_dir=tmp_path,
        verify_tokenizer=True,
    )


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
        {"max_req_input_len": 121},
        {"max_total_num_tokens": 140},
        {"context_length": 256},
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


def test_trusted_token_ids_must_match_not_just_the_token_count(monkeypatch, tmp_path):
    trace = validate_workload_trace_dict(json.loads(sample().body))
    monkeypatch.setattr(
        "bench.workload_preflight.get_json",
        lambda *_: {"data": [{"max_model_len": 128}]},
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
            max_model_len=128,
            evidence_dir=tmp_path,
            verify_tokenizer=True,
        )
