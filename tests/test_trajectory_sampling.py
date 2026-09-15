"""Offline coverage of versioned history construction and round reconstruction."""

import copy
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from bench.sampler import (
    SamplerError,
    build_prompt_formatter,
    generate_trace,
    parse_sampling_rule,
)
from bench.trajectory import normalize_trajectory, sampling_context_for_campaign
from bench.validate import RequestValidationError, validate_workload_trace_dict
from round.create import try_create_round
from worker.round_job import RoundInfraError, materialize_round_trace

pytestmark = pytest.mark.unit


def rule(**overrides):
    return {
        "type": "hf_rows",
        "dataset": "nebius/SWE-agent-trajectories",
        "revision": "a" * 40,
        "n_rows": 12,
        "n_prompts": 8,
        "max_tokens": 80,
        "algo_version": 3,
        **overrides,
    }


def formatter(thinking=False):
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "user": 1, "assistant": 2, "end": 3, "think": 4, "data": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return build_prompt_formatter(
        rule(enable_thinking=thinking),
        model_repo="test/model",
        model_revision="b" * 40,
        config_loader=lambda **_: {
            "chat_template": "{% if enable_thinking %}system reason end {% endif %}{% for m in messages %}{{ m.role }} {{ m.content }} end {% endfor %}assistant think"
        },
        tokenizer_loader=lambda **_: tokenizer.to_str(),
    )


def row(thinking=False, long_characters=False):
    # Complete user prefixes render to 32, 64, 96 and 121 tokens, including
    # role markers, generation prefix and the optional template instruction.
    first = 28 - (3 if thinking else 0)
    messages = [{"role": "system", "system_prompt": "DATASET_SYSTEM_SECRET"}]
    for i, words in enumerate((first, 27, 27, 20)):
        content = " ".join(["data"] * words)
        if i == 0 and long_characters:
            content = "x" * 9000 + content[4:]
        messages += [{"role": "user", "text": content}, {"role": "ai", "text": "data"}]
    return {
        "trajectory": messages,
        "generated_patch": "FUTURE_PATCH",
        "eval_logs": "FUTURE_TEST_RESULTS",
    }


def context(engine="vllm"):
    return sampling_context_for_campaign(
        {"model": {"max_model_len": 128}}, {"name": engine}
    )


def sample(**overrides):
    kwargs = {
        "rule": rule(),
        "seed_hex": "c" * 64,
        "row_fetcher": lambda _: row(),
        "prompt_formatter": formatter(),
        "sampling_context": context(),
    }
    kwargs.update(overrides)
    return generate_trace(**kwargs)


@pytest.mark.parametrize("interval", [0, 2, 200])
@pytest.mark.parametrize("n_prompts", [4, 10, 32])
def test_trace_covers_lengths_with_one_frozen_arrival_schedule(interval, n_prompts):
    workload_rule = rule(request_interval_ms=interval, n_prompts=n_prompts, n_rows=40)
    sampled = sample(rule=workload_rule)
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert len(trace.requests) == n_prompts
    assert [r.arrival_offset_ms for r in trace.requests] == [
        i * interval for i in range(n_prompts)
    ]
    counts = {4: [1, 1, 1, 1], 10: [3, 3, 2, 2], 32: [8, 8, 8, 8]}
    assert [
        sum(r.input_tokens == target for r in trace.requests)
        for target in (32, 64, 96, 121)
    ] == counts[n_prompts]
    assert {r.input_tokens: r.max_tokens for r in trace.requests} == {
        32: 80,
        64: 64,
        96: 32,
        121: 7,
    }
    assert len(set(sampled.row_indices)) == n_prompts
    assert [r.input_tokens for r in trace.requests] != sorted(
        r.input_tokens for r in trace.requests
    )
    assert sample(rule=workload_rule).body == sampled.body


def test_sglang_headroom_is_resolved_before_any_candidate_runs():
    sampled = sample(sampling_context=context("sglang"))
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert {r.input_tokens: r.max_tokens for r in trace.requests} == {
        32: 80,
        64: 62,
        96: 30,
        121: 5,
    }


def test_normalized_history_excludes_system_and_future_messages():
    source = row()
    source["trajectory"][2]["text"] = "RECORDED_COMMAND"
    source["trajectory"][-1]["text"] = "HELD_OUT_RESPONSE"
    source["trajectory"] = json.dumps(source["trajectory"])
    sampled = sample(row_fetcher=lambda _: source)
    prompts = [r["prompt"] for r in json.loads(sampled.body)["requests"]]
    assert any("assistant RECORDED_COMMAND" in prompt for prompt in prompts)
    assert all(
        "DATASET_SYSTEM_SECRET" not in p
        and "FUTURE_PATCH" not in p
        and "FUTURE_TEST_RESULTS" not in p
        and "HELD_OUT_RESPONSE" not in p
        for p in prompts
    )
    assert all(p.endswith("end assistant think") for p in prompts)


def test_more_than_8000_characters_is_accepted_when_rendered_tokens_fit():
    sampled = sample(row_fetcher=lambda _: row(long_characters=True))
    assert all(len(r["prompt"]) > 8000 for r in json.loads(sampled.body)["requests"])


def test_thinking_instruction_counts_toward_each_input():
    sampled = sample(
        rule=rule(enable_thinking=True),
        row_fetcher=lambda _: row(thinking=True),
        prompt_formatter=formatter(True),
    )
    assert sampled.receipt["enable_thinking"] is True
    for request in json.loads(sampled.body)["requests"]:
        assert request["prompt"].startswith("system reason end user")
    assert sorted(r["input_tokens"] for r in sampled.receipt["requests"]) == [
        32,
        32,
        64,
        64,
        96,
        96,
        121,
        121,
    ]


@pytest.mark.parametrize(
    "trajectory",
    [
        "invalid JSON",
        [{"role": "tool", "text": "result"}],
        [{"role": "ai", "text": "answer"}],
        [{"role": "user", "text": {"unexpected": "object"}}],
        [{"role": "user", "text": "a"}, {"role": "user", "text": "b"}],
    ],
)
def test_malformed_histories_are_ineligible(trajectory):
    assert normalize_trajectory({"trajectory": trajectory}) == []


def test_content_fallback_and_already_normalized_assistant_are_preserved():
    messages = normalize_trajectory(
        {
            "trajectory": [
                {"role": "user", "text": None, "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    )
    assert [m for _, m in messages] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_missing_coverage_fails_without_short_fallback():
    with pytest.raises(SamplerError, match="coverage unavailable"):
        sample(row_fetcher=lambda _: {"trajectory": row()["trajectory"][:3]})


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_interval_ms": -1},
        {"request_interval_ms": 1.5},
        {"request_interval_ms": True},
        {"enable_thinking": "true"},
        {"n_prompts": 3},
        {"max_tokens": 0},
        {"revision": "main"},
    ],
)
def test_invalid_v3_settings_are_rejected(overrides):
    with pytest.raises(SamplerError):
        parse_sampling_rule(rule(**overrides))


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "field,value", [("request_interval_ms", 0), ("enable_thinking", False)]
)
def test_new_fields_are_not_ignored_on_legacy_algorithms(version, field, value):
    with pytest.raises(SamplerError, match="require algo_version 3"):
        parse_sampling_rule(rule(algo_version=version, **{field: value}))


def test_receipt_reconstruction_fetches_only_the_selected_rows():
    sampled = sample()
    fetched = []

    def fetch(index):
        fetched.append(index)
        return row()

    rebuilt = sample(row_fetcher=fetch, sampling_receipt=sampled.receipt)
    assert rebuilt.body == sampled.body
    assert fetched == list(sampled.row_indices)


@pytest.mark.parametrize(
    "mutation", ["cut", "tokenizer", "tokens", "thinking", "context", "group", "shape"]
)
def test_changed_receipt_cannot_reconstruct_the_trace(mutation):
    receipt = copy.deepcopy(sample().receipt)
    if mutation == "cut":
        receipt["requests"][0]["end_message_index"] = 2
    elif mutation == "tokenizer":
        receipt["tokenizer"]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "tokens":
        receipt["requests"][0]["max_tokens"] += 1
    elif mutation == "thinking":
        receipt["enable_thinking"] = True
    elif mutation == "group":
        receipt["requests"][0]["input_length_group"] = "unknown"
    elif mutation == "shape":
        receipt["requests"][0] = None
    else:
        receipt["context"]["max_model_len"] += 1
    with pytest.raises(SamplerError):
        sample(sampling_receipt=receipt)


def test_round_creation_and_worker_materialization_share_the_v3_contract(
    monkeypatch, tmp_path
):
    campaign = SimpleNamespace(
        campaign_id=uuid4(),
        engine=None,
        gpu_skus=["H200"],
        sampling_rule=rule(request_interval_ms=2, enable_thinking=True),
        scoring_rule={"name": "median_e2e_speedup", "failure_penalty": 0.1},
        bench={
            "model": {
                "hf_repo": "test/model",
                "hf_revision": "b" * 40,
                "max_model_len": 128,
            },
            "baseline_engine_image_digest": "sha256:" + "e" * 64,
        },
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr("round.create.create_round", create)
    result = try_create_round(
        campaign,
        {"queued": 100, "oldest_queued_at": None},
        seed_block=10,
        seed_block_hash="a" * 64,
        row_fetcher=lambda _: row(True),
        prompt_formatter=formatter(True),
    )
    assert result["scoring_rule"]["failure_penalty"] == 0.1
    path = materialize_round_trace(
        result,
        campaign,
        tmp_path,
        row_fetcher=lambda _: row(True),
        prompt_formatter=formatter(True),
    )
    trace = validate_workload_trace_dict(json.loads(path.read_bytes()))
    assert trace.meta.sampling["enable_thinking"] is True
    assert trace.requests[-1].arrival_offset_ms == 14
    campaign.bench["model"]["max_model_len"] = 256
    with pytest.raises(RoundInfraError, match="context"):
        materialize_round_trace(
            result,
            campaign,
            tmp_path,
            row_fetcher=lambda _: row(True),
            prompt_formatter=formatter(True),
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_tokens", 129),
        ("input_tokens", 129),
        ("arrival_offset_ms", -1),
        ("input_ids_sha256", "unverified"),
        ("max_tokens", "7"),
        ("arrival_offset_ms", "0"),
    ],
)
def test_trace_validation_rejects_inconsistent_workload_metadata(key, value):
    trace = json.loads(sample().body)
    trace["requests"][0][key] = value
    with pytest.raises(RequestValidationError):
        validate_workload_trace_dict(trace)


def test_legacy_trace_parsing_keeps_its_existing_numeric_coercion():
    trace = validate_workload_trace_dict(
        {
            "schema_version": 1,
            "requests": [
                {
                    "id": "legacy",
                    "prompt": "hello",
                    "max_tokens": "16",
                    "arrival_offset_ms": "200",
                    "sampling": {"temperature": 0.0, "top_p": 1.0},
                }
            ],
        }
    )
    assert trace.requests[0].max_tokens == 16
    assert trace.requests[0].arrival_offset_ms == 200


@pytest.mark.parametrize("invalid_context", [None, [], "8192"])
def test_malformed_trace_context_has_a_validation_error(invalid_context):
    trace = json.loads(sample().body)
    trace["meta"]["sampling"]["context"] = invalid_context
    with pytest.raises(RequestValidationError, match="context"):
        validate_workload_trace_dict(trace)
