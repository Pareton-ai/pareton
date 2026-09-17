"""LongWriter input isolation, replay, qualification and natural output contracts."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from bench.lifecycle import EngineError
from bench.longform import (
    qualification_contract,
    require_qualification,
    sampling_context_for_campaign,
    validate_natural_baseline,
)
from bench.qualify_longform import evaluate_response, qualify
from bench.sampler import (
    SamplerError,
    build_prompt_formatter,
    generate_trace,
    parse_sampling_rule,
)
from bench.validate import RequestValidationError, validate_workload_trace_dict
from round.create import try_create_round
from worker.round_job import RoundInfraError, materialize_round_trace

pytestmark = pytest.mark.unit


def rule(**kw):
    return {
        "type": "hf_rows",
        "algo_version": 4,
        "dataset": "zai-org/LongWriter-6k",
        "revision": "a" * 40,
        "n_rows": 40,
        "n_prompts": 4,
        "max_tokens": 10,
        "min_reference_tokens": 8,
        "min_output_tokens": 6,
        "enable_thinking": False,
        "request_interval_ms": 2,
        **kw,
    }


def formatter(workload_rule=None):
    vocab = {"[UNK]": 0, "user": 1, "assistant": 2, "ref": 3}
    vocab.update({f"prompt{i}": i + 4 for i in range(6000)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return build_prompt_formatter(
        workload_rule or rule(),
        model_repo="test/model",
        model_revision="b" * 40,
        config_loader=lambda **_: {
            "chat_template": "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}assistant{% if enable_thinking %} THINK{% endif %}"
        },
        tokenizer_loader=lambda **_: tokenizer.to_str(),
    )


def row(i, references=8):
    return {
        "messages": [
            {"role": "user", "content": f"prompt{i}"},
            {"role": "assistant", "content": " ".join(["ref"] * references)},
        ]
    }


def fields():
    return {
        "sampling_rule": rule(),
        "engine": {"name": "sglang"},
        "bench": {
            "model": {
                "hf_repo": "test/model",
                "hf_revision": "b" * 40,
                "max_model_len": 8192,
            },
            "baseline_engine_image_digest": "sha256:" + "e" * 64,
        },
    }


def sample(**kwargs):
    f = fields()
    return generate_trace(
        **{
            "rule": rule(),
            "seed_hex": "c" * 64,
            "row_fetcher": row,
            "prompt_formatter": formatter(),
            "sampling_context": sampling_context_for_campaign(f["bench"], f["engine"]),
            **kwargs,
        }
    )


def test_original_short_prompts_no_reference_no_padding_no_forced_tail():
    sampled = sample()
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert [r.arrival_offset_ms for r in trace.requests] == [0, 2, 4, 6]
    for request, index in zip(trace.requests, sampled.row_indices, strict=True):
        assert request.prompt == f"user prompt{index} assistant"
        assert request.input_tokens == 3
        assert request.input_length_group is None
        assert request.max_tokens == 10
        assert request.sampling.ignore_eos is False
    assert sampled.receipt["enable_thinking"] is False
    fetched = []
    rebuilt = sample(
        sampling_receipt=sampled.receipt,
        row_fetcher=lambda i: (fetched.append(i), row(i))[1],
    )
    assert rebuilt.body == sampled.body
    assert fetched == list(sampled.row_indices)


@pytest.mark.parametrize(
    "change", ["prompt", "reference", "tokenizer", "thinking", "rows", "context"]
)
def test_replay_rejects_changed_source_or_contract(change):
    original = sample()
    receipt = copy.deepcopy(original.receipt)
    fetcher = row
    if change in ("prompt", "reference"):

        def fetcher(i):
            value = row(i)
            value["messages"][0 if change == "prompt" else 1]["content"] += " changed"
            return value
    elif change == "tokenizer":
        receipt["tokenizer"]["sha256"] = "sha256:" + "0" * 64
    elif change == "thinking":
        receipt["enable_thinking"] = True
    elif change == "rows":
        receipt["row_indices"][0] = 99
    else:
        receipt["context"]["max_model_len"] += 1
    with pytest.raises(SamplerError):
        sample(sampling_receipt=receipt, row_fetcher=fetcher)


@pytest.mark.parametrize(
    "kw",
    [
        {"ignore_eos": True},
        {"enable_thinking": True},
        {"min_output_tokens": 11},
        {"eligible_row_indices": [0, 1, 2, 2]},
        {"eligible_row_indices": [0, 1, 2, 99]},
        {"eligible_row_indices": [3, 2, 1, 0]},
        {"min_reference_tokens": True},
        {"min_tokens": 8},
        {"qualification": {}},
    ],
)
def test_rule_rejects_forcing_and_invalid_qualification(kw):
    with pytest.raises(SamplerError):
        parse_sampling_rule(rule(**kw))


def test_reference_filter_and_duplicate_prompts_cannot_silently_shorten_workload():
    with pytest.raises(SamplerError, match="insufficient distinct"):
        sample(row_fetcher=lambda i: row(i, references=7))
    with pytest.raises(SamplerError, match="insufficient distinct"):
        sample(row_fetcher=lambda i: row(0))
    chosen = sample(rule=rule(eligible_row_indices=[2, 4, 6, 8]))
    assert set(chosen.row_indices) == {2, 4, 6, 8}


def test_qwen_fixture_uses_longwriter_and_preserves_output_ceiling():
    directory = (
        Path(__file__).resolve().parents[1] / "fixtures/campaigns/sglang_qwen38_27b"
    )
    f = json.loads((directory / "campaign-fields.json").read_text())
    r = json.loads((directory / "sampling_rule.json").read_text())
    assert f["sampling_rule"] == r
    assert r["dataset"] == "zai-org/LongWriter-6k"
    assert r["max_tokens"] == 5120
    sampled = sample(
        rule=r,
        prompt_formatter=formatter(r),
        row_fetcher=lambda i: row(i, 5120),
        sampling_context=sampling_context_for_campaign(f["bench"], f["engine"]),
    )
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert len(trace.requests) == 32
    assert trace.meta.sampling["enable_thinking"] is False
    assert all(
        not r.sampling.ignore_eos and r.max_tokens == 5120 for r in trace.requests
    )
    with pytest.raises(SamplerError, match="baseline qualification"):
        require_qualification(parse_sampling_rule(r), f["bench"], f["engine"])


def response(tokens=10, finish="length", text="A detailed, coherent answer."):
    return {
        "choices": [{"text": text, "finish_reason": finish}],
        "usage": {"completion_tokens": tokens, "prompt_tokens": 3},
    }


def test_qualification_distinguishes_early_eos_from_natural_generation_at_the_cap():
    assert (
        evaluate_response(response(3, "stop"), rule(), 3)["rejection"] == "short_output"
    )
    assert evaluate_response(response(7, "stop"), rule(), 3)["rejection"] is None
    assert evaluate_response(response(), rule(), 3)["rejection"] is None
    assert (
        evaluate_response(response(text=""), rule(), 3)["rejection"] == "empty_output"
    )
    assert evaluate_response(response(text="repeat this endlessly " * 500), rule(), 3)[
        "rejection"
    ]
    with pytest.raises(EngineError):
        evaluate_response(response(3, "length"), rule(), 3)


def test_qualified_artifact_is_bound_to_campaign_and_never_requests_forcing(
    tmp_path, monkeypatch
):
    f = fields()
    calls = []
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **k: None
    )

    def post(url, path, body, **kw):
        calls.append(body)
        assert body["ignore_eos"] is False and "min_tokens" not in body
        assert "ref" not in body["prompt"] and "THINK" not in body["prompt"]
        return response()

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    qualified = qualify(
        fields=f,
        base_url="http://baseline",
        output_dir=tmp_path,
        pool_size=4,
        max_rows=40,
        repetitions=2,
        row_fetcher=row,
        formatter=formatter(),
    )
    assert len(calls) == 8
    assert len(qualified["eligible_row_indices"]) == 4
    assert qualified["qualification"]["contract_sha256"] == qualification_contract(
        qualified, f["bench"], f["engine"]
    )
    require_qualification(qualified, f["bench"], f["engine"])
    f["bench"]["model"]["hf_revision"] = "d" * 40
    with pytest.raises(SamplerError, match="baseline qualification"):
        require_qualification(qualified, f["bench"], f["engine"])


def test_short_outputs_leave_evidence_but_no_launch_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "bench.qualify_longform.post_json", lambda *a, **k: response(2, "stop")
    )
    with pytest.raises(SamplerError, match="no launch rule written"):
        qualify(
            fields=fields(),
            base_url="http://baseline",
            output_dir=tmp_path,
            pool_size=4,
            max_rows=4,
            row_fetcher=row,
            formatter=formatter(),
        )
    assert (tmp_path / "qualification.jsonl").exists()
    assert not (tmp_path / "sampling_rule.json").exists()


def test_every_measured_baseline_repetition_must_stay_long():
    trace = validate_workload_trace_dict(json.loads(sample().body))
    replay = SimpleNamespace(
        completion_token_samples={r.id: (10, 10, 10) for r in trace.requests}
    )
    validate_natural_baseline(trace, replay)
    replay.completion_token_samples[trace.requests[0].id] = (10, 2, 10)
    with pytest.raises(EngineError, match="requalify"):
        validate_natural_baseline(trace, replay)


def test_short_baseline_aborts_round_before_candidate_start(monkeypatch, tmp_path):
    from bench.main import run_round
    from bench.output import OutputLayout
    from bench.validate import validate_bench_request_dict

    request = validate_bench_request_dict(
        json.loads(Path("fixtures/bench/sample_request.json").read_text())
    )
    trace = validate_workload_trace_dict(json.loads(sample().body))
    starts = []

    @contextmanager
    def start(engine, **_):
        starts.append(engine.kind)
        yield "http://baseline"

    monkeypatch.setattr(
        "bench.workload_preflight.validate_engine_workload", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "bench.main.run_sla_engine",
        lambda *a, **k: SimpleNamespace(
            completion_token_samples={r.id: (10, 2, 10) for r in trace.requests}
        ),
    )
    layout = OutputLayout(tmp_path)
    layout.prepare()
    with pytest.raises(EngineError, match="requalify") as caught:
        run_round(
            req=request,
            provider=SimpleNamespace(start=start),
            prompts=[],
            trace=trace,
            layout=layout,
        )
    assert caught.value.error_role == "baseline"
    assert starts == ["baseline"]
    status = json.loads(layout.entry_status_path.read_text())
    assert status["entries"]["baseline"]["status"] == "infra_failed"


@pytest.mark.parametrize(
    "mutation", ["forced", "thinking", "allowance", "arrival", "input"]
)
def test_trace_validation_rejects_changed_generation_contract(mutation):
    trace = json.loads(sample().body)
    request = trace["requests"][0]
    if mutation == "forced":
        request["sampling"]["ignore_eos"] = True
    elif mutation == "thinking":
        trace["meta"]["sampling"]["enable_thinking"] = True
    elif mutation == "allowance":
        request["max_tokens"] = 1
    elif mutation == "arrival":
        request["arrival_offset_ms"] = 9
    else:
        request["input_tokens"] = 8192
    with pytest.raises(RequestValidationError):
        validate_workload_trace_dict(trace)


def test_round_creation_and_worker_replay_preserve_qualified_rows(
    monkeypatch, tmp_path
):
    f = fields()
    campaign = SimpleNamespace(
        campaign_id=uuid4(),
        gpu_skus=["RTX5090"],
        sampling_rule=rule(eligible_row_indices=[2, 4, 6, 8]),
        scoring_rule={"name": "median_e2e_speedup", "failure_penalty": 0.1},
        bench=f["bench"],
        engine=f["engine"],
    )
    monkeypatch.setattr("round.create.create_round", lambda **kw: kw)
    result = try_create_round(
        campaign,
        {"queued": 100, "oldest_queued_at": None},
        seed_block=10,
        seed_block_hash="a" * 64,
        row_fetcher=row,
        prompt_formatter=formatter(),
    )
    path = materialize_round_trace(
        result, campaign, tmp_path, row_fetcher=row, prompt_formatter=formatter()
    )
    trace = validate_workload_trace_dict(json.loads(path.read_bytes()))
    assert trace.meta.sampling["min_output_tokens"] == 6
    assert set(result["sampling_receipt"]["row_indices"]) == {2, 4, 6, 8}
    assert all(not request.sampling.ignore_eos for request in trace.requests)
    campaign.bench["model"]["max_model_len"] *= 2
    with pytest.raises(RoundInfraError, match="sampling receipt"):
        materialize_round_trace(
            result, campaign, tmp_path, row_fetcher=row, prompt_formatter=formatter()
        )
