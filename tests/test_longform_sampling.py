"""LongWriter history tiers, replay, qualification and natural output contracts."""

import copy
import hashlib
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


@pytest.fixture
def baseline_identity(monkeypatch):
    identity = {"engine_ref": "sha256:" + "e" * 64, "container_id": "baseline"}
    monkeypatch.setattr(
        "bench.qualify_longform.verify_baseline_image", lambda **kw: dict(identity)
    )
    return identity


def rule(**kw):
    return {
        "type": "hf_rows",
        "algo_version": 4,
        "dataset": "zai-org/LongWriter-6k",
        "revision": "a" * 40,
        "n_rows": 40,
        "n_prompts": 4,
        "max_tokens": 10,
        "followup_prompt": "another",
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


def row(i, references=None):
    if references is None:
        references = (1900, 3800, 7600, 15200)[i % 4] - 6
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
                "max_model_len": 32768,
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


def test_source_exchange_and_followup_fill_tiers_without_padding_or_forcing():
    sampled = sample()
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert [r.arrival_offset_ms for r in trace.requests] == [0, 2, 4, 6]
    for request, index in zip(trace.requests, sampled.row_indices, strict=True):
        assert request.prompt == formatter().render(
            row(index)["messages"] + [{"role": "user", "content": "another"}]
        )
        assert request.input_tokens == (1900, 3800, 7600, 15200)[index % 4]
        assert request.input_length_group == ("2k", "4k", "8k", "16k")[index % 4]
        assert request.max_tokens == 10
        assert request.sampling.ignore_eos is False
    assert {r.input_length_group for r in trace.requests} == {"2k", "4k", "8k", "16k"}
    assert sampled.receipt["enable_thinking"] is False
    fetched = []
    rebuilt = sample(
        sampling_receipt=sampled.receipt,
        row_fetcher=lambda i: (fetched.append(i), row(i))[1],
    )
    assert rebuilt.body == sampled.body
    assert fetched == list(sampled.row_indices)


def test_legacy_v4_trace_is_byte_identical_without_temperature():
    sampled = sample()
    assert sampled.sha256 == (
        "sha256:a7f1a51d2e934924046f4fa4da83e0b7fb2e5953137495fd3f4d8e136b4ee9b4"
    )
    assert "temperature" not in sampled.receipt
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert all(r.sampling.temperature == 0.0 for r in trace.requests)


def test_temperature_is_pinned_in_trace_receipt_and_qualification():
    r = rule(temperature=0.7)
    sampled = sample(rule=r)
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert all(r.sampling.temperature == 0.7 for r in trace.requests)
    assert trace.meta.sampling["temperature"] == 0.7
    assert sampled.receipt["temperature"] == 0.7
    assert sample(rule=r, sampling_receipt=sampled.receipt).body == sampled.body
    with pytest.raises(SamplerError):
        sample(rule=rule(temperature=0.0), sampling_receipt=sampled.receipt)
    for change in ("request", "metadata", "missing_metadata"):
        data = json.loads(sampled.body)
        if change == "request":
            data["requests"][0]["sampling"]["temperature"] = 0.0
        elif change == "metadata":
            data["meta"]["sampling"]["temperature"] = 0.0
        else:
            del data["meta"]["sampling"]["temperature"]
        with pytest.raises(RequestValidationError):
            validate_workload_trace_dict(data)


@pytest.mark.parametrize(
    "temperature", [True, "0.7", None, -0.1, 2.1, float("nan"), float("inf")]
)
def test_longform_rejects_invalid_temperature(temperature):
    with pytest.raises(SamplerError, match="temperature"):
        parse_sampling_rule(rule(temperature=temperature))


@pytest.mark.parametrize(
    "change",
    [
        {"temperature_range": [0.1, 1.5], "temperature": 0.7},
        {"temperature_range": [1.5, 0.1]},
        {"temperature_range": [0.1]},
        {"temperature_range": [0.1, float("nan")]},
        {"temperature_range": [True, 1.5]},
        {"randomize_seed": True},
    ],
)
def test_invalid_randomized_generation_policy_is_rejected(change):
    with pytest.raises(SamplerError):
        parse_sampling_rule(rule(**change))


def test_random_temperature_is_bound_to_round_trace_and_receipt():
    r = rule(temperature_range=[0.1, 1.5])
    sampled = sample(rule=r)
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert all(0.1 <= req.sampling.temperature <= 1.5 for req in trace.requests)
    assert len({req.sampling.temperature for req in trace.requests}) == 4
    assert all(
        "seed" not in req["sampling"] for req in json.loads(sampled.body)["requests"]
    )
    assert sample(rule=r, sampling_receipt=sampled.receipt).body == sampled.body
    other = validate_workload_trace_dict(
        json.loads(sample(rule=r, seed_hex="d" * 64).body)
    )
    assert other.requests[0].sampling != trace.requests[0].sampling
    for field in ("temperature", "seed", "generation_seed", "range"):
        data = json.loads(sampled.body)
        if field == "generation_seed":
            data["meta"]["sampling"].pop(field)
        elif field == "range":
            data["meta"]["sampling"]["temperature_range"] = [0.2, 1.4]
        else:
            data["requests"][0]["sampling"][field] = 0
        with pytest.raises(RequestValidationError):
            validate_workload_trace_dict(data)
    with pytest.raises(SamplerError):
        sample(
            rule={**r, "temperature_range": [0.2, 1.4]},
            sampling_receipt=sampled.receipt,
        )


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
        {"n_prompts": 6},
        {"followup_prompt": ""},
        {"qualification": {}},
    ],
)
def test_rule_rejects_forcing_and_invalid_qualification(kw):
    with pytest.raises(SamplerError):
        parse_sampling_rule(rule(**kw))


def test_missing_tiers_and_duplicate_prompts_cannot_silently_shorten_workload():
    with pytest.raises(SamplerError, match="insufficient distinct"):
        sample(row_fetcher=lambda i: row(i, references=7))
    with pytest.raises(SamplerError, match="insufficient distinct"):
        sample(row_fetcher=lambda i: row(0))
    chosen = sample(rule=rule(eligible_row_indices=[0, 1, 2, 3]))
    assert set(chosen.row_indices) == {0, 1, 2, 3}


def test_qwen_fixture_uses_longwriter_and_preserves_output_ceiling():
    directory = (
        Path(__file__).resolve().parents[1] / "fixtures/campaigns/sglang_qwen38_27b"
    )
    f = json.loads((directory / "campaign-fields.json").read_text())
    r = json.loads((directory / "sampling_rule.json").read_text())
    assert f["sampling_rule"] == r
    assert r["dataset"] == "zai-org/LongWriter-6k"
    assert r["max_tokens"] == 5120
    assert r["min_output_tokens"] == 3000
    assert r["temperature_range"] == [0.1, 1.01]
    assert f["bench"]["correctness"]["thresholds"]["max_mean_logprob_drop"] == 2.5
    assert "randomize_seed" not in r
    omitted_floor = {
        key: value for key, value in r.items() if key != "min_output_tokens"
    }
    assert parse_sampling_rule(omitted_floor)["min_output_tokens"] == 3000
    sampled = sample(
        rule=r,
        prompt_formatter=formatter(r),
        row_fetcher=row,
        sampling_context=sampling_context_for_campaign(f["bench"], f["engine"]),
    )
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert len(trace.requests) == 32
    assert trace.meta.sampling["enable_thinking"] is False
    assert trace.meta.sampling["min_output_tokens"] == 3000
    assert all(
        not r.sampling.ignore_eos
        and r.max_tokens == 5120
        and 0.1 <= r.sampling.temperature <= 1.5
        for r in trace.requests
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


@pytest.mark.parametrize("temperature", [None, 0.7, [0.1, 1.5]])
def test_qualified_artifact_is_bound_to_campaign_and_never_requests_forcing(
    tmp_path, monkeypatch, baseline_identity, temperature
):
    f = fields()
    if isinstance(temperature, list):
        f["sampling_rule"].update(temperature_range=temperature)
    elif temperature is not None:
        f["sampling_rule"]["temperature"] = temperature
    calls = []
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **k: None
    )

    def post(url, path, body, **kw):
        calls.append(body)
        assert body["ignore_eos"] is False and "min_tokens" not in body
        assert body["top_p"] == 1.0
        if isinstance(temperature, list):
            assert body["temperature"] in temperature
            assert body["seed"] == 0
        else:
            assert body["temperature"] == (0.0 if temperature is None else temperature)
            assert body["seed"] == 0
        assert "ref" in body["prompt"] and "THINK" not in body["prompt"]
        result = response()
        result["usage"]["prompt_tokens"] = len(formatter().encode(body["prompt"]))
        return result

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    qualified = qualify(
        fields=f,
        base_url="http://baseline",
        container="baseline",
        engine_ref="sha256:" + "e" * 64,
        output_dir=tmp_path,
        pool_size=4,
        max_rows=40,
        repetitions=2,
        row_fetcher=row,
        formatter=formatter(),
    )
    assert len(calls) == 8
    if isinstance(temperature, list):
        assert {call["seed"] for call in calls} == {0}
        for prompt in {call["prompt"] for call in calls}:
            assert [
                c["temperature"] for c in calls if c["prompt"] == prompt
            ] == temperature
    assert len(qualified["eligible_row_indices"]) == 4
    assert "evidence_sha256" not in qualified["qualification"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["evidence_sha256"] == (
        "sha256:"
        + hashlib.sha256((tmp_path / "qualification.jsonl").read_bytes()).hexdigest()
    )
    assert qualified["qualification"]["contract_sha256"] == qualification_contract(
        qualified, f["bench"], f["engine"]
    )
    require_qualification(qualified, f["bench"], f["engine"])
    changed = (
        {**qualified, "temperature_range": [0.2, 1.4]}
        if isinstance(temperature, list)
        else {**qualified, "temperature": 0.7 if temperature is None else 0.0}
    )
    with pytest.raises(SamplerError, match="baseline qualification"):
        require_qualification(changed, f["bench"], f["engine"])
    f["bench"]["model"]["hf_revision"] = "d" * 40
    with pytest.raises(SamplerError, match="baseline qualification"):
        require_qualification(qualified, f["bench"], f["engine"])


def test_short_outputs_leave_evidence_but_no_launch_rule(
    tmp_path, monkeypatch, baseline_identity
):
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "bench.qualify_longform.post_json",
        lambda url, path, body, **k: {
            **response(2, "stop"),
            "usage": {
                "completion_tokens": 2,
                "prompt_tokens": len(formatter().encode(body["prompt"])),
            },
        },
    )
    with pytest.raises(SamplerError, match="no launch rule written"):
        qualify(
            fields=fields(),
            base_url="http://baseline",
            container="baseline",
            engine_ref="sha256:" + "e" * 64,
            output_dir=tmp_path,
            pool_size=4,
            max_rows=4,
            row_fetcher=row,
            formatter=formatter(),
        )
    assert (tmp_path / "qualification.jsonl").exists()
    assert not (tmp_path / "sampling_rule.json").exists()


def test_every_measured_baseline_repetition_must_stay_long():
    r = rule(max_tokens=5120, min_output_tokens=3000)
    trace = validate_workload_trace_dict(json.loads(sample(rule=r).body))
    assert evaluate_response(response(3000, "stop"), r, 3)["rejection"] is None
    assert (
        evaluate_response(response(2999, "stop"), r, 3)["rejection"] == "short_output"
    )
    from bench.correctness import baseline_prompt_drops

    replay = SimpleNamespace(
        completion_token_samples={r.id: (5120, 3000, 4000) for r in trace.requests},
        output_samples={r.id: ("Clean answer.",) * 3 for r in trace.requests},
        result=SimpleNamespace(role="baseline"),
    )
    assert baseline_prompt_drops(trace, replay, dropped={}) == {}
    replay.completion_token_samples[trace.requests[0].id] = (5120, 2999, 4000)
    dropped = baseline_prompt_drops(trace, replay, dropped={})
    assert list(dropped) == [trace.requests[0].id]
    assert "2999" in dropped[trace.requests[0].id]


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
            completion_token_samples={r.id: (10, 2, 10) for r in trace.requests},
            output_samples={r.id: ("Clean answer.",) * 3 for r in trace.requests},
            result=SimpleNamespace(role="baseline"),
        ),
    )
    layout = OutputLayout(tmp_path)
    layout.prepare()
    with pytest.raises(EngineError, match="no stable workload prompts") as caught:
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
        request["input_tokens"] = 32768
    with pytest.raises(RequestValidationError):
        validate_workload_trace_dict(trace)


@pytest.mark.parametrize(
    "generation_policy",
    [{}, {"temperature": 0.7}, {"temperature_range": [0.1, 1.01]}],
    ids=["legacy-greedy", "fixed-temperature", "temperature-range"],
)
def test_round_creation_and_worker_replay_preserve_qualified_rows(
    monkeypatch, tmp_path, generation_policy
):
    f = fields()
    campaign = SimpleNamespace(
        campaign_id=uuid4(),
        gpu_skus=["RTX5090"],
        sampling_rule=rule(eligible_row_indices=[0, 1, 2, 3], **generation_policy),
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
    assert (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        == result["sampled_trace_sha256"]
    )
    trace = validate_workload_trace_dict(json.loads(path.read_bytes()))
    for key, value in generation_policy.items():
        assert trace.meta.sampling[key] == value
    assert trace.meta.sampling["min_output_tokens"] == 6
    assert set(result["sampling_receipt"]["row_indices"]) == {0, 1, 2, 3}
    assert all(not request.sampling.ignore_eos for request in trace.requests)
    campaign.bench["model"]["max_model_len"] *= 2
    with pytest.raises(RoundInfraError, match="sampling receipt"):
        materialize_round_trace(
            result, campaign, tmp_path, row_fetcher=row, prompt_formatter=formatter()
        )


def test_qualification_requires_long_outputs_in_every_tier(
    tmp_path, monkeypatch, baseline_identity
):
    fmt = formatter()
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **k: None
    )
    calls = []

    def post(url, path, body, **kw):
        size = len(fmt.encode(body["prompt"]))
        calls.append(size)
        result = response(2, "stop") if size > 10000 else response()
        result["usage"]["prompt_tokens"] = size
        return result

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    with pytest.raises(SamplerError, match="input tiers"):
        qualify(
            fields=fields(),
            base_url="http://baseline",
            container="baseline",
            engine_ref="sha256:" + "e" * 64,
            output_dir=tmp_path,
            pool_size=4,
            max_rows=40,
            row_fetcher=row,
            formatter=fmt,
        )
    assert len([size for size in calls if size <= 10000]) == 0
    assert len([size for size in calls if size > 10000]) == 10
    assert not (tmp_path / "sampling_rule.json").exists()


@pytest.mark.parametrize("mutation", ["label", "quota", "metadata", "followup"])
def test_tier_contract_and_followup_are_bound_to_trace_or_receipt(mutation):
    sampled = sample()
    trace = json.loads(sampled.body)
    if mutation == "followup":
        with pytest.raises(SamplerError, match="receipt"):
            sample(
                rule=rule(followup_prompt="changed"), sampling_receipt=sampled.receipt
            )
        return
    if mutation == "label":
        trace["requests"][0]["input_length_group"] = "32k"
    elif mutation == "quota":
        for request in trace["requests"]:
            request.update(input_tokens=1900, input_length_group="2k")
    else:
        del trace["meta"]["sampling"]["length_groups"]
    with pytest.raises(RequestValidationError):
        validate_workload_trace_dict(trace)


def test_cpu_preview_matches_round_sampling_and_exposes_messages(tmp_path, monkeypatch):
    from bench.preview_longform import preview
    from bench.sampler import compute_sample_seed

    monkeypatch.setattr(
        "bench.preview_longform.build_prompt_formatter", lambda *a, **k: formatter()
    )
    monkeypatch.setattr("bench.preview_longform.fetch_hf_row", lambda r, i: row(i))
    campaign_id = uuid4()
    root = tmp_path / "preview"
    actual = preview(
        fields=fields(),
        output_dir=root,
        campaign_id=campaign_id,
        seed_block=10,
        block_hash="a" * 64,
    )
    expected = sample(
        seed_hex=compute_sample_seed(block_hash="a" * 64, campaign_id=campaign_id),
        sample_seed_block=10,
        sample_seed_block_hash="a" * 64,
    )
    assert actual.body == expected.body
    assert actual.receipt == expected.receipt
    assert len((root / "index.tsv").read_text().splitlines()) == 5
    for i in range(4):
        messages = json.loads((root / f"hf-{i:03d}.messages.json").read_text())
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[-1]["content"] == "another"
        assert (root / f"hf-{i:03d}.prompt.txt").read_text() == formatter().render(
            messages
        )


def test_qualification_concurrency_is_bounded_and_evidence_is_complete(
    tmp_path, monkeypatch, baseline_identity
):
    import threading
    from collections import Counter

    fmt = formatter()
    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    active = peak = 0
    tiers = []
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **k: None
    )

    def post(url, path, body, **kw):
        nonlocal active, peak
        size = len(fmt.encode(body["prompt"]))
        with lock:
            active += 1
            peak = max(peak, active)
            tiers.append(size)
        try:
            barrier.wait()
            result = response()
            result["usage"]["prompt_tokens"] = size
            return result
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    qualified = qualify(
        fields=fields(),
        base_url="http://baseline",
        container="baseline",
        engine_ref="sha256:" + "e" * 64,
        output_dir=tmp_path,
        pool_size=8,
        max_rows=40,
        concurrency=2,
        row_fetcher=row,
        formatter=fmt,
    )
    assert peak == 2
    assert tiers == [15200] * 4 + [7600] * 4 + [3800] * 4 + [1900] * 4
    records = [
        json.loads(line)
        for line in (tmp_path / "qualification.jsonl").read_text().splitlines()
    ]
    assert records[0]["concurrency"] == 2
    assert len(records) == 17
    assert Counter(record["row_index"] for record in records[1:]) == {
        index: 2 for index in qualified["eligible_row_indices"]
    }
    assert all(record["rejection"] is None for record in records[1:])
    assert json.loads((tmp_path / "summary.json").read_text())["concurrency"] == 2


def test_qualification_stops_when_scarce_tier_cannot_fill(
    tmp_path, monkeypatch, baseline_identity
):
    f = fields()
    f["sampling_rule"] = rule(n_rows=100, n_prompts=32)
    fmt = formatter()
    calls = []
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **k: None
    )

    def post(url, path, body, **kw):
        size = len(fmt.encode(body["prompt"]))
        calls.append(size)
        result = response(2, "stop")
        result["usage"]["prompt_tokens"] = size
        return result

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    with pytest.raises(SamplerError, match="16k cannot fill 16 slots"):
        qualify(
            fields=f,
            base_url="http://baseline",
            container="baseline",
            engine_ref="sha256:" + "e" * 64,
            output_dir=tmp_path,
            pool_size=64,
            max_rows=100,
            concurrency=1,
            row_fetcher=row,
            formatter=fmt,
        )
    assert calls == [15200] * 10
    assert not (tmp_path / "sampling_rule.json").exists()


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_qualification_rejects_invalid_concurrency(tmp_path, concurrency):
    with pytest.raises(SamplerError, match="concurrency"):
        qualify(
            fields=fields(),
            base_url="http://baseline",
            container="baseline",
            engine_ref="sha256:" + "e" * 64,
            output_dir=tmp_path,
            concurrency=concurrency,
        )
    assert not (tmp_path / "qualification.jsonl").exists()
