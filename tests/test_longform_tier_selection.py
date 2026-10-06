"""Explicit v5 tier subsets across qualification, receipts and eligibility gates."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest
from test_longform_sampling import fields, formatter, response, row, rule, sample

from bench.concurrency import request_groups, validate_tier_contract
from bench.correctness import baseline_prompt_drops
from bench.lifecycle import EngineError
from bench.longform import require_qualification
from bench.qualify_longform import qualify
from bench.sampler import SamplerError, parse_sampling_rule
from bench.validate import validate_workload_trace_dict
from worker.round_job import materialize_round_trace

pytestmark = pytest.mark.unit


def reduced_rule(**overrides):
    result = rule(
        algo_version=5,
        n_prompts=16,
        n_rows=64,
        request_concurrency=4,
        input_tiers=["8k", "16k"],
        max_baseline_prompt_drops=4,
        **overrides,
    )
    result.pop("request_interval_ms")
    return result


@pytest.mark.parametrize(
    "value", [None, [], ["8k", "8k"], ["16k", "8k"], ["32k"], "8k", [8]]
)
def test_reject_invalid_selected_tiers(value):
    with pytest.raises(SamplerError, match="input_tiers"):
        parse_sampling_rule({**reduced_rule(), "input_tiers": value})


@pytest.mark.parametrize("value", [None, -1, 9, True, 4.5, "4"])
def test_reject_invalid_exclusion_limit(value):
    with pytest.raises(SamplerError, match="max_baseline_prompt_drops"):
        parse_sampling_rule({**reduced_rule(), "max_baseline_prompt_drops": value})


def test_optional_policy_is_v5_only_and_defaults_remain_absent():
    for policy in ({"input_tiers": ["8k", "16k"]}, {"max_baseline_prompt_drops": 4}):
        with pytest.raises(SamplerError, match="version 5"):
            parse_sampling_rule({**rule(), **policy})
    historical = reduced_rule()
    historical.pop("input_tiers")
    historical.pop("max_baseline_prompt_drops")
    parsed = parse_sampling_rule(historical)
    sampled = sample(rule=parsed)
    for value in (
        parsed,
        sampled.receipt,
        json.loads(sampled.body)["meta"]["sampling"],
    ):
        assert "input_tiers" not in value and "max_baseline_prompt_drops" not in value
    assert len(json.loads(sampled.body)["meta"]["sampling"]["length_groups"]) == 4
    with pytest.raises(SamplerError, match="multiple of 2"):
        parse_sampling_rule({**reduced_rule(), "n_prompts": 15})


def test_two_tier_qualification_and_worker_receipt(tmp_path, monkeypatch):
    f = fields()
    f["sampling_rule"] = reduced_rule()
    fmt = formatter(f["sampling_rule"])
    calls = []
    monkeypatch.setattr(
        "bench.qualify_longform.verify_baseline_image",
        lambda **kw: {"engine_ref": "sha256:" + "e" * 64, "container_id": "baseline"},
    )
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **kw: None
    )

    def post(url, path, body, **kw):
        tokens = len(fmt.encode(body["prompt"]))
        assert tokens in (7600, 15200)
        calls.append(tokens)
        result = response()
        result["usage"]["prompt_tokens"] = tokens
        return result

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    qualified = qualify(
        fields=f,
        base_url="http://baseline",
        container="baseline",
        engine_ref="sha256:" + "e" * 64,
        output_dir=tmp_path / "qualification",
        pool_size=32,
        max_rows=64,
        repetitions=2,
        concurrency=4,
        row_fetcher=row,
        formatter=fmt,
    )
    assert Counter(calls) == {7600: 32, 15200: 32}
    assert len(qualified["eligible_row_indices"]) == 32
    assert all(i % 4 in (2, 3) for i in qualified["eligible_row_indices"])
    require_qualification(qualified, f["bench"], f["engine"])
    sampled = sample(rule=qualified, prompt_formatter=fmt)
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert Counter(r.input_length_group for r in trace.requests) == {"8k": 8, "16k": 8}
    assert trace.meta.sampling["max_baseline_prompt_drops"] == 4
    restored = materialize_round_trace(
        {"sampled_trace_sha256": sampled.sha256, "sampling_receipt": sampled.receipt},
        SimpleNamespace(bench=f["bench"], engine=f["engine"]),
        tmp_path / "worker",
        row_fetcher=row,
        prompt_formatter=fmt,
    )
    assert restored.read_bytes() == sampled.body
    for key in ("input_tiers", "max_baseline_prompt_drops"):
        changed = dict(qualified)
        del changed[key]
        with pytest.raises(SamplerError, match="baseline qualification"):
            require_qualification(changed, f["bench"], f["engine"])
        changed_trace = json.loads(sampled.body)
        if key == "input_tiers":
            del changed_trace["meta"]["sampling"][key]
            with pytest.raises(ValueError, match="tier contract"):
                validate_workload_trace_dict(changed_trace)
    tampered = copy.deepcopy(sampled.receipt)
    tampered["input_tiers"] = ["2k", "4k"]
    with pytest.raises(SamplerError):
        sample(rule=qualified, sampling_receipt=tampered, prompt_formatter=fmt)


def test_four_exclusions_are_a_union_across_baselines():
    trace = validate_workload_trace_dict(json.loads(sample(rule=reduced_rule()).body))
    ids = [r.id for r in trace.requests]
    replay = SimpleNamespace(
        completion_token_samples={rid: (8, 8) for rid in ids},
        output_samples={rid: ("Clean answer.",) * 2 for rid in ids},
        result=SimpleNamespace(role="baseline"),
    )
    for rid in ids[:4]:
        replay.completion_token_samples[rid] = (8, 5)
    dropped = baseline_prompt_drops(trace, replay, dropped={})
    assert set(dropped) == set(ids[:4])
    replay.result.role = "baseline-drift"
    replay.completion_token_samples = {rid: (8, 8) for rid in ids}
    assert baseline_prompt_drops(trace, replay, dropped=dropped) == dropped
    replay.completion_token_samples[ids[4]] = (5, 8)
    with pytest.raises(EngineError, match="limit of 4"):
        baseline_prompt_drops(trace, replay, dropped=dropped)


def test_scoring_and_eligible_tiers_must_match_explicit_selection():
    rule = reduced_rule()
    scoring = {"tier_weights": {"8k": 0.5, "16k": 0.5}}
    validate_tier_contract(rule, scoring)
    with pytest.raises(ValueError, match="must match"):
        validate_tier_contract(rule, {})
    with pytest.raises(ValueError, match="must match"):
        validate_tier_contract({}, scoring)
    trace = validate_workload_trace_dict(json.loads(sample(rule=rule).body))
    assert list(map(len, request_groups(trace.requests, 4, rule["input_tiers"]))) == [
        8,
        8,
    ]
    with pytest.raises(EngineError, match="nonempty"):
        request_groups(trace.requests, 4)  # Missing tiers are not an implicit opt-in.
    with pytest.raises(EngineError, match="nonempty"):
        request_groups(
            [r for r in trace.requests if r.input_length_group == "8k"],
            4,
            rule["input_tiers"],
        )


def test_selected_tiers_and_exclusion_limit_are_manifest_pins():
    from test_manifest import _manifest_kwargs

    from campaign.manifest import compute_manifest_hash, freeze_manifest_fields

    kwargs = _manifest_kwargs()
    kwargs["sampling_rule"] = reduced_rule()
    kwargs["scoring_rule"] = {
        "name": "weighted_tier_completion_speedup",
        "tier_weights": {"8k": 0.5, "16k": 0.5},
    }
    original = compute_manifest_hash(freeze_manifest_fields(**kwargs))
    changed = copy.deepcopy(kwargs)
    changed["sampling_rule"]["max_baseline_prompt_drops"] = 3
    assert compute_manifest_hash(freeze_manifest_fields(**changed)) != original
    changed = copy.deepcopy(kwargs)
    changed["sampling_rule"]["input_tiers"] = ["2k", "4k"]
    with pytest.raises(ValueError, match="must match"):
        freeze_manifest_fields(**changed)
    changed["scoring_rule"]["tier_weights"] = {"2k": 0.5, "4k": 0.5}
    assert compute_manifest_hash(freeze_manifest_fields(**changed)) != original


def test_sglang_two_warmups_and_three_replays_use_sixteen_prompts(tmp_path):
    from test_concurrency import requests

    from bench.mock_engine import MockEngine, MockEngineConfig
    from bench.schemas import SlaBenchConfig, SlaThresholds
    from bench.sla_bench import run_sla_engine

    selected = [r for r in requests() if r.input_length_group in ("8k", "16k")]
    with MockEngine(MockEngineConfig(model="subset", token_latency_s=0.001)) as engine:
        replay = run_sla_engine(
            engine.base_url,
            role="baseline",
            requests=selected,
            cfg=SlaBenchConfig(3, SlaThresholds(1e9, 1e9)),
            evidence_dir=tmp_path,
            request_concurrency=4,
            input_tiers=["8k", "16k"],
            engine_name="sglang",
        )
    for name in ("warmup", "warmup_2", "rep_1", "rep_2", "rep_3"):
        rows = [
            json.loads(line)
            for line in (tmp_path / "baseline" / name / "requests.jsonl")
            .read_text()
            .splitlines()
        ]
        assert Counter(
            r["input_length_group"] for r in rows if not r.get("_rep_meta")
        ) == {"8k": 8, "16k": 8}
    assert set(replay.result.tier_completion) == {"8k", "16k"}
    assert all(o["observed_peak"] <= 4 for o in replay.result.concurrency_observations)
