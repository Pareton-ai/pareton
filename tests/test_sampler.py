"""Unit tests for deterministic on-the-fly workload sampling (offline, no network)."""

from __future__ import annotations

import hashlib
import json

import pytest

from bench.sampler import (
    ALGO_VERSION,
    MAX_PROMPT_CHARS,
    PromptFormatter,
    SamplerError,
    build_prompt_formatter,
    compute_sample_seed,
    extract_prompt,
    fetch_hf_row,
    generate_trace,
    parse_sampling_rule,
    sample_workload,
    select_row_indices,
)

pytestmark = pytest.mark.unit

CAMPAIGN = "11111111-1111-1111-1111-111111111111"


def _user_row(text: str) -> dict:
    return {"trajectory": [{"role": "user", "text": text}]}


def _rule(**overrides) -> dict:
    base = {
        "type": "hf_rows",
        "seed_block_offset": 1,
        "dataset": "nebius/SWE-agent-trajectories",
        "revision": "deadbeef" * 5,
        "config": "default",
        "split": "train",
        "n_rows": 8,
        "n_prompts": 3,
        "max_tokens": 128,
        "algo_version": 1,
    }
    base.update(overrides)
    return base


def _fetcher(rows: list[dict]):
    def fetch(idx: int) -> dict:
        return rows[idx]

    return fetch


def test_sample_seed_deterministic():
    a = compute_sample_seed(block_hash="ab" * 32, campaign_id=CAMPAIGN)
    b = compute_sample_seed(block_hash="0x" + "ab" * 32, campaign_id=CAMPAIGN)
    assert a == b
    assert len(a) == 64


def test_sample_seed_is_shared_across_a_round():
    """Every image in one round draws the same prompt set, so the patch hash
    is not in the seed material."""
    seed = compute_sample_seed(block_hash="ab" * 32, campaign_id=CAMPAIGN)
    other = compute_sample_seed(block_hash="ab" * 32, campaign_id="other-campaign")
    assert seed != other


def test_parse_sampling_rule_requires_hf_rows():
    parsed = parse_sampling_rule(_rule())
    assert parsed["type"] == "hf_rows"
    assert parsed["n_prompts"] == 3
    assert parsed["algo_version"] == ALGO_VERSION == 1
    assert "prompt_format" not in parsed
    assert "ignore_eos" not in parsed
    assert parse_sampling_rule(_rule(ignore_eos=True))["ignore_eos"] is True
    omitted = _rule()
    del omitted["seed_block_offset"]
    assert parse_sampling_rule(omitted)["seed_block_offset"] == 1
    with pytest.raises(SamplerError, match="unsupported"):
        parse_sampling_rule({"type": "uniform_index"})
    with pytest.raises(SamplerError, match="must be an object"):
        parse_sampling_rule(None)
    with pytest.raises(SamplerError, match="ignore_eos must be a boolean"):
        parse_sampling_rule(_rule(ignore_eos="true"))


def test_ignore_eos_is_pinned_only_when_enabled():
    rows = [_user_row(f"prompt-{i}") for i in range(8)]
    ordinary = generate_trace(
        rule=_rule(), seed_hex="aa" * 32, row_fetcher=_fetcher(rows)
    )
    forced = generate_trace(
        rule=_rule(ignore_eos=True),
        seed_hex="aa" * 32,
        row_fetcher=_fetcher(rows),
    )
    ordinary_doc = json.loads(ordinary.body)
    forced_doc = json.loads(forced.body)
    assert "ignore_eos" not in ordinary.receipt
    assert all(
        "ignore_eos" not in request["sampling"] for request in ordinary_doc["requests"]
    )
    assert forced.receipt["ignore_eos"] is True
    assert all(
        request["sampling"]["ignore_eos"] is True for request in forced_doc["requests"]
    )
    assert forced.sha256 != ordinary.sha256


def test_sampler_rejects_an_unreleased_algorithm_version():
    with pytest.raises(SamplerError, match="unsupported algo_version"):
        parse_sampling_rule(_rule(algo_version=2))


def _chat_rule() -> dict:
    return _rule()


def _fake_tokenizer_config() -> dict:
    return {
        "chat_template": (
            "<user>{{ messages[0]['content'] }}</user>"
            "{% if add_generation_prompt %}<assistant>{% endif %}"
            "{% if enable_thinking is undefined or enable_thinking is true %}"
            "<think>{% else %}<no-think>{% endif %}"
        ),
        "eos_token": "<eos>",
    }


def _chat_formatter() -> PromptFormatter:
    return build_prompt_formatter(
        _chat_rule(),
        model_repo="org/model",
        model_revision="a" * 40,
        config_loader=lambda **_kwargs: _fake_tokenizer_config(),
    )


def test_hf_chat_formatter_renders_and_records_the_template_pin():
    formatter = _chat_formatter()
    assert formatter.render("issue text") == (
        "<user>issue text</user><assistant><no-think>"
    )
    assert formatter.receipt["chat_template"]["model_repo"] == "org/model"
    assert formatter.receipt["chat_template"]["model_revision"] == "a" * 40
    assert formatter.receipt["chat_template"]["sha256"].startswith("sha256:")
    assert formatter.receipt["chat_template"]["enable_thinking"] is False


def test_hf_chat_formatter_rejects_a_changed_template():
    with pytest.raises(SamplerError, match="chat template sha256 mismatch"):
        build_prompt_formatter(
            _chat_rule(),
            model_repo="org/model",
            model_revision="a" * 40,
            expected_template_sha256="sha256:" + "0" * 64,
            config_loader=lambda **_kwargs: _fake_tokenizer_config(),
        )


def test_chat_formatted_trace_hashes_the_rendered_prompt():
    rows = [_user_row(f"prompt-{i}") for i in range(8)]
    sampled = generate_trace(
        rule=_chat_rule(),
        seed_hex="aa" * 32,
        row_fetcher=_fetcher(rows),
        prompt_formatter=_chat_formatter(),
    )
    trace = json.loads(sampled.body)
    assert trace["requests"][0]["prompt"].startswith("<user>")
    assert trace["requests"][0]["prompt"].endswith("</user><assistant><no-think>")
    assert "chat_template" in sampled.receipt


def test_trace_without_a_formatter_keeps_raw_prompt_compatibility():
    rows = [_user_row(f"prompt-{i}") for i in range(8)]
    sampled = generate_trace(
        rule=_chat_rule(),
        seed_hex="aa" * 32,
        row_fetcher=_fetcher(rows),
    )
    trace = json.loads(sampled.body)
    assert trace["requests"][0]["prompt"].startswith("prompt-")
    assert "chat_template" not in sampled.receipt


def test_fixed_seed_identical_trace_sha256_twice():
    rows = [_user_row(f"prompt-{i}") for i in range(8)]
    rule = _rule()
    a = generate_trace(rule=rule, seed_hex="aa" * 32, row_fetcher=_fetcher(rows))
    b = generate_trace(rule=rule, seed_hex="aa" * 32, row_fetcher=_fetcher(rows))
    assert a.sha256 == b.sha256
    assert a.row_indices == b.row_indices
    assert a.body == b.body
    assert hashlib.sha256(a.body).hexdigest() == a.sha256.split(":", 1)[1]


def test_two_seeds_different_row_sets():
    rows = [_user_row(f"prompt-{i}") for i in range(8)]
    rule = _rule()
    a = generate_trace(rule=rule, seed_hex="aa" * 32, row_fetcher=_fetcher(rows))
    b = generate_trace(rule=rule, seed_hex="bb" * 32, row_fetcher=_fetcher(rows))
    assert a.row_indices != b.row_indices
    assert a.sha256 != b.sha256


def test_empty_and_too_long_rows_skipped_deterministically():
    rows = [
        _user_row(""),
        _user_row("ok-1"),
        _user_row("x" * (MAX_PROMPT_CHARS + 1)),
        _user_row("ok-2"),
        _user_row("ok-3"),
        _user_row("ok-4"),
        _user_row("ok-5"),
        _user_row("ok-6"),
    ]
    rule = _rule(n_prompts=3)
    sampled = generate_trace(rule=rule, seed_hex="cc" * 32, row_fetcher=_fetcher(rows))
    assert 0 not in sampled.row_indices
    assert 2 not in sampled.row_indices
    again = generate_trace(rule=rule, seed_hex="cc" * 32, row_fetcher=_fetcher(rows))
    assert sampled.row_indices == again.row_indices
    assert sampled.sha256 == again.sha256


def test_extract_prompt_first_user_message():
    row = {
        "trajectory": [
            {"role": "system", "text": "sys"},
            {"role": "user", "text": "issue text"},
            {"role": "user", "text": "later"},
        ]
    }
    assert extract_prompt(row) == "issue text"


def test_sample_workload_uses_future_block():
    rows = [_user_row(f"prompt-{i}") for i in range(8)]
    sampled = sample_workload(
        rule=_rule(),
        commit_block=100,
        block_hash="ee" * 32,
        campaign_id=CAMPAIGN,
        row_fetcher=_fetcher(rows),
    )
    assert sampled.sample_seed_block == 101
    assert sampled.receipt["type"] == "hf_rows"
    assert sampled.receipt["row_indices"] == list(sampled.row_indices)


def test_select_row_indices_modulo_distinct():
    idxs = select_row_indices(
        seed_hex="f" * 64,
        n_rows=5,
        n_prompts=3,
        row_ok=lambda _i: True,
    )
    assert len(idxs) == 3
    assert len(set(idxs)) == 3
    assert all(0 <= i < 5 for i in idxs)


def test_fetch_hf_row_uses_load_dataset_and_caches(monkeypatch):
    from bench.sampler import _HF_SPLIT_CACHE

    _HF_SPLIT_CACHE.clear()
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("PARETON_HF_TOKEN", raising=False)
    calls: list[dict] = []
    split = [
        {"trajectory": [{"role": "user", "text": "a"}]},
        {"trajectory": [{"role": "user", "text": "b"}]},
        {"trajectory": [{"role": "user", "text": "c"}]},
    ]

    def fake_load(**kwargs):
        calls.append(kwargs)
        return split

    monkeypatch.setattr("bench.sampler._call_load_dataset", fake_load)
    rule = _rule()
    assert fetch_hf_row(rule, 1)["trajectory"][0]["text"] == "b"
    assert fetch_hf_row(rule, 2)["trajectory"][0]["text"] == "c"
    assert len(calls) == 1
    assert calls[0]["path"] == rule["dataset"]
    assert calls[0]["revision"] == rule["revision"]
    assert calls[0]["name"] == "default"
    assert calls[0]["split"] == "train"
    assert calls[0]["token"] == "hf_test_token"


def test_fetch_hf_row_index_error(monkeypatch):
    from bench.sampler import _HF_SPLIT_CACHE

    _HF_SPLIT_CACHE.clear()
    monkeypatch.setattr(
        "bench.sampler._call_load_dataset",
        lambda **_k: [{"trajectory": []}],
    )
    with pytest.raises(SamplerError, match="missing at offset"):
        fetch_hf_row(_rule(), 5)
