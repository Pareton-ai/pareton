"""Unit tests for deterministic on-the-fly workload sampling (offline, no network)."""

from __future__ import annotations

import hashlib

import pytest

from bench.sampler import (
    MAX_PROMPT_CHARS,
    SamplerError,
    calib_seed,
    compute_sample_seed,
    extract_prompt,
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
        "seed_block_offset": 10,
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
    a = compute_sample_seed(
        block_hash="ab" * 32,
        patch_hash="cd" * 32,
        campaign_id=CAMPAIGN,
    )
    b = compute_sample_seed(
        block_hash="0x" + "ab" * 32,
        patch_hash="sha256:" + "cd" * 32,
        campaign_id=CAMPAIGN,
    )
    assert a == b
    assert len(a) == 64


def test_parse_sampling_rule_requires_hf_rows():
    parsed = parse_sampling_rule(_rule())
    assert parsed["type"] == "hf_rows"
    assert parsed["n_prompts"] == 3
    with pytest.raises(SamplerError, match="unsupported"):
        parse_sampling_rule({"type": "uniform_index"})
    with pytest.raises(SamplerError, match="must be an object"):
        parse_sampling_rule(None)


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
    sampled = generate_trace(
        rule=rule, seed_hex="cc" * 32, row_fetcher=_fetcher(rows)
    )
    assert 0 not in sampled.row_indices
    assert 2 not in sampled.row_indices
    again = generate_trace(
        rule=rule, seed_hex="cc" * 32, row_fetcher=_fetcher(rows)
    )
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
        patch_hash="ff" * 32,
        campaign_id=CAMPAIGN,
        row_fetcher=_fetcher(rows),
    )
    assert sampled.sample_seed_block == 110
    assert sampled.receipt["type"] == "hf_rows"
    assert sampled.receipt["row_indices"] == list(sampled.row_indices)


def test_calib_seed_stable():
    assert calib_seed(CAMPAIGN, 0) == calib_seed(CAMPAIGN, 0)
    assert calib_seed(CAMPAIGN, 0) != calib_seed(CAMPAIGN, 1)


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
