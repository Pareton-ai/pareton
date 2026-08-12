"""Unit tests for deterministic workload sampling (offline, no chain)."""

from __future__ import annotations

import pytest

from bench.sampler import (
    SamplerError,
    compute_sample_seed,
    parse_sampling_rule,
    resolve_workload_pool,
    sample_from_pool,
    sample_workload,
)

pytestmark = pytest.mark.unit


def test_resolve_pool_back_compat_single():
    pool = resolve_workload_pool(
        workload_pool=None,
        workload_trace_sha256="sha256:" + ("a" * 64),
        workload_trace_url="file:///t.json",
    )
    assert len(pool) == 1
    assert pool[0]["sha256"] == "sha256:" + ("a" * 64)


def test_sample_seed_deterministic():
    a = compute_sample_seed(
        block_hash="ab" * 32,
        patch_hash="cd" * 32,
        campaign_id="11111111-1111-1111-1111-111111111111",
    )
    b = compute_sample_seed(
        block_hash="0x" + "ab" * 32,
        patch_hash="sha256:" + "cd" * 32,
        campaign_id="11111111-1111-1111-1111-111111111111",
    )
    assert a == b
    assert len(a) == 64


def test_sample_index_stable_for_fixed_hashes():
    pool = [
        {"sha256": "sha256:" + ("1" * 64), "url": "file:///1.json"},
        {"sha256": "sha256:" + ("2" * 64), "url": "file:///2.json"},
        {"sha256": "sha256:" + ("3" * 64), "url": "file:///3.json"},
    ]
    sampled = sample_workload(
        pool=pool,
        commit_block=100,
        seed_block_offset=10,
        block_hash="ee" * 32,
        patch_hash="ff" * 32,
        campaign_id="22222222-2222-2222-2222-222222222222",
    )
    again = sample_workload(
        pool=pool,
        commit_block=100,
        seed_block_offset=10,
        block_hash="ee" * 32,
        patch_hash="ff" * 32,
        campaign_id="22222222-2222-2222-2222-222222222222",
    )
    assert sampled.index == again.index
    assert sampled.sha256 == again.sha256
    assert sampled.sample_seed_block == 110
    assert 0 <= sampled.index < 3


def test_sample_from_pool_modulo():
    pool = [{"sha256": "sha256:" + ("a" * 64), "url": "u"} for _ in range(5)]
    # seed that is clearly > len(pool)
    idx, entry = sample_from_pool(pool, seed_hex="f" * 64)
    assert idx == int("f" * 64, 16) % 5
    assert entry["url"] == "u"


def test_parse_sampling_rule_defaults_and_rejects():
    assert parse_sampling_rule(None)["seed_block_offset"] == 10
    with pytest.raises(SamplerError, match="unsupported"):
        parse_sampling_rule({"type": "weighted"})
    with pytest.raises(SamplerError, match="empty"):
        sample_from_pool([], seed_hex="aa" * 32)
