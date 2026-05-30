"""Unit tests for validator.baseline -- datatypes and cache keys, no Docker."""

from __future__ import annotations

import pytest

from validator.baseline import (
    BaselineCache,
    BaselinePromptResult,
    derive_cache_key,
)

pytestmark = pytest.mark.unit


def _sample_result(**overrides) -> BaselinePromptResult:
    defaults = dict(
        tokens=["Hello", " world"],
        top_logprobs=[
            [{"token": "Hello", "logprob": -0.01}],
            [{"token": " world", "logprob": -0.02}],
        ],
        ttft_s=0.045,
        throughput_tps=120.5,
        output_tokens=2,
    )
    defaults.update(overrides)
    return BaselinePromptResult(**defaults)


# --------------------------------------------------------------------------- #
# BaselinePromptResult
# --------------------------------------------------------------------------- #


class TestBaselinePromptResult:
    def test_round_trip(self):
        r = _sample_result(decode_elapsed_secs=[0.0, 0.5])
        restored = BaselinePromptResult.from_dict(r.to_dict())
        assert restored.tokens == r.tokens
        assert restored.ttft_s == r.ttft_s
        assert restored.throughput_tps == r.throughput_tps
        assert restored.output_tokens == r.output_tokens
        assert restored.decode_elapsed_secs == [0.0, 0.5]

    def test_from_dict_without_decode_elapsed_defaults_empty(self):
        r = _sample_result()
        data = {
            "tokens": r.tokens,
            "top_logprobs": r.top_logprobs,
            "ttft_s": r.ttft_s,
            "throughput_tps": r.throughput_tps,
            "output_tokens": r.output_tokens,
        }
        restored = BaselinePromptResult.from_dict(data)
        assert restored.decode_elapsed_secs == []


# --------------------------------------------------------------------------- #
# BaselineCache
# --------------------------------------------------------------------------- #


class TestBaselineCache:
    def test_round_trip(self):
        cache = BaselineCache(
            cache_key="abc123",
            results=[_sample_result(), _sample_result(ttft_s=0.05)],
        )
        restored = BaselineCache.from_dict(cache.to_dict())
        assert restored.cache_key == cache.cache_key
        assert len(restored.results) == 2
        assert restored.results[1].ttft_s == 0.05


# --------------------------------------------------------------------------- #
# derive_cache_key
# --------------------------------------------------------------------------- #


class TestDeriveCacheKey:
    def test_deterministic(self):
        k1 = derive_cache_key("0xabc123")
        k2 = derive_cache_key("0xabc123")
        assert k1 == k2

    def test_different_hashes_differ(self):
        k1 = derive_cache_key("0xabc123")
        k2 = derive_cache_key("0xdef456")
        assert k1 != k2

    def test_different_baseline_digest_differs(self):
        k1 = derive_cache_key("0xabc123", "sha256:" + "a" * 64)
        k2 = derive_cache_key("0xabc123", "sha256:" + "b" * 64)
        assert k1 != k2

    def test_same_hash_and_digest_deterministic(self):
        d = "sha256:" + "c" * 64
        assert derive_cache_key("0xabc", d) == derive_cache_key("0xabc", d)

    def test_length_is_16(self):
        assert len(derive_cache_key("anything")) == 16

    def test_hex_chars_only(self):
        key = derive_cache_key("test")
        assert all(c in "0123456789abcdef" for c in key)
