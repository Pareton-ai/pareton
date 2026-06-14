"""Unit tests for validator.baseline -- datatypes and cache keys, no Docker."""

from __future__ import annotations

import pytest

from validator.baseline import derive_cache_key

pytestmark = pytest.mark.unit


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
