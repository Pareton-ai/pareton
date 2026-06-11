"""Unit tests for GPU orchestrator env wiring."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from validator.gpu_orchestrator import _build_env_exports
from validator.providers import PodHandle


def _handle(provider: str) -> PodHandle:
    return PodHandle(
        provider=provider,
        pod_id="pod-1",
        gpu_count=8,
        hourly_price_cents=1000,
    )


class TestBuildEnvExports:
    def test_exports_vllm_cache_dir_when_set(self):
        with patch.dict(
            os.environ, {"CACHEON_VLLM_CACHE_DIR": "/custom/vllm-cache"}, clear=False
        ):
            exports = _build_env_exports(_handle("targon"))
        assert 'export CACHEON_VLLM_CACHE_DIR="/custom/vllm-cache"' in exports

    def test_exports_empty_vllm_cache_dir_when_explicitly_disabled(self):
        with patch.dict(os.environ, {"CACHEON_VLLM_CACHE_DIR": ""}, clear=False):
            exports = _build_env_exports(_handle("lium"))
        assert 'export CACHEON_VLLM_CACHE_DIR=""' in exports

    @pytest.mark.parametrize("provider", ["targon", "lium", "shadeform"])
    def test_omits_vllm_cache_dir_when_unset(self, provider: str):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CACHEON_VLLM_CACHE_DIR", None)
            exports = _build_env_exports(_handle(provider))
        assert "CACHEON_VLLM_CACHE_DIR" not in exports
