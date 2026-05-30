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
    def test_targon_includes_vllm_cache_dir_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CACHEON_VLLM_CACHE_DIR", None)
            exports = _build_env_exports(_handle("targon"))
        assert 'export CACHEON_VLLM_CACHE_DIR="/workspace/vllm-cache"' in exports

    def test_targon_respects_explicit_cache_dir(self):
        with patch.dict(
            os.environ, {"CACHEON_VLLM_CACHE_DIR": "/custom/vllm-cache"}, clear=False
        ):
            exports = _build_env_exports(_handle("targon"))
        assert 'export CACHEON_VLLM_CACHE_DIR="/custom/vllm-cache"' in exports

    def test_targon_empty_cache_dir_disables_mount(self):
        with patch.dict(os.environ, {"CACHEON_VLLM_CACHE_DIR": ""}, clear=False):
            exports = _build_env_exports(_handle("targon"))
        assert 'export CACHEON_VLLM_CACHE_DIR=""' in exports

    @pytest.mark.parametrize("provider", ["lium", "shadeform"])
    def test_non_targon_omits_vllm_cache_dir(self, provider: str):
        exports = _build_env_exports(_handle(provider))
        assert "CACHEON_VLLM_CACHE_DIR" not in exports
