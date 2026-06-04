"""Unit tests for CACHEON_PREFERRED_GPU config and search selection."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from validator.config import _parse_preferred_gpu
from validator.providers import GpuInstance, search_all_providers

pytestmark = pytest.mark.unit


def _instance(
    gpu_type: str,
    *,
    hourly_price_cents: int = 1000,
    provider: str = "stub",
) -> GpuInstance:
    vram = {"H200": 141, "H100": 80, "B200": 180, "B300": 288}[gpu_type]
    return GpuInstance(
        provider=provider,
        instance_id=f"{provider}-{gpu_type}-{hourly_price_cents}",
        description="",
        hourly_price_cents=hourly_price_cents,
        num_gpus=8,
        gpu_type=gpu_type,
        vram_per_gpu_gb=vram,
        total_vram_gb=8 * vram,
        storage_gb=500,
        memory_gb=0,
        vcpus=0,
        docker_in_docker=True,
    )


class _StubProvider:
    name = "stub"

    def __init__(self, instances: list[GpuInstance]) -> None:
        self._instances = instances

    def search(self) -> list[GpuInstance]:
        return self._instances


class TestParsePreferredGpu:
    def test_accepts_canonical_and_case_insensitive(self):
        with patch.dict(os.environ, {"CACHEON_PREFERRED_GPU": "h200"}, clear=False):
            assert _parse_preferred_gpu() == "H200"

    def test_empty_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CACHEON_PREFERRED_GPU", None)
            assert _parse_preferred_gpu() == ""

    def test_invalid_value_treated_as_empty(self):
        with patch.dict(os.environ, {"CACHEON_PREFERRED_GPU": "A100"}, clear=False):
            assert _parse_preferred_gpu() == ""


class TestSearchAllProvidersPreferredGpu:
    def test_pin_returns_only_matching_gpu(self):
        prov = _StubProvider(
            [
                _instance("H200", hourly_price_cents=500),
                _instance("H100", hourly_price_cents=100),
            ]
        )
        best = search_all_providers([prov], preferred_gpu="H200")
        assert best is not None
        assert best.gpu_type == "H200"

    def test_pin_miss_returns_none(self):
        prov = _StubProvider([_instance("H100", hourly_price_cents=100)])
        assert search_all_providers([prov], preferred_gpu="H200") is None

    def test_pin_picks_cheapest_among_providers(self):
        p1 = _StubProvider([_instance("B200", hourly_price_cents=800)])
        p2 = _StubProvider(
            [_instance("B200", hourly_price_cents=400, provider="other")]
        )
        p2.name = "other"
        best = search_all_providers([p1, p2], preferred_gpu="B200")
        assert best is not None
        assert best.hourly_price_cents == 400

    def test_empty_uses_tier_a_over_tier_b(self):
        prov = _StubProvider(
            [
                _instance("H100", hourly_price_cents=50),
                _instance("H200", hourly_price_cents=500),
            ]
        )
        best = search_all_providers([prov], preferred_gpu="")
        assert best is not None
        assert best.gpu_type == "H200"

    def test_empty_falls_back_to_h100_when_no_tier_a(self):
        prov = _StubProvider([_instance("H100", hourly_price_cents=200)])
        best = search_all_providers([prov], preferred_gpu="")
        assert best is not None
        assert best.gpu_type == "H100"
