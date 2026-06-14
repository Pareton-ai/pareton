"""Unit tests for CACHEON_GPU_SSH config parsing."""

from __future__ import annotations

import pytest

from validator.config import GpuSshTarget, parse_gpu_ssh_target

pytestmark = pytest.mark.unit


class TestParseGpuSshTarget:
    def test_empty_returns_none(self):
        assert parse_gpu_ssh_target("") is None
        assert parse_gpu_ssh_target("   ") is None

    def test_user_host_default_port(self):
        assert parse_gpu_ssh_target("root@203.0.113.10") == GpuSshTarget(
            user="root", host="203.0.113.10", port=22
        )

    def test_user_host_explicit_port(self):
        assert parse_gpu_ssh_target("ubuntu@gpu.example.com:2222") == GpuSshTarget(
            user="ubuntu", host="gpu.example.com", port=2222
        )

    def test_host_only_defaults_root(self):
        assert parse_gpu_ssh_target("203.0.113.10") == GpuSshTarget(
            user="root", host="203.0.113.10", port=22
        )

    def test_host_with_port_no_user(self):
        assert parse_gpu_ssh_target("203.0.113.10:2222") == GpuSshTarget(
            user="root", host="203.0.113.10", port=2222
        )
