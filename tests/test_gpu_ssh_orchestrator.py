"""Unit tests for CACHEON_GPU_SSH orchestration path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from validator.config import GpuSshTarget
from validator.eval_schema import ChallengerInfo, EvalJob
from validator.gpu_orchestrator import run_gpu_eval
from validator.providers import PodHandle

pytestmark = pytest.mark.unit


def _eval_job() -> EvalJob:
    return EvalJob(
        block=100,
        block_hash="0xabc",
        challengers=[
            ChallengerInfo(
                uid=1,
                hotkey="hk1",
                commit_block=99,
                image="img:tag",
                digest="sha256:abc",
            )
        ],
        created_at=0,
        leader=None,
        runner_up=None,
    )


class TestRunGpuEvalSshPath:
    @patch("validator.gpu_orchestrator._start_remote_eval", return_value=True)
    @patch("validator.providers.static_ssh_provider.StaticSshProvider")
    @patch("validator.gpu_orchestrator.validator_config.GPU_SSH_ENABLED", True)
    @patch(
        "validator.gpu_orchestrator.validator_config.GPU_SSH",
        GpuSshTarget(user="root", host="10.0.0.1", port=22),
    )
    @patch("validator.gpu_orchestrator.validator_config.GPU_POD_PROFILE", "targon")
    def test_skips_cloud_search_and_setup(self, mock_provider_cls, mock_eval):
        provider = MagicMock()
        mock_provider_cls.return_value = provider
        handle = PodHandle(
            provider="targon",
            pod_id="root@10.0.0.1:22",
            gpu_count=8,
            hourly_price_cents=0,
        )
        provider.connect.return_value = handle
        provider.wait_ready.return_value = handle

        assert run_gpu_eval("/tmp/state", _eval_job()) is True

        provider.connect.assert_called_once()
        provider.wait_ready.assert_called_once()
        provider.teardown.assert_not_called()
        mock_eval.assert_called_once()

    @patch("validator.gpu_orchestrator._build_providers")
    @patch("validator.gpu_orchestrator.validator_config.GPU_SSH_ENABLED", False)
    def test_provider_rent_uses_cloud_providers(self, mock_build):
        mock_build.return_value = []
        assert run_gpu_eval("/tmp/state", _eval_job()) is False
        mock_build.assert_called_once()
