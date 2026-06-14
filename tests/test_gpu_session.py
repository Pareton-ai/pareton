"""Unit tests for long-lived GPU session orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from validator.eval_schema import ChallengerInfo, EvalJob
from validator.gpu_orchestrator import (
    _GpuSession,
    _ensure_gpu_session,
    _provision_gpu_session,
    run_gpu_eval,
    teardown_rented_gpu_session,
)
from validator.providers import GpuInstance, PodHandle


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


def _handle(provider: str = "targon") -> PodHandle:
    return PodHandle(
        provider=provider,
        pod_id="wrk-1",
        gpu_count=8,
        hourly_price_cents=1000,
        raw={"volume_uid": "vol-1"},
    )


@pytest.fixture(autouse=True)
def _clear_session():
    import validator.gpu_orchestrator as orch

    orch._session = None
    yield
    orch._session = None


class TestGpuSessionProvisioning:
    @patch("validator.gpu_orchestrator._remote_setup", return_value=True)
    @patch("validator.gpu_orchestrator.search_all_providers")
    @patch("validator.gpu_orchestrator._build_providers")
    def test_provision_stores_session(
        self, mock_build, mock_search, mock_setup, tmp_path
    ):
        provider = MagicMock()
        provider.name = "targon"
        provider.READY_TIMEOUT_S = 600
        handle = _handle()
        provider.rent.return_value = handle
        provider.wait_ready.return_value = handle
        mock_build.return_value = [provider]
        mock_search.return_value = GpuInstance(
            provider="targon",
            instance_id="h200-small",
            description="H200",
            hourly_price_cents=500,
            num_gpus=8,
            gpu_type="H200",
            vram_per_gpu_gb=141,
            total_vram_gb=1128,
            storage_gb=0,
            memory_gb=0,
            vcpus=0,
            docker_in_docker=True,
        )

        result = _provision_gpu_session(str(tmp_path))

        assert result is not None
        provider.rent.assert_called_once()
        provider.wait_ready.assert_called_once()
        mock_setup.assert_called_once()

        import validator.gpu_orchestrator as orch

        assert orch._session is not None
        assert orch._session.setup_complete is True

    @patch("validator.gpu_orchestrator._abort_provision")
    @patch("validator.gpu_orchestrator._remote_setup", return_value=False)
    @patch("validator.gpu_orchestrator.search_all_providers")
    @patch("validator.gpu_orchestrator._build_providers")
    def test_setup_failure_aborts(
        self, mock_build, mock_search, mock_setup, mock_abort, tmp_path
    ):
        provider = MagicMock()
        provider.name = "targon"
        provider.READY_TIMEOUT_S = 600
        handle = _handle()
        provider.rent.return_value = handle
        provider.wait_ready.return_value = handle
        mock_build.return_value = [provider]
        mock_search.return_value = GpuInstance(
            provider="targon",
            instance_id="h200-small",
            description="H200",
            hourly_price_cents=500,
            num_gpus=8,
            gpu_type="H200",
            vram_per_gpu_gb=141,
            total_vram_gb=1128,
            storage_gb=0,
            memory_gb=0,
            vcpus=0,
            docker_in_docker=True,
        )

        assert _provision_gpu_session(str(tmp_path)) is None
        mock_abort.assert_called_once_with(provider, handle)


class TestGpuSessionReuse:
    @patch("validator.gpu_orchestrator._provision_gpu_session")
    def test_reuses_existing_session(self, mock_provision, tmp_path):
        provider = MagicMock()
        handle = _handle()
        import validator.gpu_orchestrator as orch

        orch._session = _GpuSession(
            provider=provider, handle=handle, setup_complete=True
        )

        got = _ensure_gpu_session(str(tmp_path))

        assert got == (provider, handle)
        mock_provision.assert_not_called()


class TestRunGpuEvalCloudPath:
    @patch("validator.gpu_orchestrator._start_remote_eval", return_value=True)
    @patch("validator.gpu_orchestrator._ensure_gpu_session")
    @patch("validator.gpu_orchestrator.validator_config.GPU_SSH_ENABLED", False)
    def test_does_not_teardown_after_eval(self, mock_session, mock_eval):
        provider = MagicMock()
        handle = _handle()
        mock_session.return_value = (provider, handle)

        assert run_gpu_eval("/tmp/state", _eval_job()) is True

        provider.teardown.assert_not_called()
        mock_eval.assert_called_once()


class TestShutdownTeardown:
    def test_teardown_rented_session(self):
        provider = MagicMock()
        handle = _handle()
        import validator.gpu_orchestrator as orch

        orch._session = _GpuSession(
            provider=provider, handle=handle, setup_complete=True
        )

        teardown_rented_gpu_session()

        provider.teardown.assert_called_once_with(handle)
        assert orch._session is None

    def test_teardown_noop_when_no_session(self):
        teardown_rented_gpu_session()
