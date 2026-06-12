"""Unit tests for Targon provider volume lifecycle."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

# CI unit job does not install validator/requirements-cpu.txt (paramiko).
if "paramiko" not in sys.modules:
    _paramiko = MagicMock()
    _paramiko.SSHClient = MagicMock
    _paramiko.AutoAddPolicy = MagicMock
    _paramiko.PKey = MagicMock
    _paramiko.AuthenticationException = Exception
    sys.modules["paramiko"] = _paramiko

from validator.providers import GpuInstance
from validator.providers.targon_provider import TargonProvider

pytestmark = pytest.mark.unit


def _instance() -> GpuInstance:
    return GpuInstance(
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
        raw={},
    )


class TestTargonProviderRent:
    @patch.object(TargonProvider, "_ensure_ssh_key", return_value="shk-test")
    @patch.object(TargonProvider, "_deploy_workload")
    @patch.object(TargonProvider, "_wait_volume_ready")
    @patch.object(TargonProvider, "_create_volume", return_value="vol-new")
    @patch.object(TargonProvider, "_post")
    def test_rent_creates_volume_and_attaches_to_workload(
        self,
        mock_post: MagicMock,
        mock_create_volume: MagicMock,
        mock_wait_ready: MagicMock,
        mock_deploy: MagicMock,
        mock_ssh_key: MagicMock,
    ) -> None:
        mock_post.return_value = {"uid": "wrk-test"}

        provider = TargonProvider("test-key")
        handle = provider.rent(_instance())

        mock_create_volume.assert_called_once()
        mock_wait_ready.assert_called_once_with("vol-new")
        mock_deploy.assert_called_once_with("wrk-test")

        workload_body = mock_post.call_args.kwargs["json"]
        assert workload_body["volumes"] == [
            {"uid": "vol-new", "mount_path": "/workspace"}
        ]
        assert handle.pod_id == "wrk-test"
        assert handle.raw["volume_uid"] == "vol-new"

    @patch.object(TargonProvider, "_abort_rent")
    @patch.object(TargonProvider, "_ensure_ssh_key", return_value="shk-test")
    @patch.object(
        TargonProvider, "_wait_volume_ready", side_effect=TimeoutError("slow")
    )
    @patch.object(TargonProvider, "_create_volume", return_value="vol-new")
    def test_rent_aborts_when_volume_not_ready(
        self,
        mock_create_volume: MagicMock,
        mock_wait_ready: MagicMock,
        mock_ssh_key: MagicMock,
        mock_abort: MagicMock,
    ) -> None:
        provider = TargonProvider("test-key")

        with pytest.raises(TimeoutError):
            provider.rent(_instance())

        mock_abort.assert_called_once_with(None, "vol-new")


class TestTargonProviderTeardown:
    @patch.object(TargonProvider, "_teardown_volume")
    @patch.object(TargonProvider, "_teardown_workload")
    def test_teardown_deletes_workload_then_volume(
        self,
        mock_workload: MagicMock,
        mock_volume: MagicMock,
    ) -> None:
        from validator.providers import PodHandle

        provider = TargonProvider("test-key")
        handle = PodHandle(
            provider="targon",
            pod_id="wrk-1",
            gpu_count=8,
            hourly_price_cents=100,
            raw={"volume_uid": "vol-1"},
        )

        provider.teardown(handle)

        mock_workload.assert_called_once_with("wrk-1")
        mock_volume.assert_called_once_with("vol-1")

    @patch("validator.providers.targon_provider.time.sleep")
    @patch.object(TargonProvider, "_post")
    def test_teardown_volume_retries_on_conflict(
        self, mock_post: MagicMock, mock_sleep: MagicMock
    ) -> None:
        provider = TargonProvider("test-key")
        response = MagicMock()
        response.status_code = 409
        mock_post.side_effect = [
            requests.HTTPError(response=response),
            None,
        ]

        provider._teardown_volume("vol-retry")

        assert mock_post.call_count == 2
        mock_post.assert_called_with("/volumes/vol-retry/delete")


class TestTargonVolumeReady:
    @patch("validator.providers.targon_provider.time.sleep")
    @patch.object(TargonProvider, "_get")
    def test_wait_accepts_registered_status(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_get.return_value = {
            "uid": "vol-1",
            "status": "registered",
            "message": "Volume registered, awaiting deploy",
        }

        provider = TargonProvider("test-key")
        provider._wait_volume_ready("vol-1")

        mock_get.assert_called_once_with("/volumes/vol-1/state")
        mock_sleep.assert_not_called()

    @patch("validator.providers.targon_provider.time.sleep")
    @patch.object(TargonProvider, "_get")
    def test_wait_accepts_ready_status(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_get.return_value = {"uid": "vol-1", "status": "READY"}

        provider = TargonProvider("test-key")
        provider._wait_volume_ready("vol-1")

        mock_get.assert_called_once_with("/volumes/vol-1/state")
        mock_sleep.assert_not_called()
