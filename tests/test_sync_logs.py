"""Unit tests for S3 log purge in validator.sync."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestPurgeRemoteLogs:
    def test_deletes_only_stale_keys(self):
        cutoff = datetime(2026, 5, 10, 12, 0, 0)
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "state-mainnet/logs/cpu_20260101_120000.log",
                    },
                    {
                        "Key": "state-mainnet/logs/cpu_8303961_20260516_120000.log",
                    },
                    {"Key": "state-mainnet/logs/random_file.txt"},
                ]
            }
        ]
        mock_s3 = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        with (
            patch("validator.sync._client", return_value=mock_s3),
            patch("validator.sync.delete_remote_keys") as mock_delete,
            patch("validator.sync.S3_PREFIX", "state-mainnet"),
            patch("validator.sync.BUCKET", "cacheon-validator"),
        ):
            from validator.sync import purge_remote_logs

            removed = purge_remote_logs(cutoff)

        assert removed == 1
        mock_delete.assert_called_once_with(
            ["logs/cpu_20260101_120000.log"],
            bucket="cacheon-validator",
            prefix="state-mainnet",
        )

    def test_noop_when_empty(self):
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_s3 = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        with (
            patch("validator.sync._client", return_value=mock_s3),
            patch("validator.sync.delete_remote_keys") as mock_delete,
        ):
            from validator.sync import purge_remote_logs

            removed = purge_remote_logs(datetime(2026, 5, 20))

        assert removed == 0
        mock_delete.assert_not_called()


class TestDownloadSkipPrefixes:
    def test_skips_logs_prefix_by_default(self, tmp_path):
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "state-mainnet/state.json"},
                    {"Key": "state-mainnet/logs/cpu_100_20260101_120000.log"},
                    {"Key": "state-mainnet/container_logs/uid1_abcd_100.log"},
                ]
            }
        ]
        mock_s3 = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        with (
            patch("validator.sync._client", return_value=mock_s3),
            patch("validator.sync.S3_PREFIX", "state-mainnet"),
            patch("validator.sync.BUCKET", "cacheon-validator"),
        ):
            from validator.sync import download

            count = download(tmp_path, skip_prefixes=("logs/",))

        assert count == 2
        assert mock_s3.download_file.call_count == 2
        downloaded_locals = {
            call.args[2] for call in mock_s3.download_file.call_args_list
        }
        assert str(tmp_path / "state.json") in downloaded_locals
        assert str(tmp_path / "container_logs/uid1_abcd_100.log") in downloaded_locals
        assert not (tmp_path / "logs/cpu_100_20260101_120000.log").exists()

    def test_skip_cpu_logs_only_when_pulling_gpu_logs(self, tmp_path):
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "state-mainnet/logs/cpu_100_20260101_120000.log"},
                    {"Key": "state-mainnet/logs/gpu_100_20260101_130000.log"},
                ]
            }
        ]
        mock_s3 = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        with (
            patch("validator.sync._client", return_value=mock_s3),
            patch("validator.sync.S3_PREFIX", "state-mainnet"),
        ):
            from validator.sync import download

            count = download(
                tmp_path,
                only=["logs/"],
                skip_prefixes=("logs/cpu_",),
            )

        assert count == 1
        mock_s3.download_file.assert_called_once()
        assert mock_s3.download_file.call_args.args[2].endswith(
            "logs/gpu_100_20260101_130000.log"
        )
