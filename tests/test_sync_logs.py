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
                        "Key": "state-mainnet/logs/cpu_validator_20260101_120000.log",
                    },
                    {
                        "Key": "state-mainnet/logs/cpu_validator_20260516_120000.log",
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
            ["logs/cpu_validator_20260101_120000.log"],
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
