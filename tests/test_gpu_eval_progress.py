"""GPU eval entrypoint progress edge cases."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from validator.gpu_eval import main
from validator.state import ValidatorState

pytestmark = pytest.mark.unit


@patch("validator.gpu_eval.purge_old_logs")
@patch("validator.gpu_eval.clear_progress")
@patch("validator.gpu_eval.EvalJob.load", return_value=None)
@patch("validator.gpu_eval.ValidatorState.load", return_value=ValidatorState())
@patch("validator.gpu_eval.load_fingerprint_registry", return_value={"entries": {}})
@patch("validator.gpu_eval.validator_config.SKIP_S3", True)
@patch("validator.gpu_eval.validator_config.BASELINE_DIGEST", "sha256:abc")
@patch("cacheon_db.config.require_database_url", return_value="postgresql://test/db")
def test_no_job_exits_failure(
    mock_purge, mock_clear, mock_job, mock_state, mock_fp, _mock_db, monkeypatch
):
    monkeypatch.setenv("CACHEON_DATABASE_URL", "postgresql://test/db")
    assert main() == 2
    mock_clear.assert_called_once()
