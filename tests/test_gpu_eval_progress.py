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
@patch("validator.gpu_eval.validator_config.SKIP_S3", True)
@patch("validator.gpu_eval.validator_config.BASELINE_DIGEST", "sha256:abc")
def test_no_work_clears_progress(mock_purge, mock_clear, mock_job, mock_state):
    assert main() == 0
    mock_clear.assert_called_once()
