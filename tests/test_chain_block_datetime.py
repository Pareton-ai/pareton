"""Offline checks for commitment block timestamps."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from chain import rpc

pytestmark = pytest.mark.unit


def test_fetch_block_datetime_reads_sdk_timestamp():
    expected = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    subtensor = SimpleNamespace(
        block_info=lambda block: SimpleNamespace(
            timestamp=expected.timestamp() if block == 42 else None
        )
    )

    assert rpc.fetch_block_datetime(subtensor, 42) == expected


@pytest.mark.parametrize(
    "block_info",
    [lambda _block: None, lambda _block: SimpleNamespace(timestamp=None)],
)
def test_fetch_block_datetime_fails_closed_without_timestamp(block_info):
    subtensor = SimpleNamespace(block_info=block_info)
    assert rpc.fetch_block_datetime(subtensor, 42) is None
