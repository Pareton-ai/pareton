"""Unit tests for miner/commit.py helpers."""

from __future__ import annotations

import pytest

from miner.commit import _commit_retry_command, _parse_payment_reuse

pytestmark = pytest.mark.unit

_TX = "0x" + "a" * 64
_BLK = "0x" + "b" * 64


class TestParsePaymentReuse:
    def test_both_none(self):
        assert _parse_payment_reuse(None, None) is None

    def test_valid_pair(self):
        assert _parse_payment_reuse(_TX, _BLK) == (_TX, _BLK)

    def test_tx_without_block(self):
        with pytest.raises(ValueError, match="together"):
            _parse_payment_reuse(_TX, None)

    def test_invalid_hex(self):
        with pytest.raises(ValueError, match="invalid"):
            _parse_payment_reuse("0x" + "g" * 64, _BLK)


class TestCommitRetryCommand:
    def test_includes_payment_flags(self):
        class Args:
            image = "user/repo:v1"
            digest = "sha256:" + "c" * 64
            wallet_name = "default"
            wallet_hotkey = "default"
            network = "finney"
            netuid = 14

        cmd = _commit_retry_command(Args(), _TX, _BLK)
        assert "--payment-tx" in cmd
        assert "--payment-block" in cmd
        assert _TX in cmd
