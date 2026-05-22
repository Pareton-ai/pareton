"""Unit tests for submission payment gating."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from validator.chain import (
    CommitmentRecord,
    PaymentVerifyOutcome,
    PaymentVerifyResult,
)
from validator.cpu_validator import _gate_unpaid_commitments
from validator.state import ValidatorState, _eval_key

pytestmark = pytest.mark.unit

_TX = "0x" + "a" * 64
_BLK = "0x" + "b" * 64
_DIGEST = "sha256:" + "c" * 64


def _commit(uid: int, hotkey: str, block: int) -> CommitmentRecord:
    return CommitmentRecord(
        uid=uid,
        hotkey=hotkey,
        coldkey="5Cold",
        commit_block=block,
        image="user/repo:v1",
        digest=_DIGEST,
        raw="{}",
        payment_tx=_TX,
        payment_block=_BLK,
    )


def test_gate_skips_rpc_for_evaluated_commitment(monkeypatch):
    state = ValidatorState()
    com = _commit(1, "hk1", 100)
    state.evaluations[_eval_key("hk1", 100)] = MagicMock()

    called = False

    def _verify(*_args, **_kwargs):
        nonlocal called
        called = True
        return PaymentVerifyResult(outcome=PaymentVerifyOutcome.OK)

    monkeypatch.setattr("validator.cpu_validator.verify_submission_payment", _verify)
    monkeypatch.setattr(
        "validator.cpu_validator.validator_config.SUBMISSION_FEE_RAO", 1
    )

    valid, rejected = _gate_unpaid_commitments(MagicMock(), state, {1: com})
    assert 1 in valid
    assert rejected == []
    assert called is False


def test_gate_skips_rpc_for_precheck_failure(monkeypatch):
    state = ValidatorState()
    com = _commit(1, "hk1", 100)
    state.record_precheck_failure("hk1", 100, "payment failed")

    called = False

    def _verify(*_args, **_kwargs):
        nonlocal called
        called = True
        return PaymentVerifyResult(outcome=PaymentVerifyOutcome.OK)

    monkeypatch.setattr("validator.cpu_validator.verify_submission_payment", _verify)
    monkeypatch.setattr(
        "validator.cpu_validator.validator_config.SUBMISSION_FEE_RAO", 1
    )

    valid, rejected = _gate_unpaid_commitments(MagicMock(), state, {1: com})
    assert valid == {}
    assert rejected == []
    assert called is False


def test_gate_defers_on_rpc_error(monkeypatch):
    state = ValidatorState()
    com = _commit(1, "hk1", 100)

    monkeypatch.setattr(
        "validator.cpu_validator.verify_submission_payment",
        lambda *_a, **_k: PaymentVerifyResult(
            outcome=PaymentVerifyOutcome.DEFER,
            reason="connection reset",
        ),
    )
    monkeypatch.setattr(
        "validator.cpu_validator.validator_config.SUBMISSION_FEE_RAO", 1
    )

    valid, rejected = _gate_unpaid_commitments(MagicMock(), state, {1: com})
    assert valid == {}
    assert rejected == []


def test_gate_rejects_definitive_failure(monkeypatch):
    state = ValidatorState()
    com = _commit(1, "hk1", 100)

    monkeypatch.setattr(
        "validator.cpu_validator.verify_submission_payment",
        lambda *_a, **_k: PaymentVerifyResult(
            outcome=PaymentVerifyOutcome.REJECT,
            reason="missing transfer event",
        ),
    )
    monkeypatch.setattr(
        "validator.cpu_validator.validator_config.SUBMISSION_FEE_RAO", 1
    )

    valid, rejected = _gate_unpaid_commitments(MagicMock(), state, {1: com})
    assert valid == {}
    assert len(rejected) == 1
    assert "missing transfer event" in rejected[0][1]
