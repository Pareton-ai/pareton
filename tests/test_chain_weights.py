"""Unit tests for validator-side weight setting (no chain/network).

Every subtensor here is a fake. Nothing in this file may reach finney.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

import chain.weights as weights
from chain.weights import (
    ValidatorPermitError,
    WeightSetError,
    assert_validator_permit,
    set_weights,
)

# Stand-in for wallet material: no log line or error message may contain it.
SECRET_HOTKEY = "5SigningHotkeyThatMustNeverBeLogged"


class FakeMeta:
    def __init__(self, *, registered: bool = True, permit: bool = True, uid: int = 7):
        self._neuron = (
            SimpleNamespace(uid=uid, validator_permit=permit) if registered else None
        )

    def by_hotkey(self, hotkey: str):
        return self._neuron


def result(ok: bool, message: str = ""):
    """The shape `Subtensor.execute` actually returns on the pinned SDK.

    Scripting bare bools here would test a shape bittensor 11.x never
    produces, which is how a retry path can look covered and not be.
    """
    return SimpleNamespace(success=ok, message=message)


class FakeSubtensor:
    """Records every execute call and replays a scripted list of returns."""

    def __init__(self, *returns):
        self._returns = list(returns)
        self.calls: list[object] = []

    def execute(self, intent, wallet, **kwargs):
        self.calls.append((intent, kwargs))
        result = self._returns[min(len(self.calls) - 1, len(self._returns) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def wallet():
    return SimpleNamespace(hotkey=SimpleNamespace(ss58_address=SECRET_HOTKEY))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(weights.time, "sleep", lambda _s: None)


def _set(subtensor, wallet, meta, **kwargs):
    return set_weights(
        subtensor,
        wallet,
        meta,
        netuid=10,
        uids=[7, 201],
        weights=[0.1, 0.9],
        **kwargs,
    )


def test_accepted_result_submits_once(wallet):
    sub = FakeSubtensor(result(True))
    _set(sub, wallet, FakeMeta())
    assert len(sub.calls) == 1


def test_a_result_without_success_is_a_failure(wallet):
    """An SDK that changes this shape must retry and raise, never silently win."""
    sub = FakeSubtensor(SimpleNamespace())
    with pytest.raises(WeightSetError):
        _set(sub, wallet, FakeMeta(), attempts=2)
    assert len(sub.calls) == 2


def test_rejection_retries_then_raises_with_last_reason(wallet):
    sub = FakeSubtensor(
        result(False, "first"), result(False, "second"), result(False, "last")
    )
    with pytest.raises(WeightSetError) as excinfo:
        _set(sub, wallet, FakeMeta(), attempts=3)
    assert len(sub.calls) == 3
    assert "last" in str(excinfo.value)
    assert "first" not in str(excinfo.value)


def test_exception_mid_attempt_is_retried_not_propagated(wallet):
    sub = FakeSubtensor(ConnectionError("substrate closed"), result(True))
    _set(sub, wallet, FakeMeta(), attempts=3)
    assert len(sub.calls) == 2


def test_exception_on_every_attempt_raises_typed_error(wallet):
    sub = FakeSubtensor(ConnectionError("substrate closed"))
    with pytest.raises(WeightSetError, match="substrate closed"):
        _set(sub, wallet, FakeMeta(), attempts=2)


def test_rejection_reason_reaches_the_error(wallet):
    sub = FakeSubtensor(result(False, "rate limit"))
    with pytest.raises(WeightSetError, match="rate limit"):
        _set(sub, wallet, FakeMeta(), attempts=1)


def test_missing_permit_raises_before_any_submit(wallet):
    sub = FakeSubtensor(result(True))
    with pytest.raises(ValidatorPermitError):
        _set(sub, wallet, FakeMeta(permit=False))
    assert sub.calls == []


def test_unregistered_hotkey_raises_before_any_submit(wallet):
    sub = FakeSubtensor(result(True))
    with pytest.raises(ValidatorPermitError):
        _set(sub, wallet, FakeMeta(registered=False))
    assert sub.calls == []


def test_permit_error_is_catchable_as_weight_set_error():
    """The caller logs one type and keeps its loop running."""
    assert issubclass(ValidatorPermitError, WeightSetError)


def test_assert_validator_permit_returns_uid():
    assert assert_validator_permit(FakeMeta(uid=3), SECRET_HOTKEY, 10) == 3


def test_submitted_vector_matches_what_was_passed(wallet):
    sub = FakeSubtensor(result(True))
    set_weights(
        sub,
        wallet,
        FakeMeta(),
        netuid=10,
        uids=[3, 12, 201],
        weights=[0.05, 0.1, 0.85],
        version_key=2032,
    )
    intent, kwargs = sub.calls[0]
    assert intent.netuid == 10
    assert intent.uids == [3, 12, 201]
    assert intent.weights == [0.05, 0.1, 0.85]
    assert intent.version_key == 2032
    assert kwargs["wait_for_inclusion"] and kwargs["wait_for_finalization"]


def test_mismatched_lengths_raise_before_any_submit(wallet):
    sub = FakeSubtensor(result(True))
    with pytest.raises(WeightSetError):
        set_weights(sub, wallet, FakeMeta(), netuid=10, uids=[1, 2], weights=[1.0])
    assert sub.calls == []


def test_failure_never_emits_wallet_material(wallet, caplog):
    """A leak here would print a real hotkey in CI. Keep it impossible."""
    caplog.set_level(logging.DEBUG, logger=weights.__name__)
    sub = FakeSubtensor(result(False, "rate limit"))
    with pytest.raises(WeightSetError) as excinfo:
        _set(sub, wallet, FakeMeta(), attempts=2)
    emitted = str(excinfo.value) + caplog.text
    assert SECRET_HOTKEY not in emitted

    caplog.clear()
    with pytest.raises(ValidatorPermitError) as permit_exc:
        _set(sub, wallet, FakeMeta(permit=False))
    assert SECRET_HOTKEY not in str(permit_exc.value) + caplog.text
