"""Unit tests for the standalone auditor weight relay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from scripts import auditor


class Response:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class Client:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.body)


def body(**overrides):
    value = {
        "computed_at_block": 8_965_603,
        "version_key": 2032,
        "burn_uid": 201,
        "weights": [0.1, 0.9],
    }
    value.update(overrides)
    return value


def test_fetch_weights_reads_the_live_dense_shape():
    client = Client(body())

    result = auditor.fetch_weights(client)

    assert result["weights"] == [0.1, 0.9]
    assert result["version_key"] == 2032
    assert client.calls == [(auditor.API_URL, {"timeout": 30})]


@pytest.mark.parametrize(
    "overrides",
    [
        {"weights": []},
        {"weights": [0.0, 0.0]},
        {"weights": [0.1, float("nan")]},
        {"weights": [0.1, -0.1]},
        {"version_key": "2032"},
        {"burn_uid": None},
    ],
)
def test_fetch_weights_rejects_an_unsafe_response(overrides):
    with pytest.raises((TypeError, ValueError)):
        auditor.fetch_weights(Client(body(**overrides)))


def test_submit_latest_uses_dense_uids(monkeypatch):
    seen = {}
    wallet = SimpleNamespace()
    subtensor = SimpleNamespace()
    meta = SimpleNamespace()

    monkeypatch.setattr(auditor, "fetch_metagraph", lambda *args, **kwargs: meta)

    def fake_set_weights(got_subtensor, got_wallet, got_meta, **kwargs):
        seen.update(kwargs)
        assert (got_subtensor, got_wallet, got_meta) == (subtensor, wallet, meta)

    monkeypatch.setattr(auditor, "set_weights", fake_set_weights)

    auditor.submit_latest(subtensor, wallet, client=Client(body()))

    assert list(seen["uids"]) == [0, 1]
    assert seen["weights"] == [0.1, 0.9]
    assert seen["version_key"] == 2032


def test_set_weights_refuses_a_hotkey_without_a_permit():
    meta = SimpleNamespace(
        by_hotkey=lambda _hk: SimpleNamespace(uid=16, validator_permit=False)
    )
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Test"))

    with pytest.raises(auditor.ValidatorPermitError):
        auditor.set_weights(
            SimpleNamespace(),
            wallet,
            meta,
            uids=[0, 1],
            weights=[0.1, 0.9],
            version_key=2032,
        )


def test_main_reads_settings_from_the_environment(monkeypatch):
    seen = {}
    monkeypatch.setenv("PARETON_WALLET_NAME", "env-coldkey")
    monkeypatch.setenv("PARETON_WALLET_HOTKEY", "env-hotkey")
    monkeypatch.setattr(auditor, "run", lambda **kwargs: seen.update(kwargs) or True)

    assert auditor.main([]) == 0
    assert seen == {
        "coldkey": "env-coldkey",
        "hotkey": "env-hotkey",
        "once": False,
    }


def test_command_line_flags_win_over_the_environment(monkeypatch):
    seen = {}
    monkeypatch.setenv("PARETON_WALLET_NAME", "env-coldkey")
    monkeypatch.setenv("PARETON_WALLET_HOTKEY", "env-hotkey")
    monkeypatch.setattr(auditor, "run", lambda **kwargs: seen.update(kwargs) or True)

    assert auditor.main(["--coldkey", "flag-ck"]) == 0
    assert seen["coldkey"] == "flag-ck"
    assert seen["hotkey"] == "env-hotkey"


def test_main_requires_the_wallet_settings(monkeypatch):
    monkeypatch.delenv("PARETON_WALLET_NAME", raising=False)
    monkeypatch.delenv("PARETON_WALLET_HOTKEY", raising=False)

    with pytest.raises(SystemExit):
        auditor.main([])


def test_set_weights_treats_an_unknown_result_shape_as_a_failure():
    """An SDK result without `success` must raise, never log a silent win."""
    meta = SimpleNamespace(
        by_hotkey=lambda _hk: SimpleNamespace(uid=16, validator_permit=True)
    )
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Test"))
    subtensor = SimpleNamespace(execute=lambda *a, **k: SimpleNamespace())

    with pytest.raises(auditor.WeightSetError):
        auditor.set_weights(
            subtensor,
            wallet,
            meta,
            uids=[0, 1],
            weights=[0.1, 0.9],
            version_key=2032,
        )
