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


def test_submit_latest_uses_dense_uids_and_one_chain_attempt(monkeypatch):
    seen = {}
    wallet = SimpleNamespace()
    subtensor = SimpleNamespace()
    meta = SimpleNamespace()

    monkeypatch.setattr(
        auditor,
        "fetch_metagraph",
        lambda *args, **kwargs: (meta, 8_965_604, "0xabc"),
    )

    def fake_set_weights(got_subtensor, got_wallet, got_meta, **kwargs):
        seen.update(kwargs)
        assert (got_subtensor, got_wallet, got_meta) == (subtensor, wallet, meta)

    monkeypatch.setattr(auditor, "set_weights", fake_set_weights)

    auditor.submit_latest(subtensor, wallet, client=Client(body()))

    assert list(seen["uids"]) == [0, 1]
    assert seen["weights"] == [0.1, 0.9]
    assert seen["version_key"] == 2032
    assert seen["burn_uid"] == 201
    assert seen["attempts"] == 1


def test_submit_latest_uses_selected_network_and_netuid(monkeypatch):
    seen = {}

    def fake_metagraph(subtensor, netuid, **kwargs):
        seen["metagraph"] = (subtensor, netuid, kwargs)
        return SimpleNamespace(), 100, "0xabc"

    monkeypatch.setattr(auditor, "fetch_metagraph", fake_metagraph)
    monkeypatch.setattr(
        auditor,
        "set_weights",
        lambda subtensor, wallet, meta, **kwargs: seen.update(set=kwargs),
    )

    auditor.submit_latest(
        "subtensor",
        "wallet",
        network="test",
        netuid=292,
        client=Client(body()),
    )

    assert seen["metagraph"] == (
        "subtensor",
        292,
        {"network": "test", "attempts": 1},
    )
    assert seen["set"]["netuid"] == 292
