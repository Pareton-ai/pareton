"""Unit tests for the pareton-weights cadence (PAR-105).

No database, no chain, no network. Store and SDK calls are fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from chain.weights import WeightSetError
from weights.build import WeightVectorError
from weights import loop

SIGNER = "5Signer"
LEADER = "5LeaderGone"
BURN = 3


class FakeMeta:
    def __init__(self, hotkeys, *, permit: bool = True, uid_of: dict | None = None):
        self.hotkeys = list(hotkeys)
        self._permit = permit
        self._uid_of = uid_of

    def by_hotkey(self, hotkey: str):
        if hotkey not in self.hotkeys:
            return None
        uid = (
            self._uid_of[hotkey]
            if self._uid_of is not None and hotkey in self._uid_of
            else self.hotkeys.index(hotkey)
        )
        return SimpleNamespace(
            uid=uid,
            validator_permit=self._permit if hotkey == SIGNER else False,
        )


@pytest.fixture
def wallet():
    return SimpleNamespace(hotkey=SimpleNamespace(ss58_address=SIGNER))


@pytest.fixture
def meta():
    return FakeMeta([SIGNER, "5Other", "5Third", "5Burn"])


@pytest.fixture
def calls():
    return {"insert": [], "mark": [], "set": [], "vacate": []}


@pytest.fixture
def wired(monkeypatch, calls):
    monkeypatch.setattr(loop.config, "BURN_UID", BURN)
    monkeypatch.setattr(loop.config, "NETUID", 10)
    monkeypatch.setattr(loop.config, "VERSION_KEY", 2032)
    monkeypatch.setattr(loop.config, "OVERTAKE_EPSILON", 0.01)
    monkeypatch.setattr(loop, "list_idle_seated_leaders", lambda: [])
    monkeypatch.setattr(loop, "list_open_campaign_emissions", lambda: [])
    monkeypatch.setattr(loop, "obs", SimpleNamespace(weights_computed=lambda **k: None))

    def _insert(**kwargs):
        calls["insert"].append(kwargs)
        return 7

    def _mark(row_id, *, ok, error):
        calls["mark"].append({"row_id": row_id, "ok": ok, "error": error})

    def _set(*args, **kwargs):
        calls["set"].append(kwargs)

    def _vacate(campaign_id, *, epsilon):
        calls["vacate"].append(str(campaign_id))
        return True

    monkeypatch.setattr(loop, "insert_weight_set", _insert)
    monkeypatch.setattr(loop, "mark_weight_set_result", _mark)
    monkeypatch.setattr(loop, "set_weights", _set)
    monkeypatch.setattr(loop, "vacate_leader_if_idle", _vacate)
    return calls


def _tick(process, *, head, wallet, meta, enabled=True):
    return process.tick(
        head=head,
        subtensor=SimpleNamespace(),
        wallet=wallet,
        meta=meta,
        enabled=enabled,
    )


def test_cycle_due_on_first_run_and_after_cadence():
    assert loop.cycle_due(None, 10, 360) is True
    assert loop.cycle_due(1000, 1359, 360) is False
    assert loop.cycle_due(1000, 1360, 360) is True


def test_hotkey_uids_uses_by_hotkey_and_raises_on_mismatch(meta):
    mapping, uid_count = loop._hotkey_uids(meta)
    assert uid_count == 4
    assert mapping[SIGNER] == 0
    assert mapping["5Burn"] == 3

    missing = FakeMeta([SIGNER, "5Ghost"])
    missing.by_hotkey = lambda hk: None
    with pytest.raises(WeightVectorError, match="by_hotkey"):
        loop._hotkey_uids(missing)

    bad = FakeMeta([SIGNER, "5Other"], uid_of={SIGNER: 99, "5Other": 1})
    with pytest.raises(WeightVectorError, match="outside"):
        loop._hotkey_uids(bad)


def test_running_round_is_skipped_by_reconcile(monkeypatch, wallet, meta, wired):
    monkeypatch.setattr(loop, "list_idle_seated_leaders", lambda: [])
    loop.run_cycle(SimpleNamespace(), wallet, meta, current_block=1000, enabled=True)
    assert wired["vacate"] == []


def test_idle_deregistered_leader_is_vacated(monkeypatch, wallet, meta, wired):
    monkeypatch.setattr(
        loop,
        "list_idle_seated_leaders",
        lambda: [{"campaign_id": "c1", "hotkey": LEADER}],
    )
    loop.run_cycle(SimpleNamespace(), wallet, meta, current_block=1000, enabled=True)
    assert wired["vacate"] == ["c1"]


def test_missing_permit_does_not_vacate(monkeypatch, wallet, wired):
    meta = FakeMeta([SIGNER, "5Other", "5Third", "5Burn"], permit=False)
    monkeypatch.setattr(
        loop,
        "list_idle_seated_leaders",
        lambda: [{"campaign_id": "c1", "hotkey": LEADER}],
    )
    process = loop.WeightsProcess(last_block=None, cadence=360)
    assert _tick(process, head=50, wallet=wallet, meta=meta) == "aborted"
    assert wired["vacate"] == []
    assert wired["insert"] == []
    assert process.last_block == 50


def test_set_weights_failure_records_set_ok_false_and_survives(
    monkeypatch, wallet, meta, wired
):
    def boom(*args, **kwargs):
        raise WeightSetError("chain rejected the weight vector")

    monkeypatch.setattr(loop, "set_weights", boom)
    process = loop.WeightsProcess(last_block=None, cadence=360)
    assert _tick(process, head=50, wallet=wallet, meta=meta) == "computed"
    assert wired["insert"]
    assert wired["mark"] == [
        {"row_id": 7, "ok": False, "error": "chain rejected the weight vector"}
    ]
    assert process.last_block == 50
    assert loop.cycle_due(process.last_block, 51, 360) is False


def test_kill_switch_stores_and_never_calls_chain(wallet, meta, wired):
    process = loop.WeightsProcess(last_block=None, cadence=360)
    assert (
        _tick(process, head=10, wallet=wallet, meta=meta, enabled=False) == "computed"
    )
    assert wired["insert"]
    assert wired["set"] == []
    assert wired["mark"] == []


def test_guard_violation_stores_nothing_and_advances_last_block(
    monkeypatch, wallet, meta, wired
):
    def bad(*args, **kwargs):
        raise WeightVectorError("burn is negative")

    monkeypatch.setattr(loop, "build_weight_vector", bad)
    process = loop.WeightsProcess(last_block=None, cadence=360)
    assert _tick(process, head=10, wallet=wallet, meta=meta) == "aborted"
    assert wired["insert"] == []
    assert wired["set"] == []
    assert wired["mark"] == []
    assert process.last_block == 10


def test_uid_map_mismatch_aborts_and_advances(wallet, wired):
    missing = FakeMeta([SIGNER, "5Ghost"])
    missing.by_hotkey = lambda hk: None
    process = loop.WeightsProcess(last_block=None, cadence=360)
    assert _tick(process, head=10, wallet=wallet, meta=missing) == "aborted"
    assert wired["insert"] == []
    assert wired["vacate"] == []
    assert process.last_block == 10


def test_tick_does_not_catch_unexpected_errors(monkeypatch, wallet, meta, wired):
    def boom(*args, **kwargs):
        raise RuntimeError("metagraph shape changed")

    monkeypatch.setattr(loop, "run_cycle", boom)
    process = loop.WeightsProcess(last_block=None, cadence=360)
    with pytest.raises(RuntimeError, match="metagraph shape changed"):
        _tick(process, head=10, wallet=wallet, meta=meta)
