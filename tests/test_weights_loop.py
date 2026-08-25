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
    def __init__(self, hotkeys, *, permit: bool = True):
        self.hotkeys = list(hotkeys)
        self._permit = permit

    def by_hotkey(self, hotkey: str):
        if hotkey not in self.hotkeys:
            return None
        return SimpleNamespace(
            uid=self.hotkeys.index(hotkey),
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


def _tick(process, *, head, wallet, meta, enabled=True, force=False):
    return process.tick(
        head=head,
        subtensor=SimpleNamespace(),
        wallet=wallet,
        meta=meta,
        enabled=enabled,
        force=force,
    )


def test_cycle_due_on_first_run_and_after_cadence():
    assert loop.cycle_due(None, 10, 360) is True
    assert loop.cycle_due(1000, 1359, 360) is False
    assert loop.cycle_due(1000, 1360, 360) is True


def test_long_cycle_skips_the_next_tick_instead_of_overlapping(
    monkeypatch, wallet, meta, wired
):
    process = loop.WeightsProcess(last_block=None, cadence=360)
    nested = []

    real = loop.run_cycle

    def reentrant(*args, **kwargs):
        nested.append(_tick(process, head=1000, wallet=wallet, meta=meta, enabled=True))
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "run_cycle", reentrant)
    assert _tick(process, head=1000, wallet=wallet, meta=meta) == "computed"
    assert nested == ["skipped_overlap"]
    assert process.last_block == 1000


def test_waiting_when_cadence_has_not_elapsed(wallet, meta, wired):
    process = loop.WeightsProcess(last_block=1000, cadence=360)
    assert _tick(process, head=1100, wallet=wallet, meta=meta) == "waiting"
    assert wired["insert"] == []


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
    assert _tick(process, head=51, wallet=wallet, meta=meta) == "waiting"


def test_kill_switch_stores_and_never_calls_chain(wallet, meta, wired):
    process = loop.WeightsProcess(last_block=None, cadence=360)
    assert (
        _tick(process, head=10, wallet=wallet, meta=meta, enabled=False) == "computed"
    )
    assert wired["insert"]
    assert wired["set"] == []
    assert wired["mark"] == []


def test_guard_violation_stores_nothing(monkeypatch, wallet, meta, wired):
    def bad(*args, **kwargs):
        raise WeightVectorError("burn is negative")

    monkeypatch.setattr(loop, "build_weight_vector", bad)
    process = loop.WeightsProcess(last_block=None, cadence=360)
    assert _tick(process, head=10, wallet=wallet, meta=meta) == "aborted"
    assert wired["insert"] == []
    assert wired["set"] == []
    assert wired["mark"] == []
    assert process.last_block is None
