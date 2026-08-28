"""Unit tests for chain watcher ingest guards (no DB/network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

import config
from chain import watcher
from chain.commitment import PatchCommitment
from chain.payment import BlockPaymentView


def _com(**overrides) -> PatchCommitment:
    kwargs = dict(
        uid=1,
        hotkey="hk1",
        coldkey="ck1",
        commit_block=10,
        campaign_id="11111111-1111-4111-8111-111111111111",
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "b" * 64,
        retrieval_url="https://cdn.example.com/stage0/campaigns/c/patches/hk1/1.diff",
        raw="",
    )
    kwargs.update(overrides)
    return PatchCommitment(**kwargs)


@pytest.fixture(autouse=True)
def _cdn(monkeypatch):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")


def test_ingest_skips_hotkey_mismatch(monkeypatch):
    monkeypatch.setattr(
        watcher, "get_campaign", lambda _cid: SimpleNamespace(status="open")
    )
    called = {"insert": False}
    monkeypatch.setattr(
        watcher,
        "insert_submission",
        lambda **_k: called.__setitem__("insert", True) or "sid",
    )
    sid = watcher.ingest_commitment(
        _com(
            retrieval_url="https://cdn.example.com/stage0/campaigns/c/patches/other/1.diff"
        )
    )
    assert sid is None
    assert called["insert"] is False


def test_ingest_skips_non_allowlisted_url(monkeypatch):
    monkeypatch.setattr(
        watcher, "get_campaign", lambda _cid: SimpleNamespace(status="open")
    )
    called = {"insert": False}
    monkeypatch.setattr(
        watcher,
        "insert_submission",
        lambda **_k: called.__setitem__("insert", True) or "sid",
    )
    sid = watcher.ingest_commitment(
        _com(
            retrieval_url="https://evil.example.com/stage0/campaigns/c/patches/hk1/1.diff"
        )
    )
    assert sid is None
    assert called["insert"] is False


RECIPIENT = "5CiieAa5nzSMbw4LPkh2hqv9rfMPZX9ZfEcSjh3SYWNBzk3K"
FEE_TAO = 0.05
FEE_RAO = 50_000_000


def _transfer(*, signer="ck1", dest=RECIPIENT, amount=FEE_RAO) -> dict:
    return {
        "address": signer,
        "call": {
            "call_module": "Balances",
            "call_function": "transfer_keep_alive",
            "call_args": [
                {"name": "dest", "value": dest},
                {"name": "value", "value": amount},
            ],
        },
    }


def _paid_view(
    *,
    signer="ck1",
    dest=RECIPIENT,
    amount=FEE_RAO,
    index: int = 0,
    succeeded: bool = True,
) -> BlockPaymentView:
    extrinsics = [None] * index + [_transfer(signer=signer, dest=dest, amount=amount)]
    event_id = "ExtrinsicSuccess" if succeeded else "ExtrinsicFailed"
    events = [
        {
            "extrinsic_idx": index,
            "event": {
                "module_id": "System",
                "event_id": event_id,
                "attributes": {},
            },
        }
    ]
    return BlockPaymentView(extrinsics=extrinsics, events=events)


@pytest.fixture()
def inserted(monkeypatch):
    """Open campaign + captured insert kwargs, with no DB behind either."""
    monkeypatch.setattr(
        watcher,
        "get_campaign",
        lambda _cid: SimpleNamespace(
            status="open",
            submission_fee={"amount_tao": str(FEE_TAO), "recipient": RECIPIENT},
        ),
    )
    monkeypatch.setattr(watcher, "payment_ref_consumed", lambda _b, _t: False)
    seen: dict = {}

    def _insert(**kwargs):
        seen.update(kwargs)
        return "sid"

    monkeypatch.setattr(watcher, "insert_submission", _insert)
    return seen


def test_ingest_without_fee_needs_no_payment_proof(monkeypatch, inserted):
    monkeypatch.setattr(
        watcher,
        "get_campaign",
        lambda _cid: SimpleNamespace(
            status="open",
            submission_fee={"amount_tao": "0", "recipient": RECIPIENT},
        ),
    )
    sid = watcher.ingest_commitment(_com())
    assert sid == "sid"
    assert inserted["payment_block"] is None
    assert inserted["payment_tx"] is None


def test_ingest_with_fee_rejects_missing_proof(inserted):
    sid = watcher.ingest_commitment(_com())
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_accepts_verified_proof(inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=2),
        fetch_block=lambda _b: _paid_view(index=2),
    )
    assert sid == "sid"
    assert inserted["payment_block"] == 900
    assert inserted["payment_tx"] == 2


def test_ingest_with_fee_rejects_reused_proof(monkeypatch, inserted):
    monkeypatch.setattr(watcher, "payment_ref_consumed", lambda _b, _t: True)
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_payment_from_another_miner(inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(signer="ck-of-someone-else"),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_underpayment(inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(amount=FEE_RAO - 1),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_wrong_recipient(inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(dest="5SomeoneElse"),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_failed_transfer(inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(succeeded=False),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_unreadable_payment_block(inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: None,
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_when_chain_is_unreachable(inserted):
    # No fetcher wired means the proof cannot be checked, so nothing proceeds.
    sid = watcher.ingest_commitment(_com(payment_block=900, payment_tx=0))
    assert sid is None
    assert inserted == {}
