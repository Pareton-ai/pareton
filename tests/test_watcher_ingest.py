"""Unit tests for chain watcher ingest guards (no DB/network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

import config
from campaign.store import CampaignHotkeyDisqualified
from chain import watcher
from chain.commitment import PatchCommitment
from chain.payment import BlockPaymentView
from gate.integrity import hash_patch_bytes, patch_fingerprint_bytes


PATCH = b"diff --git a/vllm/x.py b/vllm/x.py\n+x = 1\n"


def _com(**overrides) -> PatchCommitment:
    kwargs = dict(
        uid=1,
        hotkey="hk1",
        coldkey="ck1",
        commit_block=10,
        campaign_id="11111111-1111-4111-8111-111111111111",
        baseline_commit="a" * 40,
        patch_hash=hash_patch_bytes(PATCH),
        retrieval_url="https://cdn.example.com/stage0/campaigns/c/patches/hk1/1.diff",
        raw="",
    )
    kwargs.update(overrides)
    return PatchCommitment(**kwargs)


@pytest.fixture(autouse=True)
def _cdn(monkeypatch):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(watcher, "fetch_patch_bytes", lambda _url, **_kwargs: PATCH)
    monkeypatch.setattr(watcher, "get_submission_for_campaign", lambda *_args: None)
    watcher._failed_hash_checks.clear()
    yield
    watcher._failed_hash_checks.clear()


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


def test_ingest_skips_patch_hash_already_seen(monkeypatch):
    monkeypatch.setattr(
        watcher, "get_campaign", lambda _cid: SimpleNamespace(status="open")
    )
    monkeypatch.setattr(watcher, "get_submission_for_campaign", lambda *_args: {})
    called = {"fetch": False, "insert": False}
    monkeypatch.setattr(
        watcher,
        "fetch_patch_bytes",
        lambda _url, **_kwargs: called.__setitem__("fetch", True) or PATCH,
    )
    monkeypatch.setattr(
        watcher,
        "insert_submission",
        lambda **_kwargs: called.__setitem__("insert", True),
    )

    assert watcher.ingest_commitment(_com()) is None
    assert called == {"fetch": False, "insert": False}


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
        watcher, "get_campaign", lambda _cid: SimpleNamespace(status="open")
    )
    monkeypatch.setattr(watcher, "payment_ref_consumed", lambda _b, _t: False)
    seen: dict = {}

    def _insert(**kwargs):
        seen.update(kwargs)
        return "sid"

    monkeypatch.setattr(
        watcher,
        "insert_submission",
        _insert,
    )
    return seen


@pytest.fixture()
def fee_on(monkeypatch):
    monkeypatch.setattr(config, "SUBMISSION_FEE_TAO", FEE_TAO)
    monkeypatch.setattr(config, "PAYMENT_RECIPIENT_ADDRESS", RECIPIENT)


def test_ingest_without_fee_needs_no_payment_proof(monkeypatch, inserted):
    monkeypatch.setattr(config, "SUBMISSION_FEE_TAO", 0)
    sid = watcher.ingest_commitment(_com())
    assert sid == "sid"
    assert inserted["payment_block"] is None
    assert inserted["payment_tx"] is None
    assert inserted["patch_fingerprint"] == patch_fingerprint_bytes(PATCH)


def test_ingest_rejects_fingerprint_duplicate(monkeypatch, inserted):
    monkeypatch.setattr(config, "SUBMISSION_FEE_TAO", 0)
    monkeypatch.setattr(
        watcher,
        "insert_submission",
        lambda **kwargs: None,
    )
    assert watcher.ingest_commitment(_com()) is None


def test_ingest_caches_raw_hash_mismatch_before_insert(monkeypatch, inserted):
    monkeypatch.setattr(config, "SUBMISSION_FEE_TAO", 0)
    attempt_limits: list[int | None] = []

    def _fetch(_url, *, attempts=None):
        attempt_limits.append(attempts)
        return b"other"

    monkeypatch.setattr(watcher, "fetch_patch_bytes", _fetch)
    assert watcher.ingest_commitment(_com()) is None
    assert watcher.ingest_commitment(_com()) is None
    assert inserted == {}
    assert attempt_limits == [1]


def test_ingest_retries_fetch_failure_on_later_scan(monkeypatch, inserted):
    monkeypatch.setattr(config, "SUBMISSION_FEE_TAO", 0)
    attempt_limits: list[int | None] = []

    def _fetch(_url, *, attempts=None):
        attempt_limits.append(attempts)
        raise RuntimeError("cdn unavailable")

    monkeypatch.setattr(watcher, "fetch_patch_bytes", _fetch)
    assert watcher.ingest_commitment(_com()) is None
    assert watcher.ingest_commitment(_com()) is None
    assert inserted == {}
    assert attempt_limits == [1, 1]


def test_ingest_skips_campaign_disqualified_hotkey(monkeypatch, inserted):
    def _blocked(**_kwargs):
        raise CampaignHotkeyDisqualified("blocked")

    monkeypatch.setattr(watcher, "insert_submission", _blocked)
    assert watcher.ingest_commitment(_com()) is None


def test_ingest_with_fee_rejects_missing_proof(fee_on, inserted):
    sid = watcher.ingest_commitment(_com())
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_accepts_verified_proof(fee_on, inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=2),
        fetch_block=lambda _b: _paid_view(index=2),
    )
    assert sid == "sid"
    assert inserted["payment_block"] == 900
    assert inserted["payment_tx"] == 2


def test_ingest_with_fee_rejects_reused_proof(monkeypatch, fee_on, inserted):
    monkeypatch.setattr(watcher, "payment_ref_consumed", lambda _b, _t: True)
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_payment_from_another_miner(fee_on, inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(signer="ck-of-someone-else"),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_underpayment(fee_on, inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(amount=FEE_RAO - 1),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_wrong_recipient(fee_on, inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(dest="5SomeoneElse"),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_failed_transfer(fee_on, inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: _paid_view(succeeded=False),
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_unreadable_payment_block(fee_on, inserted):
    sid = watcher.ingest_commitment(
        _com(payment_block=900, payment_tx=0),
        fetch_block=lambda _b: None,
    )
    assert sid is None
    assert inserted == {}


def test_ingest_with_fee_rejects_when_chain_is_unreachable(fee_on, inserted):
    # No fetcher wired means the proof cannot be checked, so nothing proceeds.
    sid = watcher.ingest_commitment(_com(payment_block=900, payment_tx=0))
    assert sid is None
    assert inserted == {}
