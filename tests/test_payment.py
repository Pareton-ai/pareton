"""Unit tests for submission-fee proof verification (no chain/network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from chain.payment import (
    BlockPaymentView,
    PaymentCheck,
    extract_transfer,
    extrinsic_succeeded,
    fee_rao,
    fetch_block_payment_view,
    verify_payment,
)

RECIPIENT = "5CiieAa5nzSMbw4LPkh2hqv9rfMPZX9ZfEcSjh3SYWNBzk3K"
HOTKEY = "5HotkeyOfTheMiner"
COLDKEY = "5ColdkeyThatOwnsIt"
FEE_RAO = 50_000_000  # 0.05 TAO


def _transfer(
    *,
    signer: str = COLDKEY,
    dest: object = RECIPIENT,
    amount: object = FEE_RAO,
    function: str = "transfer_keep_alive",
    module: str = "Balances",
) -> dict:
    return {
        "address": signer,
        "call": {
            "call_module": module,
            "call_function": function,
            "call_args": [
                {"name": "dest", "type": "MultiAddress", "value": dest},
                {"name": "value", "type": "Compact<u128>", "value": amount},
            ],
        },
    }


def _timestamp_inherent() -> dict:
    return {
        "call": {
            "call_module": "Timestamp",
            "call_function": "set",
            "call_args": [{"name": "now", "value": 1_700_000_000_000}],
        }
    }


def _system_event(event_id: str, index: int = 0, *, via_phase: bool = False) -> dict:
    event = {
        "event": {
            "module_id": "System",
            "event_id": event_id,
            "attributes": {},
        }
    }
    if via_phase:
        event["phase"] = {"ApplyExtrinsic": index}
    else:
        event["extrinsic_idx"] = index
    return event


def _check(
    extrinsics: list, index: int = 0, events: list | None = None, **overrides
) -> PaymentCheck:
    kwargs = dict(
        recipient=RECIPIENT,
        min_amount_rao=FEE_RAO,
        hotkey=HOTKEY,
        coldkey=COLDKEY,
    )
    kwargs.update(overrides)
    if events is None:
        events = [_system_event("ExtrinsicSuccess", index)]
    return verify_payment(
        extrinsics=extrinsics, events=events, extrinsic_index=index, **kwargs
    )


@pytest.mark.parametrize(
    ("fee_tao", "expected"),
    [("0.05", 50_000_000), (0.05, 50_000_000), (0.1, 100_000_000), (1, 10**9)],
)
def test_fee_rao_is_exact_integer(fee_tao, expected):
    # Amounts are compared in integer RAO, never as float TAO.
    assert fee_rao(fee_tao) == expected


def test_extract_transfer_reads_signer_dest_amount():
    assert extract_transfer(_transfer()) == (COLDKEY, RECIPIENT, FEE_RAO)


def test_extract_transfer_accepts_wrapped_multiaddress():
    got = extract_transfer(_transfer(dest={"Id": RECIPIENT}))
    assert got == (COLDKEY, RECIPIENT, FEE_RAO)


def test_extract_transfer_ignores_other_calls():
    assert extract_transfer(_timestamp_inherent()) is None
    assert extract_transfer(_transfer(module="SubtensorModule")) is None
    assert extract_transfer(_transfer(function="transfer_all")) is None
    assert extract_transfer(None) is None


def test_transfer_allow_death_counts_as_payment():
    check = _check([_transfer(function="transfer_allow_death")])
    assert check.ok


def test_payment_signed_by_coldkey_is_accepted():
    check = _check([_transfer(signer=COLDKEY)])
    assert check.ok
    assert check.amount_rao == FEE_RAO


def test_payment_signed_by_hotkey_is_accepted():
    assert _check([_transfer(signer=HOTKEY)]).ok


def test_overpayment_is_accepted():
    assert _check([_transfer(amount=FEE_RAO * 3)]).ok


def test_payment_at_index_inside_a_full_block():
    extrinsics = [_timestamp_inherent(), _transfer()]
    assert _check(extrinsics, index=1).ok
    # The index must point at the transfer, not merely exist in the block.
    assert _check(extrinsics, index=0).reason == "payment_not_a_transfer"


@pytest.mark.parametrize("index", [-1, 1, 99])
def test_index_outside_the_block_is_rejected(index):
    check = _check([_transfer()], index=index)
    assert not check.ok
    assert check.reason == "payment_index_out_of_range"


def test_payment_to_another_recipient_is_rejected():
    check = _check([_transfer(dest="5SomeoneElsesWallet")])
    assert not check.ok
    assert check.reason == "payment_recipient_mismatch"


def test_payment_below_the_fee_is_rejected():
    check = _check([_transfer(amount=FEE_RAO - 1)])
    assert not check.ok
    assert check.reason == "payment_below_fee"


def test_payment_from_an_unrelated_key_is_rejected():
    # Pointing at somebody else's transfer must not buy a submission.
    check = _check([_transfer(signer="5AnotherMinersColdkey")])
    assert not check.ok
    assert check.reason == "payment_signer_not_miner"


def test_amount_that_is_not_an_integer_is_rejected():
    check = _check([_transfer(amount=None)])
    assert not check.ok
    assert check.reason == "payment_not_a_transfer"


def test_failed_extrinsic_does_not_count_as_payment():
    # Failed Substrate extrinsics stay in the block with intact call args.
    check = _check(
        [_transfer()],
        events=[_system_event("ExtrinsicFailed")],
    )
    assert not check.ok
    assert check.reason == "payment_extrinsic_failed"


def test_missing_success_event_is_rejected():
    check = _check([_transfer()], events=[])
    assert not check.ok
    assert check.reason == "payment_outcome_unknown"


def test_success_event_via_apply_extrinsic_phase():
    assert extrinsic_succeeded(
        [_system_event("ExtrinsicSuccess", 2, via_phase=True)], 2
    )
    check = _check(
        [_timestamp_inherent(), _timestamp_inherent(), _transfer()],
        index=2,
        events=[_system_event("ExtrinsicSuccess", 2, via_phase=True)],
    )
    assert check.ok


def test_fetch_block_payment_view_returns_none_when_unreadable():
    class Boom:
        def block_info(self, _block):
            raise RuntimeError("rpc down")

    class Missing:
        def block_info(self, _block):
            return None

    assert fetch_block_payment_view(Boom(), 42) is None
    assert fetch_block_payment_view(Missing(), 42) is None


def test_fetch_block_payment_view_loads_extrinsics_and_events():
    class Sub:
        def block_info(self, _block):
            return SimpleNamespace(extrinsics=[_transfer()])

        def query(self, _item, block=None):
            assert block == 42
            return [_system_event("ExtrinsicSuccess")]

    view = fetch_block_payment_view(Sub(), 42)
    assert view == BlockPaymentView(
        extrinsics=[_transfer()],
        events=[_system_event("ExtrinsicSuccess")],
    )
