"""Campaign submission fee contract tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from campaign.fees import submission_fee_rao, validate_submission_fee
from campaign.manifest import build_manifest
from campaign.models import SLA

FEE = {
    "amount_tao": "0.0005",
    "recipient": "5CiieAa5nzSMbw4LPkh2hqv9rfMPZX9ZfEcSjh3SYWNBzk3K",
}


def _manifest_kwargs(**overrides):
    kwargs = {
        "campaign_id": uuid4(),
        "profile_id": uuid4(),
        "baseline_repo": "https://github.com/vllm-project/vllm.git",
        "baseline_commit": "a" * 40,
        "base_image_digest": "sha256:" + "b" * 64,
        "gpu_skus": ["H200"],
        "workload_trace_sha256": "sha256:" + "c" * 64,
        "workload_trace_url": "https://example.com/t",
        "sla": SLA(),
        "scoring_config_sha256": None,
        "scoring_config_url": None,
        "allowed_paths": ["vllm/**"],
        "denied_paths": ["tests/**"],
        "priority_metric": "throughput",
        "success_threshold": ">=10% at SLA",
        "submission_fee": FEE,
    }
    kwargs.update(overrides)
    return kwargs


def test_submission_fee_is_canonical_and_exact():
    fee = validate_submission_fee(
        {"amount_tao": "0.000500000", "recipient": f"  {FEE['recipient']}  "}
    )
    assert fee == FEE
    assert submission_fee_rao(fee) == 500_000


@pytest.mark.parametrize(
    "amount",
    [
        "-1",
        "nan",
        "0.0000000001",
        True,
        None,
        0.15,
        "1.00000000000000000000000000001",
        "18446744073.709551616",
    ],
)
def test_submission_fee_rejects_invalid_amount(amount):
    with pytest.raises(ValueError, match="submission_fee.amount_tao"):
        validate_submission_fee({"amount_tao": amount, "recipient": FEE["recipient"]})


def test_fee_amount_and_recipient_do_not_change_the_manifest_hash():
    kwargs = _manifest_kwargs()
    base = build_manifest(**kwargs)
    other_amount = build_manifest(
        **{**kwargs, "submission_fee": {**FEE, "amount_tao": "0.001"}}
    )
    other_recipient = build_manifest(
        **{**kwargs, "submission_fee": {**FEE, "recipient": "5Other"}}
    )
    assert (
        len(
            {
                base.manifest_hash,
                other_amount.manifest_hash,
                other_recipient.manifest_hash,
            }
        )
        == 1
    )
    assert base.to_public_dict()["submission_fee"] == FEE


@pytest.mark.parametrize("amount", ["1e-999999999", "0.0000000010000000000000000001"])
def test_tiny_amounts_never_round_to_free(amount):
    with pytest.raises(ValueError):
        submission_fee_rao({**FEE, "amount_tao": amount})


def test_fee_history_selects_payment_block_not_current_price():
    from campaign.fees import fee_at_block

    history = [
        {**FEE, "effective_from_block": 0},
        {**FEE, "amount_tao": "0.15", "effective_from_block": 1000},
    ]
    assert fee_at_block(history, 999) == FEE
    assert fee_at_block(history, 1000)["amount_tao"] == "0.15"


@pytest.mark.parametrize("blocks", [[], [1], [0, 0], [0, -1], [0, True], [0, 1.5]])
def test_fee_history_rejects_missing_or_ambiguous_boundaries(blocks):
    from campaign.fees import validate_fee_history

    with pytest.raises(ValueError):
        validate_fee_history([{**FEE, "effective_from_block": b} for b in blocks])


def test_negative_zero_is_canonical_zero():
    assert validate_submission_fee({**FEE, "amount_tao": "-0"})["amount_tao"] == "0"


@pytest.mark.parametrize("block", [0, -1, True, 1.5, 2**63])
def test_fee_update_rejects_invalid_chain_height(block):
    from campaign.set_fee import set_fee

    with pytest.raises(ValueError, match="positive integer"):
        set_fee("unused", "0.15", current_block=block)


@pytest.mark.parametrize("amount", ["0.15", "0.0001", "0"])
def test_next_block_fee_update_preserves_terms_and_rejects_same_block(
    monkeypatch, amount
):
    from contextlib import contextmanager
    from campaign import set_fee
    from campaign.fees import fee_at_block

    history = [{**FEE, "effective_from_block": 0}]

    class Cursor:
        def __init__(self):
            self.written = None

        def execute(self, query, params):
            if query.startswith("UPDATE"):
                self.written = params[0].adapted
                assert "manifest_hash" not in query
                assert "customer_signoff" not in query

        def fetchone(self):
            return (history,)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    cur = Cursor()

    class Conn:
        def cursor(self):
            return cur

    @contextmanager
    def connect():
        yield Conn()

    monkeypatch.setattr(set_fee, "db_connection", connect)
    result = set_fee.set_fee("campaign", amount, current_block=900)
    assert cur.written == [*history, result]
    assert result["effective_from_block"] == 901
    assert result["amount_tao"] == amount
    assert fee_at_block(cur.written, 900) == FEE
    assert fee_at_block(cur.written, 901) == {**FEE, "amount_tao": amount}
    history.append(result)
    with pytest.raises(ValueError, match="retry after the chain advances"):
        set_fee.set_fee("campaign", "0.2", current_block=900)
    assert cur.written == history


@pytest.mark.parametrize(
    "history",
    [
        [],
        [{**FEE, "amount_tao": "0.15", "effective_from_block": 0}],
        [{**FEE, "recipient": "different", "effective_from_block": 0}],
    ],
)
def test_insert_rejects_empty_or_conflicting_history_before_db(monkeypatch, history):
    from campaign import store

    manifest = build_manifest(**_manifest_kwargs())
    manifest.submission_fee_history = history
    monkeypatch.setattr(
        store, "db_connection", lambda: pytest.fail("database accessed")
    )
    with pytest.raises(ValueError):
        store.insert_campaign(manifest)


@pytest.mark.parametrize("supplied", [False, True])
def test_insert_preserves_valid_history_or_defaults_none(monkeypatch, supplied):
    from contextlib import contextmanager
    from campaign import store

    manifest = build_manifest(**_manifest_kwargs())
    history = [{**FEE, "effective_from_block": 0}]
    if supplied:
        history.append({**FEE, "amount_tao": "0.15", "effective_from_block": 100})
        manifest.submission_fee_history = [
            {**history[0], "amount_tao": "0.0005000"},
            history[1],
        ]
    recorded = []

    class Cursor:
        def execute(self, _sql, params):
            recorded.append(params[-1].adapted)

        def fetchone(self):
            return [manifest.campaign_id]

    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()

    @contextmanager
    def connection():
        yield Connection()

    monkeypatch.setattr(store, "db_connection", connection)
    assert store.insert_campaign(manifest) == manifest.campaign_id
    assert recorded == [history]


def test_set_fee_cli_uses_observed_block_without_activation_argument(
    monkeypatch, capsys
):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import bittensor as bt
    from campaign import set_fee

    @contextmanager
    def subtensor(**_kwargs):
        yield SimpleNamespace(block=900)

    calls = []

    def publish(campaign_id, amount, *, current_block):
        calls.append((campaign_id, amount, current_block))
        return {**FEE, "amount_tao": amount, "effective_from_block": current_block + 1}

    monkeypatch.setattr(bt, "Subtensor", subtensor)
    monkeypatch.setattr(set_fee, "set_fee", publish)
    campaign_id = str(uuid4())
    args = ["--campaign-id", campaign_id, "--amount-tao", "0.15"]
    assert set_fee.main(args) == 0
    assert calls == [(campaign_id, "0.15", 900)]
    assert "Published 0.15 TAO from block 901" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        set_fee.main([*args, "--effective-from-block", "1000"])
    assert exc.value.code == 2
    assert len(calls) == 1
