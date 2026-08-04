"""Chain SDK canary: assert the bittensor 11.x surface Pareton depends on.

Offline (no network). Run after any bittensor bump — failures here mean
`chain/rpc.py` and `miner/commit_patch.py` need a port to the new SDK.
"""

from __future__ import annotations

import chain.rpc as rpc


def test_sdk_metagraph_read_surface():
    import bittensor as bt
    import bittensor.metagraph as mg

    assert callable(mg.fetch)
    assert callable(mg.fetch_commitments)
    fields = set(mg.NeuronCommitment.__dataclass_fields__)
    assert {"hotkey", "uid", "block", "data", "encrypted", "revealed"} <= fields
    assert hasattr(bt.Metagraph, "hotkeys")
    assert hasattr(bt.Metagraph, "coldkeys")
    assert callable(bt.Metagraph.by_hotkey)


def test_sdk_submit_surface():
    import bittensor as bt
    from bittensor import timelock

    assert callable(bt.Subtensor)
    assert callable(bt.Client)
    assert callable(bt.calls.Commitments.set_commitment)
    assert callable(timelock.encrypt)
    assert bt.Policy(allow_raw_calls=True).allow_raw_calls
    for name in ("block", "block_info", "submit_call"):
        assert hasattr(bt.Subtensor, name)


def test_miner_uses_unwrapped_metagraph():
    """Registration check needs a bare Metagraph. chain.rpc's same-named
    helper returns (meta, block, hash); importing that one crashes
    by_hotkey on every non-dry-run commit."""
    import chain.commitment as cc
    import miner.commit_patch as cp

    assert cp.fetch_metagraph is cc.fetch_metagraph


def test_commitment_entries_mapping():
    class Plaintext:
        uid = 3
        block = 100
        encrypted = False
        data = '{"v":1}'
        revealed = [(90, '{"v":0}')]

    assert rpc._commitment_entries(Plaintext()) == [(90, '{"v":0}'), (100, '{"v":1}')]

    class Sealed(Plaintext):
        encrypted = True
        data = "encrypted-blob"

    assert rpc._commitment_entries(Sealed()) == [(90, '{"v":0}')]

    class Empty(Plaintext):
        encrypted = True
        data = ""
        revealed = []

    assert rpc._commitment_entries(Empty()) == []


def test_scan_chain_folds_commitments(monkeypatch):
    from chain import watcher
    from chain.commitment import encode_patch_commitment

    payload = encode_patch_commitment(
        campaign_id="11111111-1111-4111-8111-111111111111",
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "b" * 64,
        retrieval_url="https://example.com/p.patch",
    )

    class Meta:
        hotkeys = ["hk1", "hk2"]
        coldkeys = ["ck1", "ck2"]

    monkeypatch.setattr(
        watcher,
        "fetch_chain_view",
        lambda *_a, **_k: (Meta(), {"hk2": [(7, payload)]}, 7, None),
    )
    created, hotkeys = watcher.scan_chain(object(), 10, ingest=lambda _com: "sid-1")
    assert created == ["sid-1"]
    assert hotkeys == ["hk1", "hk2"]
