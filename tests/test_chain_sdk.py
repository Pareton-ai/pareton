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


def test_timelock_reveal_at_maps_to_near_round():
    """11.x reveal_in is seconds, not blocks; a tiny value maps to a far-future
    DRAND round. Use reveal_at with an absolute near-future timestamp so the
    commitment reveals within minutes, not decades. The DRAND round itself may
    be large due to epoch offset; assert the reveal time is near instead."""
    from datetime import datetime, timedelta, timezone

    from bittensor import timelock

    reveal_at = datetime.now(timezone.utc) + timedelta(minutes=2)
    sealed = timelock.encrypt("x", reveal_at=reveal_at)
    assert sealed.reveal_at <= reveal_at + timedelta(seconds=30)


def test_miner_uses_unwrapped_metagraph():
    """Registration check needs a bare Metagraph. chain.rpc's same-named
    helper returns (meta, block, hash); importing that one crashes
    by_hotkey on every non-dry-run commit."""
    import chain.commitment as cc
    import miner.commit_patch as cp

    assert cp.fetch_metagraph is cc.fetch_metagraph


def test_plaintext_fields_decode_via_sdk():
    """Worker contract: miner plaintext fields must survive the SDK's own
    commitment decoder (_decode_fields concatenates Raw* variants)."""
    import bittensor.metagraph as mg

    from chain.commitment import encode_patch_commitment
    from miner.commit_patch import _plaintext_fields

    payload = encode_patch_commitment(
        campaign_id="11111111-1111-4111-8111-111111111111",
        baseline_commit="a" * 40,
        patch_hash="sha256:" + "b" * 64,
        retrieval_url="https://example.com/p.patch",
    )
    fields = _plaintext_fields(payload)
    variants = [list(f)[0] for f in fields]
    full, rem = divmod(len(payload), 128)
    expected = ["Raw128"] * full + ([f"Raw{rem}"] if rem else [])
    assert variants == expected
    assert mg._decode_fields(fields) == payload


def test_plaintext_fields_maxfields_guard():
    import pytest

    import miner.commit_patch as cp

    with pytest.raises(ValueError, match="MaxFields=3"):
        cp._plaintext_fields("x" * 385)


def test_verify_exception_still_exits_zero(monkeypatch, tmp_path):
    """A crashed read-back poll must not fail an already-landed commit."""
    from types import SimpleNamespace

    import bittensor as bt

    import miner.commit_patch as cp

    patch = tmp_path / "p.diff"
    patch.write_bytes(b"diff --git a/x b/x\n")

    monkeypatch.setattr(
        cp, "_http_json", lambda *_a, **_k: {"baseline_commit": "a" * 40}
    )
    monkeypatch.setattr(
        cp,
        "fetch_metagraph",
        lambda *_a, **_k: SimpleNamespace(by_hotkey=lambda _hk: object()),
    )
    monkeypatch.setattr(
        bt,
        "Wallet",
        lambda **_k: SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk")),
    )
    monkeypatch.setattr(
        bt,
        "Subtensor",
        lambda *a, **k: SimpleNamespace(
            submit_call=lambda *_a2, **_k2: SimpleNamespace(success=True)
        ),
    )
    monkeypatch.setattr(bt.calls.Commitments, "set_commitment", lambda **_k: object())

    def _boom(*_a, **_k):
        raise RuntimeError("finney peer dropped")

    monkeypatch.setattr(cp, "_await_visible", _boom)

    rc = cp.main(
        [
            "--campaign-id",
            "11111111-1111-4111-8111-111111111111",
            "--patch",
            str(patch),
            "--retrieval-url",
            "https://example.com/stage0/campaigns/c/patches/hk/p.diff",
            "--wallet-name",
            "w",
        ]
    )
    assert rc == 0


def test_dry_run_rejects_oversized_payload(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import bittensor as bt

    import miner.commit_patch as cp

    patch = tmp_path / "p.diff"
    patch.write_bytes(b"diff --git a/x b/x\n")

    monkeypatch.setattr(
        cp, "_http_json", lambda *_a, **_k: {"baseline_commit": "a" * 40}
    )
    monkeypatch.setattr(
        bt,
        "Wallet",
        lambda **_k: SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk")),
    )
    # Force a payload that exceeds MaxFields=3 after encode.
    monkeypatch.setattr(
        cp,
        "encode_patch_commitment",
        lambda **_k: "x" * 385,
    )

    rc = cp.main(
        [
            "--campaign-id",
            "11111111-1111-4111-8111-111111111111",
            "--patch",
            str(patch),
            "--retrieval-url",
            "https://example.com/stage0/campaigns/c/patches/hk/p.diff",
            "--wallet-name",
            "w",
            "--dry-run",
        ]
    )
    assert rc == 1


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
        retrieval_url="https://example.com/stage0/campaigns/c/patches/hk2/p.patch",
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


def test_scan_chain_orders_by_commit_block(monkeypatch):
    """Lower UID must not steal first-seen when its commit_block is later."""
    from chain import watcher
    from chain.commitment import encode_patch_commitment

    campaign_id = "11111111-1111-4111-8111-111111111111"
    patch_hash = "sha256:" + "b" * 64
    early = encode_patch_commitment(
        campaign_id=campaign_id,
        baseline_commit="a" * 40,
        patch_hash=patch_hash,
        retrieval_url="https://example.com/stage0/campaigns/c/patches/hk2/p.patch",
    )
    late = encode_patch_commitment(
        campaign_id=campaign_id,
        baseline_commit="a" * 40,
        patch_hash=patch_hash,
        retrieval_url="https://example.com/stage0/campaigns/c/patches/hk1/p.patch",
    )

    class Meta:
        # hk1 is UID 0 (would win under UID-order); hk2 committed earlier.
        hotkeys = ["hk1", "hk2"]
        coldkeys = ["ck1", "ck2"]

    monkeypatch.setattr(
        watcher,
        "fetch_chain_view",
        lambda *_a, **_k: (
            Meta(),
            {"hk1": [(20, late)], "hk2": [(10, early)]},
            20,
            None,
        ),
    )
    seen: list[tuple[int, str]] = []

    def _ingest(com):
        seen.append((com.commit_block, com.hotkey))
        return f"sid-{com.hotkey}"

    created, _hotkeys = watcher.scan_chain(object(), 10, ingest=_ingest)
    assert seen == [(10, "hk2"), (20, "hk1")]
    assert created == ["sid-hk2", "sid-hk1"]
