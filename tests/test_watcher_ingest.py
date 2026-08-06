"""Unit tests for chain watcher ingest guards (no DB/network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from chain.commitment import PatchCommitment
from chain import watcher
import config


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
