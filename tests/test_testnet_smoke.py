"""Offline tests for the live testnet smoke helpers and workflow contract."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import testnet_smoke

pytestmark = pytest.mark.unit


def _set_test_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARETON_DATABASE_URL", "postgresql://test.example/db")
    monkeypatch.setenv("PARETON_TEST_DATABASE_URL", "postgresql://test.example/db")
    monkeypatch.setenv("PARETON_NETWORK", "test")
    monkeypatch.setenv("PARETON_NETUID", "543")
    monkeypatch.setattr(
        testnet_smoke,
        "EXPECTED_TEST_DATABASE_HOST_SHA256",
        hashlib.sha256(b"test.example").hexdigest(),
    )


def test_runtime_guard_accepts_explicit_test_configuration(monkeypatch):
    _set_test_runtime(monkeypatch)
    testnet_smoke._require_test_runtime()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("PARETON_DATABASE_URL", "postgresql://main.example/db", "must equal"),
        ("PARETON_NETWORK", "finney", "must be test"),
        ("PARETON_NETUID", "10", "must be 543"),
    ],
)
def test_runtime_guard_rejects_unsafe_configuration(monkeypatch, key, value, message):
    _set_test_runtime(monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match=message):
        testnet_smoke._require_test_runtime()


def test_runtime_guard_rejects_unapproved_database_host(monkeypatch):
    _set_test_runtime(monkeypatch)
    monkeypatch.setenv("PARETON_DATABASE_URL", "postgresql://prod.example/db")
    monkeypatch.setenv("PARETON_TEST_DATABASE_URL", "postgresql://prod.example/db")
    with pytest.raises(RuntimeError, match="approved Neon test branch"):
        testnet_smoke._require_test_runtime()


def test_database_host_hash_normalizes_neon_pooler_alias():
    direct = "postgresql://u:p@ep-test.us-east-2.aws.neon.tech/db"
    pooled = "postgresql://u:p@ep-test-pooler.us-east-2.aws.neon.tech/db"
    assert testnet_smoke._database_host_sha256(direct) == (
        testnet_smoke._database_host_sha256(pooled)
    )


def test_unique_patch_is_allowed_and_applies(tmp_path):
    patch = testnet_smoke.build_unique_patch("123-2")
    assert b"vllm/pareton_testnet_smoke/123-2.txt" in patch
    assert patch != testnet_smoke.build_unique_patch("123-3")

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    patch_path = tmp_path / "smoke.diff"
    patch_path.write_bytes(patch)
    subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_prepare_patch_writes_allowlisted_layout(tmp_path):
    values = testnet_smoke.prepare_patch(
        campaign_id="11111111-1111-4111-8111-111111111111",
        hotkey="5TestHotkey",
        run_id="run/42",
        http_root=tmp_path,
        public_base_url="http://127.0.0.1:9000",
        s3_prefix="stage0",
    )

    path = Path(values["patch_path"])
    assert path.is_file()
    assert "/stage0/campaigns/" in values["retrieval_url"]
    assert values["retrieval_url"].endswith("/run-42.diff")
    assert (
        values["patch_hash"]
        == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_restore_wallet_uses_mnemonic_without_printing_secret(monkeypatch, tmp_path):
    captured = {}

    class FakeWallet:
        def __init__(self, *, name, hotkey, path):
            captured.update(name=name, hotkey=hotkey, path=path)
            self.hotkey = SimpleNamespace(ss58_address="5Expected")

        def regenerate_hotkey(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "bittensor", SimpleNamespace(Wallet=FakeWallet))
    monkeypatch.setenv(
        "CI_TESTNET_WALLET_SEED",
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
    )

    testnet_smoke.restore_wallet(
        wallet_name="ci",
        wallet_hotkey="test",
        expected_hotkey="5Expected",
        wallet_path=tmp_path,
    )

    assert captured["mnemonic"].startswith("alpha beta")
    assert captured["use_password"] is False
    assert captured["overwrite"] is True
    assert captured["suppress"] is True


def test_poll_returns_soft_skip_when_chain_never_ingests(monkeypatch):
    _set_test_runtime(monkeypatch)
    monkeypatch.setattr(testnet_smoke, "get_submission", lambda _hash: None)
    assert (
        testnet_smoke.poll_for_built(
            "sha256:" + ("a" * 64),
            timeout_s=0,
            interval_s=0,
        )
        == 78
    )


def test_poll_refetches_submission_after_observing_built(monkeypatch):
    _set_test_runtime(monkeypatch)
    stale = {"id": "submission-1", "engine_image_ref": None}
    fresh = {"id": "submission-1", "engine_image_ref": "mock@sha256:abc"}
    rows = iter([stale, fresh])
    monkeypatch.setattr(testnet_smoke, "get_submission", lambda _hash: next(rows))
    monkeypatch.setattr(
        testnet_smoke,
        "list_events",
        lambda _sid: [
            {"state": state, "detail": {}}
            for state in testnet_smoke.EXPECTED_EVENT_SEQUENCE
        ],
    )
    asserted = []
    monkeypatch.setattr(testnet_smoke, "_assert_built", asserted.append)

    result = testnet_smoke.poll_for_built(
        "sha256:" + ("a" * 64),
        timeout_s=1,
        interval_s=0,
    )

    assert result == 0
    assert asserted == [fresh]


def test_workflow_contract_uses_testnet_and_mock_build():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "testnet-smoke.yml"
    ).read_text()

    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "environment: pareton-test" in workflow
    assert "PARETON_DATABASE_URL: ${{ secrets.PARETON_TEST_DATABASE_URL }}" in workflow
    assert "PARETON_NETWORK: test" in workflow
    assert 'PARETON_NETUID: "543"' in workflow
    assert "--retrieval-url" in workflow
    assert "--network test" in workflow
    assert "--netuid 543" in workflow
    assert "--scan-chain --mock-build" in workflow
    assert "PARETON_GHCR_TOKEN" not in workflow
    assert "PARETON_S3_ACCESS_KEY" not in workflow
    assert "pull_request:" not in workflow
