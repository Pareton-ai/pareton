"""Disclosure policy timing, persistence, and operator entry points."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from test_manifest import _manifest_kwargs

from campaign import set_patch_visibility as command
from campaign import store
from campaign.manifest import build_manifest
from storage.visibility import patch_is_revealed, patch_reveal_at


@pytest.mark.parametrize("mode", ["private", "public_after_reveal"])
def test_no_evaluation_never_reveals_even_with_zero_delay(mode):
    policy = {"mode": mode}
    if mode == "public_after_reveal":
        policy["reveal_delay_s"] = 0
    assert patch_reveal_at(None, policy) is None
    assert not patch_is_revealed(None, policy)


def test_reveal_clock_uses_aware_time_and_inclusive_boundary():
    evaluated = datetime(2026, 10, 1, 14, tzinfo=timezone(timedelta(hours=2)))
    policy = {"mode": "public_after_reveal", "reveal_delay_s": 60}
    release = datetime(2026, 10, 1, 12, 1, tzinfo=timezone.utc)
    assert patch_reveal_at(evaluated, policy) == release
    assert not patch_is_revealed(
        evaluated, policy, now=release - timedelta(microseconds=1)
    )
    assert patch_is_revealed(evaluated, policy, now=release)
    assert not patch_is_revealed(evaluated, {"mode": "private"}, now=release)
    with pytest.raises(ValueError, match="timezone"):
        patch_reveal_at(evaluated.replace(tzinfo=None), policy)


def test_operator_rejects_invalid_policy_before_database_access(monkeypatch):
    monkeypatch.setattr(
        command, "db_connection", lambda: pytest.fail("database accessed")
    )
    with pytest.raises(ValueError, match="patch_visibility"):
        command.set_patch_visibility(
            str(uuid4()), {"mode": "public_after_reveal", "reveal_delay_s": -1}
        )


def test_operator_updates_only_policy_and_reports_missing_campaign(monkeypatch):
    calls = []
    found = [True]
    cid = str(uuid4())

    class Cursor:
        def execute(self, query, params):
            calls.append((query, params))

        def fetchone(self):
            return (cid,) if found[0] else None

    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()

    @contextmanager
    def connection():
        yield Connection()

    monkeypatch.setattr(command, "db_connection", connection)
    expected = {"mode": "public_after_reveal", "reveal_delay_s": 172800}
    assert (
        command.set_patch_visibility(cid, {"mode": "public_after_reveal"}) == expected
    )
    query, params = calls.pop()
    assert "manifest_hash" not in query and "customer_signoff" not in query
    assert params[0].adapted == expected and params[1] == cid
    found[0] = False
    with pytest.raises(ValueError, match="campaign not found"):
        command.set_patch_visibility(cid, {"mode": "private"})


def test_store_roundtrip_normalizes_policy_without_rehashing():
    original = build_manifest(**_manifest_kwargs())
    row = {
        **original.to_public_dict(),
        "id": original.campaign_id,
        "patch_visibility": '{"mode":"public_after_reveal","reveal_delay_s":3600}',
    }
    loaded = store._row_to_manifest(row)
    assert loaded.manifest_hash == original.manifest_hash
    assert loaded.patch_visibility == {
        "mode": "public_after_reveal",
        "reveal_delay_s": 3600,
    }
    row.pop("patch_visibility")
    assert store._row_to_manifest(row).patch_visibility == {"mode": "private"}


@pytest.mark.parametrize("mode,delay", [("private", None), ("public_after_reveal", 0)])
def test_operator_cli_wires_policy(monkeypatch, mode, delay):
    captured = []
    cid = str(uuid4())
    monkeypatch.setattr(
        command, "set_patch_visibility", lambda c, p: captured.append((c, p))
    )
    args = ["--campaign-id", cid, "--mode", mode]
    if delay is not None:
        args += ["--reveal-delay-s", str(delay)]
    assert command.main(args) == 0
    assert captured == [
        (
            cid,
            {"mode": mode, **({"reveal_delay_s": delay} if delay is not None else {})},
        )
    ]
