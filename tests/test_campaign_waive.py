"""Offline tests for the campaign hotkey waiver CLI."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from campaign import waive
from round.store import CampaignHotkeyWaiver


def test_cli_resolves_uid_and_passes_audited_fields(monkeypatch, capsys):
    seen: dict = {}
    monkeypatch.setattr(waive, "_resolve_uid", lambda *_a, **_k: "5Bob")

    def _apply(campaign_id, hotkey, **kwargs):
        seen.update(campaign_id=campaign_id, hotkey=hotkey, **kwargs)
        return CampaignHotkeyWaiver(
            campaign_id=campaign_id,
            hotkey=hotkey,
            created=True,
        )

    monkeypatch.setattr(waive, "waive_campaign_hotkey", _apply)
    rc = waive.main(
        [
            "--campaign-id",
            "11111111-1111-4111-8111-111111111111",
            "--uid",
            "11",
            "--reason",
            "temporary test ended",
            "--evidence-ref",
            "PAR-123",
            "--operator",
            "ops@example.com",
        ]
    )

    assert rc == 0
    assert seen["hotkey"] == "5Bob"
    assert seen["reason"] == "temporary test ended"
    assert seen["evidence_ref"] == "PAR-123"
    assert seen["waived_by"] == "ops@example.com"
    output = capsys.readouterr().out
    assert "campaign waiver: created" in output
    assert "historical verdicts changed: 0" in output
