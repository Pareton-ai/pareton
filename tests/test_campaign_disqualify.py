"""Offline tests for the campaign hotkey disqualification CLI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from campaign import disqualify
from campaign.store import _campaign_hotkey_is_disqualified
from round.store import CampaignHotkeyDisqualification


class FakeMeta:
    def __init__(self):
        self.hotkeys = ["5Alice", "5Bob"]

    def by_hotkey(self, hotkey: str):
        return SimpleNamespace(uid={"5Alice": 7, "5Bob": 11}[hotkey])


class FakeCursor:
    def __init__(self, row):
        self.row = row

    def execute(self, query, params):
        assert "ORDER BY e.created_at DESC, e.id DESC" in query
        assert params == ("campaign-id", "5Bob")

    def fetchone(self):
        return self.row


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (None, False),
        (("disqualified",), True),
        ({"action": "waived"}, False),
        ({"action": "unknown"}, True),
    ],
)
def test_latest_manual_action_controls_eligibility(row, expected):
    assert (
        _campaign_hotkey_is_disqualified(FakeCursor(row), "campaign-id", "5Bob")
        is expected
    )


def test_uid_is_resolved_from_by_hotkey_not_list_position():
    assert disqualify.hotkey_for_uid(FakeMeta(), 11) == "5Bob"


def test_unknown_uid_is_rejected():
    with pytest.raises(ValueError, match="resolved to 0 hotkeys"):
        disqualify.hotkey_for_uid(FakeMeta(), 3)


def test_cli_resolves_uid_and_passes_audited_fields(monkeypatch, capsys):
    seen: dict = {}
    monkeypatch.setattr(disqualify, "_resolve_uid", lambda *_a, **_k: "5Bob")

    def _apply(campaign_id, hotkey, **kwargs):
        seen.update(campaign_id=campaign_id, hotkey=hotkey, **kwargs)
        return CampaignHotkeyDisqualification(
            campaign_id=campaign_id,
            hotkey=hotkey,
            created=True,
            submissions_disqualified=2,
            pending_jobs_stopped=1,
            leader_vacated=True,
            replacement_submission_id="22222222-2222-4222-8222-222222222222",
            replacement_hotkey="5Alice",
        )

    monkeypatch.setattr(disqualify, "disqualify_campaign_hotkey", _apply)
    rc = disqualify.main(
        [
            "--campaign-id",
            "11111111-1111-4111-8111-111111111111",
            "--uid",
            "11",
            "--reason",
            "policy violation",
            "--evidence-ref",
            "PAR-123",
            "--operator",
            "ops@example.com",
        ]
    )
    assert rc == 0
    assert seen["hotkey"] == "5Bob"
    assert seen["reason"] == "policy violation"
    assert seen["evidence_ref"] == "PAR-123"
    assert seen["disqualified_by"] == "ops@example.com"
    output = capsys.readouterr().out
    assert "leader vacated: yes" in output
    assert "replacement leader: 5Alice" in output
