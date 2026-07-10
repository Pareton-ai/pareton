"""Gate a: identity / campaign eligibility."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from campaign.models import CampaignManifest
from gate.types import GateResult


def check_identity(
    *,
    hotkey: str,
    registered_hotkeys: Iterable[str],
    campaign: CampaignManifest,
    baseline_commit: str,
    now: datetime | None = None,
) -> GateResult:
    """Validate miner eligibility and campaign window / baseline pin."""
    registered = set(registered_hotkeys)
    if hotkey not in registered:
        return GateResult.reject(
            "hotkey not registered on subnet",
            hotkey=hotkey,
        )
    if campaign.status != "open":
        return GateResult.reject(
            f"campaign status is {campaign.status}, expected open",
            campaign_id=str(campaign.campaign_id),
        )

    ts = now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    opens = campaign.window_opens_at
    closes = campaign.window_closes_at
    if opens.tzinfo is None:
        opens = opens.replace(tzinfo=timezone.utc)
    if closes.tzinfo is None:
        closes = closes.replace(tzinfo=timezone.utc)
    if ts < opens or ts > closes:
        return GateResult.reject(
            "outside campaign window",
            now=ts.isoformat(),
            opens_at=opens.isoformat(),
            closes_at=closes.isoformat(),
        )

    if baseline_commit.lower() != campaign.baseline_commit.lower():
        return GateResult.reject(
            "baseline_commit mismatch",
            committed=baseline_commit.lower(),
            expected=campaign.baseline_commit.lower(),
        )

    return GateResult.success(
        "identity_ok",
        hotkey=hotkey,
        campaign_id=str(campaign.campaign_id),
    )
