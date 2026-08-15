"""Gate a: identity / campaign eligibility."""

from __future__ import annotations

from typing import Iterable

from campaign.models import CampaignManifest
from gate.types import GateResult


def check_identity(
    *,
    hotkey: str,
    registered_hotkeys: Iterable[str],
    campaign: CampaignManifest,
    baseline_commit: str,
) -> GateResult:
    """Validate miner eligibility and campaign status / baseline pin.

    Campaigns have no submission window: ``status`` is the only intake switch,
    so a campaign accepts patches until an operator closes it.
    """
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
