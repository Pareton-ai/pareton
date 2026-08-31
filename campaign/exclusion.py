"""Manual campaign hotkey exclusion actions."""

from __future__ import annotations

from typing import Any
from uuid import UUID

ACTION_DISQUALIFIED = "disqualified"
ACTION_WAIVED = "waived"


def latest_campaign_hotkey_action(
    cur: Any, campaign_id: UUID | str, hotkey: str
) -> str | None:
    """Return the latest append-only manual action for a campaign hotkey."""
    cur.execute(
        """
        SELECT COALESCE(e.detail ->> 'action', 'disqualified') AS action
        FROM submissions prior
        JOIN submission_events e ON e.submission_id = prior.id
        WHERE prior.campaign_id = %s AND prior.hotkey = %s
          AND e.state = 'disqualified'
          AND e.detail ->> 'source' = 'manual'
          AND e.detail ->> 'scope' = 'campaign'
        ORDER BY e.created_at DESC, e.id DESC
        LIMIT 1
        """,
        (str(campaign_id), hotkey),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return str(row["action"] if isinstance(row, dict) else row[0])
