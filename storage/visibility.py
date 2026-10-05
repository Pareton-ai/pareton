"""Publication timing for finalized patch evaluations under campaign policy."""

from datetime import datetime, timedelta, timezone

from campaign.visibility import validate_patch_visibility


def patch_reveal_at(
    evaluated_at: datetime | None, policy: dict | None
) -> datetime | None:
    policy = validate_patch_visibility(policy)
    if policy["mode"] == "private" or evaluated_at is None:
        return None
    if evaluated_at.tzinfo is None:
        raise ValueError("evaluation timestamp must include a timezone")
    return evaluated_at.astimezone(timezone.utc) + timedelta(
        seconds=policy["reveal_delay_s"]
    )


def patch_is_revealed(
    evaluated_at: datetime | None, policy: dict | None, *, now: datetime | None = None
) -> bool:
    release = patch_reveal_at(evaluated_at, policy)
    return release is not None and (now or datetime.now(timezone.utc)) >= release
