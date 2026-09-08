"""Publication timing for finalized patch evaluations."""

from datetime import datetime, timedelta, timezone

import config

# Retried infra failures end in bench_queued; live round results are not events.
PATCH_TERMINAL_STATES = (
    "scored",
    "disqualified",
    "rejected",
    "rejected_duplicate",
    "infra_failed",
)


def patch_reveal_at(evaluated_at: datetime | None) -> datetime | None:
    if evaluated_at is None:
        return None
    if evaluated_at.tzinfo is None:
        raise ValueError("evaluation timestamp must include a timezone")
    return evaluated_at.astimezone(timezone.utc) + timedelta(
        seconds=config.PATCH_REVEAL_DELAY_S
    )


def patch_is_revealed(
    evaluated_at: datetime | None, *, now: datetime | None = None
) -> bool:
    release = patch_reveal_at(evaluated_at)
    return release is not None and (now or datetime.now(timezone.utc)) >= release
