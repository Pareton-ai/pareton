"""Campaign patch disclosure terms, deliberately outside the manifest hash."""

from typing import Any

DEFAULT_REVEAL_DELAY_S = 172800
MAX_REVEAL_DELAY_S = 2147483647


def validate_patch_visibility(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {"mode": "private"}
    if not isinstance(value, dict):
        raise ValueError("patch_visibility must be an object")  # noqa: TRY004
    mode = value.get("mode")
    if mode not in ("private", "public_after_reveal"):
        raise ValueError("patch_visibility.mode must be private or public_after_reveal")
    allowed = {"mode"} if mode == "private" else {"mode", "reveal_delay_s"}
    if set(value) - allowed:
        raise ValueError("patch_visibility has unknown or inapplicable fields")
    if mode == "private":
        return {"mode": mode}
    delay = value.get("reveal_delay_s", DEFAULT_REVEAL_DELAY_S)
    if type(delay) is not int or not 0 <= delay <= MAX_REVEAL_DELAY_S:
        raise ValueError(
            f"patch_visibility.reveal_delay_s must be an integer from 0 to {MAX_REVEAL_DELAY_S}"
        )
    return {"mode": mode, "reveal_delay_s": delay}
