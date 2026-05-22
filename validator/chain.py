"""Everything that talks to Bittensor from the validator.

**Pure helpers** -- `parse_commitment_data`, `build_commitments`,
`build_competition_weights`, etc. These only need plain data structures
and are easy to unit test with a fake metagraph.

**RPC wrappers** -- `fetch_metagraph`, `fetch_revealed_commitments`,
`set_weights`. They add retries and logging around `bittensor` calls.
The library is imported lazily inside those paths so importing
`validator.*` in tests does not require `bittensor` to be installed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol

if TYPE_CHECKING:
    import bittensor as bt

from . import config as validator_config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CommitmentRecord:
    """One miner's most recent on-chain commitment, already parsed.

    On-chain format (encoded with `subtensor.set_reveal_commitment`):
        {"image": "registry/repo:tag", "digest": "sha256:<64-char hex>"}

    Optional payment fields (present when the miner paid the submission fee):
        "payment_tx":    extrinsic hash of the fee batch call (0x + 64 hex chars)
        "payment_block": block hash where that extrinsic was included

    `image` is a Docker image reference (registry/repo:tag or repo:tag).
    `digest` is the image manifest digest (sha256:...) that pins the exact
    image content regardless of tag mutations.
    """

    uid: int
    hotkey: str
    coldkey: str
    commit_block: int
    image: str
    digest: str
    raw: str  # original JSON string, kept for diagnostics
    payment_tx: str | None = None
    payment_block: str | None = None

    @property
    def has_payment(self) -> bool:
        return self.payment_tx is not None and self.payment_block is not None

    def as_eval_key(self) -> tuple[str, int]:
        return (self.hotkey, self.commit_block)


class _MetagraphLike(Protocol):
    """Structural type for a bittensor metagraph. Kept minimal so tests
    can pass a plain object with `hotkeys` attribute."""

    hotkeys: list[str]


# --------------------------------------------------------------------------- #
# Pure parsing helpers
# --------------------------------------------------------------------------- #


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*$")
_TAG_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE = re.compile(r"^0x[0-9a-f]{64}$")


def is_valid_docker_image(image: str) -> bool:
    """Validate a Docker image reference like ``registry:port/repo:tag``.

    The tag is the substring after the last ``:`` only when no ``/``
    follows that colon (so ``registry:5000/repo`` is a port, not a tag).
    """
    if not image:
        return False
    name = image
    last_colon = image.rfind(":")
    if last_colon > 0 and "/" not in image[last_colon:]:
        tag = image[last_colon + 1 :]
        name = image[:last_colon]
        if not tag or not _TAG_RE.match(tag):
            return False
    return bool(_NAME_RE.match(name))


def parse_commitment_data(raw: str) -> tuple[str, str] | None:
    """Parse a commitment payload into `(image, digest)` or None.

    Rejects anything that isn't a JSON object with non-empty `image` +
    `digest` strings. `image` must look like a Docker image reference
    (e.g. `docker.io/user/repo:tag`). `digest` must be a sha256 manifest
    digest (`sha256:<64 hex chars>`). Silent on failure -- caller logs.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    image = obj.get("image")
    digest = obj.get("digest")
    if not isinstance(image, str) or not image.strip():
        return None
    if not isinstance(digest, str) or not digest.strip():
        return None
    image = image.strip()
    digest = digest.strip()
    if not is_valid_docker_image(image):
        return None
    if not DIGEST_RE.match(digest):
        return None
    return image, digest


def extract_payment_fields(raw: str) -> tuple[str, str] | None:
    """Extract ``(payment_tx, payment_block)`` from a commitment JSON string.

    Both fields must be 0x-prefixed 64-char lowercase hex strings.
    Returns None when either is absent, the wrong format, or parsing fails.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    tx = obj.get("payment_tx")
    blk = obj.get("payment_block")
    if not isinstance(tx, str) or not isinstance(blk, str):
        return None
    tx = tx.strip().lower()
    blk = blk.strip().lower()
    if not _HEX64_RE.match(tx) or not _HEX64_RE.match(blk):
        return None
    return tx, blk


def build_commitments(
    metagraph: _MetagraphLike,
    revealed: dict[str, list[tuple[int, str]]],
) -> dict[int, CommitmentRecord]:
    """Fold per-hotkey revealed commitments into `{uid: CommitmentRecord}`.

    Args:
        metagraph: object with a `hotkeys` list (uid → hotkey ss58).
        revealed: the raw dict returned by
            `subtensor.get_all_revealed_commitments(netuid)`, shaped as
            `{hotkey: [(block, data_str), ...]}`. If a hotkey has multiple
            commitments, we take the one with the highest block.

    Skips:
        - hotkeys with no commitments
        - commitments that fail JSON parsing or don't have `image`+`digest`
    """
    out: dict[int, CommitmentRecord] = {}
    hotkeys = list(metagraph.hotkeys)
    coldkeys = list(getattr(metagraph, "coldkeys", []) or [])

    for uid, hotkey in enumerate(hotkeys):
        hotkey_str = str(hotkey)
        reveals = revealed.get(hotkey_str) or []
        if not reveals:
            continue
        block, raw = max(reveals, key=lambda p: p[0])
        parsed = parse_commitment_data(raw)
        if parsed is None:
            logger.debug(
                "UID %d (%s): commitment at block %d is not valid "
                "cacheon JSON -- skipping.",
                uid,
                hotkey_str[:16] + "...",
                block,
            )
            continue
        image, digest = parsed
        payment = extract_payment_fields(raw)
        coldkey_str = str(coldkeys[uid]) if uid < len(coldkeys) else ""
        out[uid] = CommitmentRecord(
            uid=uid,
            hotkey=hotkey_str,
            coldkey=coldkey_str,
            commit_block=int(block),
            image=image,
            digest=digest,
            raw=raw,
            payment_tx=payment[0] if payment else None,
            payment_block=payment[1] if payment else None,
        )

    return out


def build_competition_weights(
    n_uids: int,
    winner_uid: int,
    winner_score: float,
    runner_up_uid: int | None = None,
    *,
    burn_uid: int = validator_config.BURN_UID,
    winner_share: float = validator_config.WINNER_WEIGHT_SHARE,
    runner_up_share: float = validator_config.RUNNER_UP_WEIGHT_SHARE,
    score_target: float = validator_config.SCORE_EMISSION_TARGET,
) -> list[float]:
    """Score-scaled weight vector with winner, optional runner-up, and burn UID.

    ``comp_frac = min(1.0, winner_score / score_target)`` controls how much
    of total emission goes to the competition pool. The remainder goes to
    ``burn_uid``. Within the competition pool, ``winner_share`` and
    ``runner_up_share`` split the allocation.

    When there is no valid runner-up, the winner receives 100% of the
    competition pool. If ``burn_uid`` collides with the winner or runner-up,
    the burn fraction folds into the winner.
    """
    if winner_uid < 0:
        raise ValueError(f"winner_uid must be non-negative, got {winner_uid}")

    comp = min(1.0, winner_score / score_target) if score_target > 0 else 1.0
    burn = 1.0 - comp

    has_runner_up = runner_up_uid is not None and runner_up_uid != winner_uid
    if has_runner_up:
        w_winner = comp * winner_share
        w_runner = comp * runner_up_share
    else:
        w_winner = comp
        w_runner = 0.0
        runner_up_uid = None

    size = max(n_uids, winner_uid + 1, burn_uid + 1)
    if runner_up_uid is not None:
        size = max(size, runner_up_uid + 1)
    weights = [0.0] * size

    weights[winner_uid] = w_winner
    if runner_up_uid is not None:
        weights[runner_up_uid] = w_runner

    if burn > 0:
        if burn_uid != winner_uid and (
            runner_up_uid is None or burn_uid != runner_up_uid
        ):
            weights[burn_uid] = burn
        else:
            weights[winner_uid] += burn

    return weights


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #


class ChainError(RuntimeError):
    """Raised when a chain RPC fails after all retries."""


def _retry(
    fn: Callable[[], Any],
    *,
    label: str,
    attempts: int,
    delay_s: float,
) -> Any:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "%s failed (attempt %d/%d): %s",
                label,
                i + 1,
                attempts,
                exc,
            )
            if i < attempts - 1:
                time.sleep(delay_s)
    raise ChainError(f"{label} failed after {attempts} attempts: {last_exc}")


# --------------------------------------------------------------------------- #
# Live RPC wrappers — bittensor imported lazily inside the functions
# --------------------------------------------------------------------------- #


def fetch_metagraph(
    subtensor: bt.Subtensor,
    netuid: int,
    *,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> tuple[bt.metagraph, int, str | None]:
    """Fetch metagraph + current block + block hash.

    Block hash is best-effort — RPC can flake under load and we'd rather
    keep going with a None hash than crash the loop.
    """

    def _inner() -> tuple[bt.metagraph, int, str | None]:
        metagraph = subtensor.metagraph(netuid)
        current_block = int(subtensor.block)
        try:
            block_hash = subtensor.substrate.get_block_hash(current_block)
        except Exception as exc:
            logger.warning(
                "Block hash lookup failed: %s — continuing with block_hash=None .",
                exc,
            )
            block_hash = None
        return metagraph, current_block, block_hash

    return _retry(
        _inner,
        label="fetch_metagraph",
        attempts=attempts,
        delay_s=delay_s,
    )


def _decode_raw_commitment(raw: str | bytes, *, _depth: int = 0) -> str:
    """Normalize a raw on-chain commitment value to a plain JSON string.

    Handles formats observed in the wild, including double-hex encoding
    (SDK hex-encodes, then substrate hex-encodes again):

    1. Plain JSON string (e.g. ``'{"image": ...}'``).
    2. Hex-encoded string with ``0x`` prefix and optional SCALE compact
       length prefix (bittensor SDK stores this way in some versions).
    3. Raw bytes with a SCALE compact length prefix followed by UTF-8
       JSON (substrate library decodes to this in some versions, appears
       as ``'E\\x02{"image": ...}'``).
    4. Double-hex: outer 0x decode yields SCALE prefix + ``0x`` + inner
       hex of JSON. Recurses to unwrap.
    """
    if _depth > 3:
        return str(raw)

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    s = str(raw)

    if s.startswith("0x"):
        try:
            decoded = bytes.fromhex(s[2:])
        except ValueError:
            return s
        text = decoded.decode("utf-8", errors="replace")
        idx_brace = text.find("{")
        idx_0x = text.find("0x")
        if idx_0x >= 0 and (idx_brace < 0 or idx_0x < idx_brace):
            return _decode_raw_commitment(text[idx_0x:], _depth=_depth + 1)
        return text[idx_brace:] if idx_brace >= 0 else text

    idx = s.find("{")
    if idx > 0:
        return s[idx:]
    return s


def _fetch_commitments_raw_substrate(
    subtensor: bt.Subtensor,
    netuid: int,
) -> dict[str, list[tuple[int, str]]]:
    """Fallback: query the substrate storage map directly.

    Bypasses bittensor's hex decoder, which crashes on some commitment
    encodings. Returns the same shape as the SDK methods.
    """
    result = subtensor.substrate.query_map(
        module="Commitments",
        storage_function="RevealedCommitments",
        params=[netuid],
    )
    out: dict[str, list[tuple[int, str]]] = {}
    for key, value in result:
        hotkey = str(key)
        entries = []
        for entry in value or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                data_raw, block = entry[0], entry[1]
            else:
                continue
            entries.append((int(block), _decode_raw_commitment(data_raw)))
        if entries:
            out[hotkey] = entries
    return out


def fetch_revealed_commitments(
    subtensor: bt.Subtensor,
    netuid: int,
    *,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> dict[str, list[tuple[int, str]]]:
    """Return `{hotkey: [(block, data_str), ...]}` for the subnet.

    Tries the bittensor SDK first; falls back to a raw substrate query
    if the SDK chokes on hex decoding (observed with mixed commitment
    encodings on chain).
    """

    def _inner() -> dict[str, list[tuple[int, str]]]:
        for method_name in (
            "get_all_revealed_commitments",
            "get_revealed_commitments",
        ):
            fn = getattr(subtensor, method_name, None)
            if callable(fn):
                try:
                    return fn(netuid) or {}
                except ValueError as exc:
                    if "fromhex" in str(exc) or "hexadecimal" in str(exc):
                        logger.warning(
                            "SDK %s hit hex decode error, falling back "
                            "to raw substrate query: %s",
                            method_name,
                            exc,
                        )
                        return _fetch_commitments_raw_substrate(subtensor, netuid)
                    raise
        raise RuntimeError(
            "subtensor has no get_all_revealed_commitments / "
            "get_revealed_commitments method -- bittensor version mismatch?"
        )

    return _retry(
        _inner,
        label="fetch_revealed_commitments",
        attempts=attempts,
        delay_s=delay_s,
    )


def set_weights(
    subtensor: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
    *,
    version_key: int = validator_config.VERSION_KEY,
    attempts: int = 3,
    delay_s: float = 30.0,
) -> None:
    """Push a pre-built weight vector on-chain. Raises `ChainError` if every
    attempt is rejected -- the main loop should sleep and retry next cycle.

    `version_key` tags the weight vector with the validator's mechanism
    version; consensus only trust-weights validators that agree on it.
    """
    n_nonzero = sum(1 for w in weights if w > 0)
    logger.info(
        "Setting weights: %d non-zero uid(s) of %d total, version_key=%d",
        n_nonzero,
        len(weights),
        version_key,
    )

    last_reason: str | None = None
    for i in range(attempts):
        try:
            result = subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            if isinstance(result, (tuple, list)):
                ok = bool(result[0])
                last_reason = str(result[1]) if len(result) > 1 else None
            else:
                ok = bool(result)
                last_reason = None
            if ok:
                logger.info("Weights set on-chain")
                return
            logger.warning(
                "set_weights attempt %d/%d rejected: %s",
                i + 1,
                attempts,
                last_reason,
            )
        except Exception as exc:
            last_reason = str(exc)
            logger.error(
                "set_weights attempt %d/%d raised: %s",
                i + 1,
                attempts,
                exc,
            )
        if i < attempts - 1:
            time.sleep(delay_s)

    raise ChainError(f"set_weights failed after {attempts} attempts: {last_reason}")


def _call_args_dict(call_args: Any) -> dict[str, Any]:
    """Normalize call_args from dict or [{name, value}, ...] list."""
    if isinstance(call_args, dict):
        return call_args
    if isinstance(call_args, list):
        out: dict[str, Any] = {}
        for item in call_args:
            if isinstance(item, dict) and item.get("name") is not None:
                out[str(item["name"])] = item.get("value")
        return out
    return {}


def _normalize_call(call: Any) -> dict[str, Any]:
    if isinstance(call, dict):
        return call
    val = getattr(call, "value", None)
    return val if isinstance(val, dict) else {}


def _flatten_calls(top_call: dict[str, Any]) -> list[dict[str, Any]]:
    fn = (top_call.get("call_function") or "").lower()
    if fn in ("batch", "batch_all"):
        inner = _call_args_dict(top_call.get("call_args")).get("calls") or []
        return [_normalize_call(c) for c in inner]
    return [top_call]


def _decode_remark_bytes(raw: Any) -> bytes:
    if raw is None:
        return b""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        hex_str = raw[2:] if raw.startswith("0x") else raw
        try:
            return bytes.fromhex(hex_str)
        except ValueError:
            return raw.encode()
    return bytes(raw)


def _normalize_account_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("Id") or value.get("id") or "")
    return str(value)


def _event_payload(event: Any) -> dict[str, Any]:
    val = getattr(event, "value", None)
    if isinstance(val, dict):
        return val
    if isinstance(event, dict):
        return event
    return {}


def _event_attributes(attrs: Any) -> dict[str, Any]:
    if isinstance(attrs, dict):
        return attrs
    if isinstance(attrs, (list, tuple)) and len(attrs) >= 3:
        return {"from": attrs[0], "to": attrs[1], "amount": attrs[2]}
    return {}


def _payment_events_valid(
    events: list[Any],
    *,
    expected_coldkey: str,
    payment_address: str,
    min_fee_rao: int,
) -> tuple[bool, bool]:
    """Return (transfer_executed, batch_not_interrupted) from extrinsic events."""
    has_transfer = False
    batch_interrupted = False
    for event in events:
        payload = _event_payload(event)
        module_id = str(payload.get("module_id") or "")
        event_id = str(payload.get("event_id") or "")
        if module_id == "Utility" and event_id == "BatchInterrupted":
            batch_interrupted = True
        if module_id != "Balances" or event_id != "Transfer":
            continue
        attrs = _event_attributes(payload.get("attributes"))
        from_acct = _normalize_account_id(attrs.get("from"))
        to_acct = _normalize_account_id(
            attrs.get("to") or attrs.get("dest") or attrs.get("destination")
        )
        amount = int(attrs.get("amount") or attrs.get("value") or 0)
        if (
            from_acct == _normalize_account_id(expected_coldkey)
            and to_acct == _normalize_account_id(payment_address)
            and amount >= min_fee_rao
        ):
            has_transfer = True
    return has_transfer, not batch_interrupted


class PaymentVerifyOutcome(str, Enum):
    OK = "ok"
    REJECT = "reject"
    DEFER = "defer"


@dataclass(frozen=True)
class PaymentVerifyResult:
    outcome: PaymentVerifyOutcome
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is PaymentVerifyOutcome.OK


def _payment_calls_valid(
    calls: list[dict[str, Any]],
    *,
    payment_address: str,
    min_fee_rao: int,
    remark_needle: bytes,
) -> tuple[bool, bool]:
    has_transfer = False
    has_remark = False
    for call in calls:
        module = (call.get("call_module") or "").lower()
        fn = (call.get("call_function") or "").lower()
        args = _call_args_dict(call.get("call_args"))
        if module == "balances" and fn.startswith("transfer"):
            dest = args.get("dest") or ""
            if isinstance(dest, dict):
                dest = str(dest.get("Id") or "")
            if (
                str(dest) == payment_address
                and int(args.get("value") or 0) >= min_fee_rao
            ):
                has_transfer = True
        if module == "system" and fn == "remark":
            if _decode_remark_bytes(args.get("remark")) == remark_needle:
                has_remark = True
    return has_transfer, has_remark


def verify_submission_payment(
    subtensor: bt.Subtensor,
    payment_tx: str,
    payment_block: str,
    expected_coldkey: str,
    expected_hotkey: str,
    expected_digest: str,
    payment_address: str,
    min_fee_rao: int,
) -> PaymentVerifyResult:
    """Verify payment via ``substrate.retrieve_extrinsic_by_hash``.

    Checks signer (coldkey), executed Balances.Transfer event, no
    Utility.BatchInterrupted, and remark call ``cacheon:<hotkey>:<digest>``.

    RPC/lookup failures return ``DEFER`` so callers can retry next tick
    without tombstoning the miner.
    """
    try:
        receipt = subtensor.substrate.retrieve_extrinsic_by_hash(
            block_hash=payment_block,
            extrinsic_hash=payment_tx,
        )
        receipt.retrieve_extrinsic()
        ext = getattr(receipt, "_ExtrinsicReceipt__extrinsic", None)
    except Exception as exc:
        logger.warning("verify_submission_payment: lookup failed: %s", exc)
        return PaymentVerifyResult(
            outcome=PaymentVerifyOutcome.DEFER,
            reason=str(exc),
        )

    if ext is None:
        reason = f"tx {payment_tx[:18]}... not in block {payment_block[:18]}..."
        logger.warning("verify_submission_payment: %s", reason)
        return PaymentVerifyResult(outcome=PaymentVerifyOutcome.REJECT, reason=reason)

    if not receipt.is_success:
        reason = f"tx {payment_tx[:18]}... extrinsic failed"
        logger.warning("verify_submission_payment: %s", reason)
        return PaymentVerifyResult(outcome=PaymentVerifyOutcome.REJECT, reason=reason)

    body = ext.value if hasattr(ext, "value") else {}
    signer = _normalize_account_id(body.get("address"))
    if signer != _normalize_account_id(expected_coldkey):
        reason = f"signer {signer[:16]}... != expected coldkey"
        logger.warning("verify_submission_payment: %s", reason)
        return PaymentVerifyResult(outcome=PaymentVerifyOutcome.REJECT, reason=reason)

    transfer_executed, batch_ok = _payment_events_valid(
        receipt.triggered_events,
        expected_coldkey=expected_coldkey,
        payment_address=payment_address,
        min_fee_rao=min_fee_rao,
    )
    if not batch_ok:
        reason = f"tx {payment_tx[:18]}... batch interrupted"
        logger.warning("verify_submission_payment: %s", reason)
        return PaymentVerifyResult(outcome=PaymentVerifyOutcome.REJECT, reason=reason)
    if not transfer_executed:
        reason = f"tx {payment_tx[:18]}... missing transfer event"
        logger.warning("verify_submission_payment: %s", reason)
        return PaymentVerifyResult(outcome=PaymentVerifyOutcome.REJECT, reason=reason)

    calls = _flatten_calls(_normalize_call(body.get("call")))
    remark_needle = f"cacheon:{expected_hotkey}:{expected_digest}".encode()
    _, has_remark = _payment_calls_valid(
        calls,
        payment_address=payment_address,
        min_fee_rao=min_fee_rao,
        remark_needle=remark_needle,
    )
    if not has_remark:
        reason = f"tx {payment_tx[:18]}... remark missing in call data"
        logger.warning("verify_submission_payment: %s", reason)
        return PaymentVerifyResult(outcome=PaymentVerifyOutcome.REJECT, reason=reason)
    return PaymentVerifyResult(outcome=PaymentVerifyOutcome.OK)


def unique_hotkeys(
    commitments: Iterable[CommitmentRecord],
) -> set[str]:
    return {c.hotkey for c in commitments}


# --------------------------------------------------------------------------- #
# Startup preflight
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreflightResult:
    uid: int
    has_validator_permit: bool
    stake: float


class NotRegisteredError(RuntimeError):
    """Validator hotkey is not registered on this subnet — fatal at startup."""


def preflight_check(
    subtensor: bt.Subtensor,
    wallet: bt.Wallet,
    netuid: int,
) -> PreflightResult:
    """Fail fast on unregistered hotkeys; warn on missing validator permit.

    Raises `NotRegisteredError` if the wallet hotkey isn't registered on the
    subnet — without this, the loop would happily run then fail every tick at
    `set_weights` with a cryptic substrate error. Permit absence is logged as
    a warning, not fatal: a freshly-staked validator may get permit granted on
    the next epoch without needing a restart.
    """
    hotkey_ss58 = wallet.hotkey.ss58_address

    if not subtensor.is_hotkey_registered(netuid=netuid, hotkey_ss58=hotkey_ss58):
        raise NotRegisteredError(
            f"Hotkey {hotkey_ss58} is not registered on netuid {netuid}. "
            f"Register it with: btcli subnet register --netuid {netuid} "
            f"--wallet.name <name> --wallet.hotkey <hotkey>"
        )

    metagraph = subtensor.metagraph(netuid)
    uid = metagraph.hotkeys.index(hotkey_ss58)

    has_permit = False
    try:
        has_permit = bool(metagraph.validator_permit[uid])
    except (AttributeError, IndexError, TypeError):
        # older bittensor or odd metagraph shape — treat as unknown
        logger.debug("Could not read validator_permit[uid=%d]", uid)

    stake = 0.0
    try:
        stake = float(metagraph.S[uid])
    except (AttributeError, IndexError, TypeError):
        logger.debug("Could not read stake S[uid=%d]", uid)

    logger.info(
        "Preflight OK: uid=%d, stake=%.2f, validator_permit=%s",
        uid,
        stake,
        has_permit,
    )

    if not has_permit:
        logger.warning(
            "Validator permit not granted for uid=%d — weights will still be "
            "emitted, but may not count toward consensus until the subnet "
            "grants permit (typically next epoch, pending stake threshold).",
            uid,
        )

    return PreflightResult(uid=uid, has_validator_permit=has_permit, stake=stake)
