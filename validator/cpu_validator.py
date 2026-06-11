"""CPU-side always-on validator: chain scan, challenger selection, weight setting.

Runs continuously on a lightweight VPS. Does NOT evaluate miners; that
happens on an ephemeral GPU pod reading ``eval_job.json`` from S3.

Loop (every CACHEON_POLL_INTERVAL_S, default 600s):
    1. Download latest state from Hippius S3
    2. Chain scan: fetch metagraph + commitments
    3. If winner's hotkey deregistered, promote runner-up or clear
    4. Select new challengers not yet evaluated
    5. If challengers found: write eval_job.json, upload to S3, run GPU eval
    6. Re-scan metagraph; re-run deregistration guard on post-GPU state
    7. If new GPU eval results or weights stale: set_weights (end of tick)
    7. Sleep

Usage:
    python -m validator.cpu_validator
    python -m validator.cpu_validator --network test --netuid 470 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path

from . import config as validator_config
from .chain import (
    ChainError,
    CommitmentRecord,
    NotRegisteredError,
    PaymentVerifyOutcome,
    build_commitments,
    build_competition_weights,
    compute_emission_pool_frac,
    fetch_metagraph,
    fetch_revealed_commitments,
    preflight_check,
    set_weights,
    verify_submission_payment,
)
from .challengers import select_challengers
from .eval_schema import ChallengerInfo, EvalJob
from .state import ValidatorState, WinnerRecord

logger = logging.getLogger(__name__)

_last_cpu_log_block: int | None = None

WEIGHTS_REFRESH_BLOCKS: int = int(
    os.environ.get("CACHEON_WEIGHTS_REFRESH_BLOCKS", "360")
)
"""Re-affirm weights at least once per tempo (~72 min at 12s/block) so the
validator stays active in consensus even when no new GPU eval results arrive."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _set_log_file(state_dir: str, block: int | None, *, level: int) -> None:
    """Point the root logger at ``logs/cpu_idle_{ts}.log`` or ``logs/cpu_{block}_{ts}.log``."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()

    logs_dir = Path(state_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if block is None:
        log_path = logs_dir / f"cpu_idle_{ts}.log"
    else:
        log_path = logs_dir / f"cpu_{block}_{ts}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(fh)
    logger.info("Logging to %s", log_path)


def _configure_logging(verbose: bool, state_dir: str) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )

    _set_log_file(state_dir, None, level=level)

    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and name.startswith("validator"):
            lg.setLevel(logging.NOTSET)
    logging.getLogger("validator").setLevel(logging.NOTSET)
    logger.setLevel(level)

    for noisy in (
        "bittensor",
        "websockets",
        "websockets.client",
        "btdecode",
        "substrateinterface",
        "urllib3",
        "async_substrate_interface",
        "paramiko",
        "paramiko.transport",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _rotate_log_for_block(state_dir: str, block: int) -> None:
    """Start a fresh CPU log file keyed to the eval round block."""
    global _last_cpu_log_block
    if _last_cpu_log_block == block:
        return
    _last_cpu_log_block = block
    _set_log_file(state_dir, block, level=logging.getLogger().level)


def _needs_weight_set(state: ValidatorState, current_block: int) -> str | None:
    """Return a reason string if weights should be (re-)set, else None."""
    if state.winner is None:
        return None
    if state.last_weights_set_block == 0:
        return "first weight set"
    if state.evaluations:
        max_eval_block = max(e.evaluation_block for e in state.evaluations.values())
        if max_eval_block > state.last_weights_set_block:
            return f"new evals (latest eval block {max_eval_block})"
    stale = current_block - state.last_weights_set_block
    if stale > WEIGHTS_REFRESH_BLOCKS:
        return f"stale ({stale} blocks since last set)"
    return None


def _reload_state(state: ValidatorState, state_dir: str) -> None:
    """Reload state from disk into the existing object (GPU may have updated it)."""
    fresh = ValidatorState.load(state_dir)
    state.winner = fresh.winner
    state.runner_up_record = fresh.runner_up_record
    state.evaluations = fresh.evaluations
    state.precheck_failures = fresh.precheck_failures
    state.last_scan_block = fresh.last_scan_block
    state.last_weights_set_block = fresh.last_weights_set_block


_CPU_UPLOAD_PRE_EVAL = ["eval_job.json", "eval_progress.json"]
"""Mid-tick upload before GPU eval. Excludes ``logs/`` (uploaded after tick)."""

_CPU_UPLOAD_STATE = ["state.json"]
"""Post-weight-set upload so ``last_weights_set_block`` persists to S3."""

_CPU_LOGS_UPLOAD = ["logs/"]

# Never download validator logs onto the running CPU process (open FileHandler).
_DOWNLOAD_SKIP_LOGS: tuple[str, ...] = ("logs/",)

# Remote GPU pod logs only; never overwrite local cpu_* files from S3.
_DOWNLOAD_SKIP_CPU_LOGS: tuple[str, ...] = ("logs/cpu_",)


def _flush_log_handlers() -> None:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()


def _try_upload(state_dir: str) -> None:
    if validator_config.SKIP_S3:
        return
    try:
        from .sync import upload

        upload(state_dir, only=_CPU_UPLOAD_PRE_EVAL)
    except Exception as exc:
        logger.error("S3 upload failed: %s", exc)


def _try_upload_state(state_dir: str) -> None:
    if validator_config.SKIP_S3:
        return
    try:
        from .sync import upload

        upload(state_dir, only=_CPU_UPLOAD_STATE)
    except Exception as exc:
        logger.error("S3 state upload failed: %s", exc)


def _try_upload_logs(state_dir: str) -> None:
    """Push complete validator logs to S3 after a tick finishes."""
    if validator_config.SKIP_S3:
        return
    _flush_log_handlers()
    try:
        from .sync import upload

        upload(state_dir, only=_CPU_LOGS_UPLOAD)
    except Exception as exc:
        logger.error("S3 log upload failed: %s", exc)


def _clean_stale_eval_job(state: ValidatorState, state_dir: str) -> bool:
    """Remove ``eval_job.json`` if every challenger in it is already known.

    Returns True if the file was deleted."""
    from .eval_schema import EVAL_JOB_FILE, EvalJob

    path = Path(state_dir) / EVAL_JOB_FILE
    if not path.exists():
        return False
    job = EvalJob.load(state_dir)
    if job is None:
        return False
    if all(state.is_known(c.hotkey, c.commit_block) for c in job.challengers):
        try:
            path.unlink()
            logger.info(
                "Removed stale eval_job.json (%d challenger(s) all known)",
                len(job.challengers),
            )
        except OSError:
            return False
        try:
            from .sync import delete_remote_keys

            delete_remote_keys([EVAL_JOB_FILE])
        except Exception:
            logger.debug("Failed to delete eval_job.json from S3", exc_info=True)
        return True
    return False


def _gate_unpaid_commitments(
    subtensor,
    state: ValidatorState,
    commitments: dict[int, CommitmentRecord],
) -> tuple[dict[int, CommitmentRecord], list[tuple[CommitmentRecord, str]]]:
    """Filter out commitments that don't have a valid submission payment.

    Active when SUBMISSION_FEE_RAO > 0. Returns (valid_commitments, rejected_list).

    Skips RPC for ``(hotkey, commit_block)`` pairs already evaluated or
    tombstoned. Transient RPC failures defer to the next tick without
    recording a precheck failure.
    """
    payment_address = validator_config.PAYMENT_ADDRESS
    fee_rao = validator_config.SUBMISSION_FEE_RAO

    if fee_rao <= 0:
        return commitments, []

    valid: dict[int, CommitmentRecord] = {}
    rejected: list[tuple[CommitmentRecord, str]] = []
    deferred = 0

    for uid, com in commitments.items():
        if state.has_evaluation(com.hotkey, com.commit_block):
            valid[uid] = com
            continue
        if state.has_precheck_failure(com.hotkey, com.commit_block):
            continue

        if not com.has_payment:
            reason = "missing payment: no payment_tx/payment_block in commitment"
            logger.info(
                "UID %d (%s) rejected: %s",
                com.uid,
                com.hotkey[:16] + "...",
                reason,
            )
            rejected.append((com, reason))
            continue

        if not com.coldkey:
            reason = "missing coldkey: cannot verify payment signer"
            logger.warning(
                "UID %d (%s) rejected: %s",
                com.uid,
                com.hotkey[:16] + "...",
                reason,
            )
            rejected.append((com, reason))
            continue

        result = verify_submission_payment(
            subtensor,
            payment_tx=com.payment_tx,
            payment_block=com.payment_block,
            expected_coldkey=com.coldkey,
            expected_hotkey=com.hotkey,
            expected_digest=com.digest,
            payment_address=payment_address,
            min_fee_rao=fee_rao,
        )
        if result.outcome is PaymentVerifyOutcome.OK:
            valid[uid] = com
        elif result.outcome is PaymentVerifyOutcome.DEFER:
            deferred += 1
            logger.warning(
                "UID %d (%s) payment verify deferred: %s",
                com.uid,
                com.hotkey[:16] + "...",
                result.reason or "lookup failed",
            )
        else:
            reason = result.reason or (
                f"payment verification failed for tx {com.payment_tx[:18]}..."
            )
            logger.warning(
                "UID %d (%s) rejected: %s",
                com.uid,
                com.hotkey[:16] + "...",
                reason,
            )
            rejected.append((com, reason))

    if rejected or deferred:
        logger.info(
            "Payment gate: %d paid, %d rejected, %d deferred (address=%s, fee=%d RAO)",
            len(valid),
            len(rejected),
            deferred,
            payment_address[:16] + "...",
            fee_rao,
        )

    return valid, rejected


def _hotkey_is_registered(metagraph, uid: int, hotkey: str) -> bool:
    """True if `uid` is in range and the on-chain hotkey matches."""
    if uid < 0 or uid >= len(metagraph.hotkeys):
        return False
    return metagraph.hotkeys[uid] == hotkey


def _resolve_runner_up_uid(state: ValidatorState, metagraph) -> int | None:
    """Return the runner-up's UID if one exists and is still registered."""
    ru = state.runner_up
    if ru is None:
        return None
    if _hotkey_is_registered(metagraph, ru.uid, ru.hotkey):
        return ru.uid
    return None


def _apply_deregistration_guard(
    state: ValidatorState,
    metagraph,
    current_block: int,
) -> bool:
    """Promote runner-up or clear winner if the leader UID no longer matches chain.

    Returns True when ``state.winner`` or ``runner_up_record`` changed."""
    if state.winner is None:
        return False
    if _hotkey_is_registered(metagraph, state.winner.uid, state.winner.hotkey):
        return False

    ru = state.runner_up
    if ru is not None and _hotkey_is_registered(metagraph, ru.uid, ru.hotkey):
        logger.warning(
            "Winner UID %d deregistered (%s). Promoting runner-up UID %d.",
            state.winner.uid,
            state.winner.hotkey[:16],
            ru.uid,
        )
        state.winner = WinnerRecord.from_evaluation(ru, won_at_block=current_block)
        state.runner_up_record = None  # promoted; no runner-up until next eval
        state.last_weights_set_block = 0  # force immediate weight update
        return True

    reason = "runner-up also gone" if ru is not None else "no runner-up"
    logger.warning(
        "Winner UID %d deregistered (%s). Clearing winner (%s).",
        state.winner.uid,
        state.winner.hotkey[:16],
        reason,
    )
    state.winner = None
    state.runner_up_record = None
    return True


def _try_set_weights(
    *,
    state: ValidatorState,
    metagraph,
    current_block: int,
    subtensor,
    wallet,
    netuid: int,
    state_dir: str,
    dry_run: bool,
    version_key: int,
) -> bool:
    """Set on-chain weights when state is current. Returns True if weights were set."""
    reason = _needs_weight_set(state, current_block)
    if not reason or state.winner is None:
        return False

    runner_up_uid = _resolve_runner_up_uid(state, metagraph)
    logger.info(
        "⚖️  Setting weights: winner=UID %d (score=%.4f), runner_up=%s, reason=%s",
        state.winner.uid,
        state.winner.score,
        runner_up_uid,
        reason,
    )

    w_dense = build_competition_weights(
        n_uids=len(metagraph.hotkeys),
        winner_uid=state.winner.uid,
        runner_up_uid=runner_up_uid,
        current_block=current_block,
    )
    burn_uid = validator_config.BURN_UID
    emission_frac = compute_emission_pool_frac(current_block)
    override_note = (
        " (CACHEON_EMISSION_FRAC_OVERRIDE)"
        if validator_config.EMISSION_FRAC_OVERRIDE is not None
        else ""
    )
    logger.info(
        "⚖️  weight vector: emission_frac=%.4f%s, winner=%d (%.4f),"
        " runner_up=%s (%.4f), burn_uid=%d (%.4f), n_uids=%d",
        emission_frac,
        override_note,
        state.winner.uid,
        w_dense[state.winner.uid] if state.winner.uid < len(w_dense) else 0.0,
        runner_up_uid,
        w_dense[runner_up_uid]
        if runner_up_uid is not None and runner_up_uid < len(w_dense)
        else 0.0,
        burn_uid,
        w_dense[burn_uid] if burn_uid < len(w_dense) else 0.0,
        len(w_dense),
    )

    uid_list = [u for u, wt in enumerate(w_dense) if wt > 0]
    w = [wt for wt in w_dense if wt > 0]

    if dry_run:
        logger.info("🧪 [DRY-RUN] would set_weights (see weight vector above)")
        state.last_weights_set_block = current_block
        state.save(state_dir)
        return True

    try:
        set_weights(
            subtensor,
            wallet,
            netuid,
            uids=uid_list,
            weights=w,
            version_key=version_key,
        )
        state.last_weights_set_block = current_block
        state.save(state_dir)
        return True
    except ChainError as exc:
        logger.error("set_weights failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Tick
# --------------------------------------------------------------------------- #


def run_tick(
    *,
    subtensor,
    wallet,
    state: ValidatorState,
    netuid: int,
    state_dir: str,
    dry_run: bool = False,
    version_key: int = validator_config.VERSION_KEY,
) -> dict:
    """One iteration of the CPU validator loop. Returns a summary dict."""

    from .eval_progress import purge_old_logs

    purge_old_logs(state_dir)

    # S3 download
    try:
        from .sync import download

        download(state_dir, skip_prefixes=_DOWNLOAD_SKIP_LOGS)
    except Exception as exc:
        logger.error("S3 download failed: %s -- using local state", exc)

    purge_old_logs(state_dir, remote=False)

    _reload_state(state, state_dir)
    _clean_stale_eval_job(state, state_dir)

    winner_desc = (
        f"UID {state.winner.uid} score={state.winner.score:.4f}"
        if state.winner
        else "none"
    )
    logger.info(
        "📋 State: winner=%s | %d eval(s) | last_weights_block=%d",
        winner_desc,
        len(state.evaluations),
        state.last_weights_set_block,
    )

    # Chain scan
    metagraph, current_block, block_hash = fetch_metagraph(subtensor, netuid)
    revealed = fetch_revealed_commitments(subtensor, netuid)
    commitments = build_commitments(metagraph, revealed)
    state.last_scan_block = current_block

    logger.info(
        "Scan block %d: %d hotkey(s), %d commitment(s)",
        current_block,
        len(metagraph.hotkeys),
        len(commitments),
    )

    # Payment gate: drop any commitment that hasn't paid the submission fee.
    commitments, payment_rejected = _gate_unpaid_commitments(
        subtensor, state, commitments
    )
    for com, rej_reason in payment_rejected:
        state.record_precheck_failure(com.hotkey, com.commit_block, rej_reason)

    if _apply_deregistration_guard(state, metagraph, current_block):
        state.save(state_dir)

    dirty = False

    # Challenger selection
    challenger_set = select_challengers(state, commitments.values())
    for com, rej_reason in challenger_set.newly_rejected:
        state.record_precheck_failure(com.hotkey, com.commit_block, rej_reason)

    logger.info(
        "⚔️  Challengers: %d new, %d rejected, %d deferred, %d known",
        len(challenger_set.challengers),
        len(challenger_set.newly_rejected),
        len(challenger_set.deferred),
        len(challenger_set.already_known),
    )

    n_challengers = len(challenger_set.challengers)
    if challenger_set.challengers:
        if block_hash is None:
            logger.warning(
                "block_hash is None; cannot write eval_job (prompts need "
                "deterministic seeding). Will retry next tick."
            )
        else:
            for com in challenger_set.challengers:
                logger.info(
                    "  New: UID %d  %s  %s", com.uid, com.hotkey[:16], com.image
                )
            leader_info = None
            if state.winner is not None:
                leader_info = ChallengerInfo(
                    uid=state.winner.uid,
                    hotkey=state.winner.hotkey,
                    commit_block=state.winner.commit_block,
                    image=state.winner.image,
                    digest=state.winner.digest,
                )
                logger.info(
                    "  Leader (re-eval): UID %d  %s  %s",
                    state.winner.uid,
                    state.winner.hotkey[:16],
                    state.winner.image,
                )

            ru = state.runner_up_record if state.winner is not None else None
            ru_info = None
            if ru is not None:
                ru_info = ChallengerInfo(
                    uid=ru.uid,
                    hotkey=ru.hotkey,
                    commit_block=ru.commit_block,
                    image=ru.image,
                    digest=ru.digest,
                )
                logger.info(
                    "  Runner-up (re-eval): UID %d  %s  %s",
                    ru.uid,
                    ru.hotkey[:16],
                    ru.image,
                )

            eval_job = EvalJob(
                block=current_block,
                block_hash=block_hash,
                challengers=[
                    ChallengerInfo(
                        uid=c.uid,
                        hotkey=c.hotkey,
                        commit_block=c.commit_block,
                        image=c.image,
                        digest=c.digest,
                    )
                    for c in challenger_set.challengers
                ],
                created_at=time.time(),
                leader=leader_info,
                runner_up=ru_info,
            )
            from .eval_progress import update_progress

            update_progress(
                state_dir,
                phase="challengers_found",
                round_block=current_block,
                challengers=[
                    {"uid": c.uid, "hotkey": c.hotkey, "image": c.image}
                    for c in challenger_set.challengers
                ],
                leader=(
                    {
                        "uid": state.winner.uid,
                        "hotkey": state.winner.hotkey,
                        "image": state.winner.image,
                    }
                    if state.winner is not None
                    else None
                ),
                runner_up=(
                    {
                        "uid": ru.uid,
                        "hotkey": ru.hotkey,
                        "image": ru.image,
                    }
                    if ru is not None
                    else None
                ),
            )
            _rotate_log_for_block(state_dir, current_block)
            eval_job.save(state_dir)
            state.save(state_dir)
            dirty = True

    if not challenger_set.challengers:
        state.save(state_dir)

    if dirty:
        _try_upload(state_dir)

    if challenger_set.challengers and block_hash is not None:
        logger.info(
            "📤 %d challenger(s) ready for GPU eval (eval_job.json uploaded)",
            n_challengers,
        )

        from .gpu_orchestrator import gpu_eval_configured, run_gpu_eval

        if gpu_eval_configured():
            success = run_gpu_eval(state_dir, eval_job)
            clear_stale_progress = not success
            try:
                from .sync import download

                if success:
                    download(state_dir, skip_prefixes=_DOWNLOAD_SKIP_CPU_LOGS)
                    _reload_state(state, state_dir)
                else:
                    logger.info(
                        "GPU eval failed; downloading remote artifacts from S3 "
                        "(container_logs/, gpu logs only)"
                    )
                    download(state_dir, only=["container_logs/"])
                    download(
                        state_dir,
                        only=["logs/"],
                        skip_prefixes=_DOWNLOAD_SKIP_CPU_LOGS,
                    )
            except Exception as exc:
                logger.error("Post-eval S3 download failed: %s", exc)
                clear_stale_progress = True
            if clear_stale_progress:
                from .eval_progress import clear_progress

                clear_progress(state_dir)

    metagraph, weights_block, _ = fetch_metagraph(subtensor, netuid)
    if _apply_deregistration_guard(state, metagraph, weights_block):
        state.save(state_dir)

    weights_set = _try_set_weights(
        state=state,
        metagraph=metagraph,
        current_block=weights_block,
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        state_dir=state_dir,
        dry_run=dry_run,
        version_key=version_key,
    )
    if weights_set:
        _try_upload_state(state_dir)

    return {
        "block": current_block,
        "commitments": len(commitments),
        "challengers": n_challengers,
        "weights_set": weights_set,
        "winner_uid": state.winner.uid if state.winner else None,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cacheon CPU validator (always-on, no GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--network",
        default=validator_config.SUBTENSOR_NETWORK,
        help="Bittensor network: finney | test | ws://...",
    )
    p.add_argument("--netuid", type=int, default=validator_config.NETUID)
    p.add_argument("--wallet-name", default=validator_config.WALLET_NAME)
    p.add_argument("--wallet-hotkey", default=validator_config.WALLET_HOTKEY)
    p.add_argument(
        "--poll-interval",
        type=int,
        default=validator_config.POLL_INTERVAL_S,
        help="Seconds between chain scans.",
    )
    p.add_argument("--state-dir", default=str(validator_config.STATE_DIR))
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=validator_config.DRY_RUN,
        help="Skip set_weights on chain.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    def _handle_sigterm(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        import bittensor as bt
    except ImportError:
        logging.basicConfig(level=logging.ERROR)
        logging.error(
            "bittensor is not installed. "
            "pip install 'bittensor>=10' before running the CPU validator."
        )
        return 2

    _configure_logging(args.verbose, args.state_dir)

    logger.info(
        "🕗 CPU validator starting: network=%s netuid=%d wallet=%s/%s poll=%ds",
        args.network,
        args.netuid,
        args.wallet_name,
        args.wallet_hotkey,
        args.poll_interval,
    )

    subtensor = bt.Subtensor(
        network="archive" if args.network == "finney" else args.network
    )
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    try:
        preflight_check(subtensor, wallet, netuid=args.netuid)
    except NotRegisteredError as exc:
        logger.error("%s", exc)
        return 4

    state = ValidatorState.load(args.state_dir)
    winner_desc = (
        f"winner=UID {state.winner.uid} (score={state.winner.score:.4f})"
        if state.winner
        else "no winner yet"
    )
    logger.info(
        "Loaded state: %s | %d eval(s) | last_weights_block=%d",
        winner_desc,
        len(state.evaluations),
        state.last_weights_set_block,
    )

    try:
        while True:
            tick_start = time.time()
            try:
                summary = run_tick(
                    subtensor=subtensor,
                    wallet=wallet,
                    state=state,
                    netuid=args.netuid,
                    state_dir=args.state_dir,
                    dry_run=args.dry_run,
                )
                logger.info(
                    "☑️ Tick completed in %.1fs: block=%d commits=%d challengers=%d "
                    "weights=%s winner=%s",
                    time.time() - tick_start,
                    summary["block"],
                    summary["commitments"],
                    summary["challengers"],
                    summary["weights_set"],
                    summary["winner_uid"],
                )
            except Exception as exc:
                logger.exception(
                    "❌ Tick failed after %.1fs: %s", time.time() - tick_start, exc
                )
            finally:
                _try_upload_logs(args.state_dir)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down.")
        from .eval_progress import clear_progress
        from .gpu_orchestrator import teardown_rented_gpu_session

        teardown_rented_gpu_session()
        clear_progress(args.state_dir)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
