"""GPU eval entrypoint: Postgres-backed state, S3 for logs only.

Zero bittensor dependency. Loads the pending eval job and validator state
from Postgres, runs baseline + challenger evaluations, mirrors results to
Postgres, and uploads eval logs to Hippius S3.

Usage (inside Docker):
    python -m validator.gpu_eval

Env vars:
    CACHEON_STATE_DIR          (default: /app/state-mainnet)
    CACHEON_DATABASE_URL       (required)
    CACHEON_MODEL_VOLUME       (default: /models)
    CACHEON_BASELINE_IMAGE     (default: vllm/vllm-openai:v0.22.0)
    CACHEON_BASELINE_DIGEST    (required)
    CACHEON_GPU_COUNT          (default: 8, auto-detect via nvidia-smi)
    CACHEON_VLLM_CACHE_DIR     (optional; host path for vLLM torch.compile cache)
    HIPPIUS_ACCESS_KEY         (required for log upload unless CACHEON_SKIP_S3=1)
    HIPPIUS_SECRET_KEY         (required for log upload unless CACHEON_SKIP_S3=1)
    CACHEON_SKIP_S3            (default: 0; set 1 to skip log upload)
    CACHEON_S3_BUCKET          (default: cacheon-validator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import config as validator_config
from .chain import CommitmentRecord
from .docker_eval import (
    EVAL_N_SCORED,
    EVAL_N_WARMUP,
    _max_model_len,
    evaluate_challenger,
    run_baseline,
    score_challenger_teacher_forcing,
    start_baseline_for_scoring,
    stop_scoring_baseline,
    _dq_record,
)
from .content_fingerprint import (
    FingerprintCheckResult,
    apply_fingerprint_supersede_records,
    load_fingerprint_registry,
)
from .state import EvaluationRecord, ValidatorState, append_leader_history

from .eval_progress import (
    clear_progress,
    complete_progress,
    purge_old_logs,
    seed_progress_from_eval_job,
    update_challenger_status,
    update_incumbent_status,
    update_progress,
)
from .eval_schema import EvalJob

logger = logging.getLogger(__name__)


@dataclass
class _PendingTeacherForcing:
    record: EvaluationRecord
    tf_output_texts: list[str]
    tf_miner_tokens: list[list[str]]
    log_prefix: str


def _configure_logging(state_dir: str, block: int) -> str:
    """Configure logging and return the run timestamp (for paired artifact names)."""
    logs_dir = Path(state_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"gpu_{block}_{ts}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_path)
    return ts


def _complete_eval_job(block: int) -> None:
    from cacheon_db.mirror import complete_eval_job

    complete_eval_job(block)


_GPU_LOGS_ONLY = ["logs/", "container_logs/"]


def _upload_eval_logs(state_dir: str) -> None:
    """Push eval log artifacts to S3 (structured state stays in Postgres)."""
    if validator_config.SKIP_S3:
        return
    try:
        from .sync import upload

        upload(state_dir, only=_GPU_LOGS_ONLY)
    except Exception:
        logger.debug("Eval log upload failed", exc_info=True)


def main() -> int:
    from cacheon_db.config import require_database_url
    from cacheon_db.exceptions import DatabaseNotConfigured

    try:
        require_database_url()
    except DatabaseNotConfigured as exc:
        logger.error("%s", exc)
        return 2

    state_dir = str(validator_config.STATE_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )
    purge_old_logs(state_dir)

    model_volume = validator_config.MODEL_VOLUME
    model_path = validator_config.MODEL_PATH
    baseline_image = validator_config.BASELINE_IMAGE
    baseline_digest = validator_config.BASELINE_DIGEST
    gpu_count = validator_config.GPU_COUNT

    logger.info(
        "GPU eval starting: state_dir=%s, model_volume=%s, model_path=%s, "
        "baseline_image=%s, gpu_count=%d",
        state_dir,
        model_volume,
        model_path,
        baseline_image,
        gpu_count,
    )

    if not baseline_digest:
        import subprocess

        try:
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    baseline_image,
                    "--format",
                    "{{index .RepoDigests 0}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and "@" in result.stdout:
                baseline_digest = result.stdout.strip().split("@", 1)[1]
                logger.info("Auto-detected baseline digest: %s", baseline_digest)
        except Exception as exc:
            logger.debug("Could not auto-detect baseline digest: %s", exc)

        if not baseline_digest:
            logger.error(
                "CACHEON_BASELINE_DIGEST is not set and could not be "
                "auto-detected. Pull the baseline image first."
            )
            return 1

    # Load state and eval job (Postgres-first; local JSON fallback in tests).
    state = ValidatorState.load()
    fingerprint_registry = load_fingerprint_registry()
    eval_job = EvalJob.load()
    has_work = eval_job is not None and (
        eval_job.challengers or eval_job.leader or eval_job.runner_up
    )
    if not has_work:
        logger.error("No pending eval job in Postgres; cannot run GPU eval")
        clear_progress()
        return 2

    block = eval_job.block
    eval_ts = _configure_logging(state_dir, block)
    block_hash = eval_job.block_hash
    seed_progress_from_eval_job(eval_job)
    logger.info(
        "Eval job: block=%d, block_hash=%s, %d challenger(s), leader=%s, runner_up=%s",
        block,
        block_hash[:16] if block_hash else "None",
        len(eval_job.challengers),
        eval_job.leader.hotkey[:16] if eval_job.leader else "none",
        eval_job.runner_up.hotkey[:16] if eval_job.runner_up else "none",
    )

    if gpu_count <= 0:
        from .docker_eval import _detect_gpu_count

        gpu_count = _detect_gpu_count()
        if gpu_count <= 0:
            logger.error("Could not detect GPU count via nvidia-smi")
            return 3
        logger.info("Auto-detected %d GPU(s)", gpu_count)

    from .prompts import EVAL_CONTEXT_TOKENS, sample_eval_prompts

    mml = _max_model_len(gpu_count, model_path=model_path)
    eval_prompts = sample_eval_prompts(
        block_hash,
        n=EVAL_N_SCORED + EVAL_N_WARMUP,
        max_context_tokens=EVAL_CONTEXT_TOKENS,
    )
    scored_eval_prompts = eval_prompts[EVAL_N_WARMUP:]
    logger.info(
        "Generated %d eval prompts (10 scored + 2 warmup, ctx=%d, max_model_len=%d)",
        len(eval_prompts),
        EVAL_CONTEXT_TOKENS,
        mml,
    )
    update_progress(
        phase="prompts_generated",
        n_eval=len(eval_prompts),
    )
    update_progress(phase="baseline_running", image=baseline_image)
    try:
        eval_baseline = run_baseline(
            eval_prompts,
            baseline_image=baseline_image,
            baseline_digest=baseline_digest,
            model_volume=model_volume,
            gpu_count=gpu_count,
            block_hash=block_hash,
            evaluation_block=block,
            state_dir=state_dir,
        )
    except Exception as exc:
        logger.exception("Baseline failed: %s", exc)
        update_progress(
            phase="baseline_failed",
            error=str(exc),
            challengers_affected=len(eval_job.challengers),
        )
        _upload_eval_logs(state_dir)
        return 4

    update_progress(phase="baseline_complete")

    scoring_log_label = f"baseline_scoring_{block}_{eval_ts}"
    pending_tf: list[_PendingTeacherForcing] = []

    def _dq_from_scoring_fail(
        record: EvaluationRecord,
        fail_reasons: list[str],
    ) -> EvaluationRecord:
        if any("scoring_infra_fail:" in r for r in fail_reasons):
            prefix = "scoring_infra_fail"
        else:
            prefix = "correctness_fail"
        return EvaluationRecord(
            uid=record.uid,
            hotkey=record.hotkey,
            commit_block=record.commit_block,
            image=record.image,
            digest=record.digest,
            score=0.0,
            speed_improvement=0.0,
            token_match_rate=record.token_match_rate,
            disqualified=True,
            disqualify_reason=f"{prefix}: " + "; ".join(fail_reasons),
            evaluated_at=record.evaluated_at,
            evaluation_block=record.evaluation_block,
            per_prompt=record.per_prompt,
        )

    TF_DEFER_DETAIL = "scoring_baseline_unavailable"

    def _queue_teacher_forcing(
        record: EvaluationRecord,
        tf_output_texts: list[str],
        tf_miner_tokens: list[list[str]],
        log_prefix: str,
    ) -> None:
        if record.disqualified or not tf_output_texts:
            return
        pending_tf.append(
            _PendingTeacherForcing(
                record=record,
                tf_output_texts=tf_output_texts,
                tf_miner_tokens=tf_miner_tokens,
                log_prefix=log_prefix,
            )
        )

    def _apply_teacher_forcing(
        scoring_url: str, pending: _PendingTeacherForcing
    ) -> EvaluationRecord:
        record = pending.record
        passed, fail_reasons, _ = score_challenger_teacher_forcing(
            scoring_url,
            scored_eval_prompts,
            pending.tf_output_texts,
            pending.tf_miner_tokens,
            log_prefix=pending.log_prefix,
        )
        if passed:
            return record
        dq = _dq_from_scoring_fail(record, fail_reasons)
        logger.warning(
            "DQ %s via teacher-forcing: %s", pending.log_prefix, dq.disqualify_reason
        )
        return dq

    def _batch_run_teacher_forcing() -> bool:
        if not pending_tf:
            return True
        logger.info(
            "Teacher-forcing batch: %d challenger(s), log_label=%s",
            len(pending_tf),
            scoring_log_label,
        )
        update_progress(phase="teacher_forcing_running")
        scoring_cid: str | None = None
        try:
            scoring_cid, scoring_url = start_baseline_for_scoring(
                model_volume=model_volume,
                gpu_count=gpu_count,
                state_dir=state_dir,
                startup_timeout_s=validator_config.SCORING_BASELINE_STARTUP_TIMEOUT_S,
            )
            logger.info("Scoring baseline started at %s", scoring_url)
            for pending in pending_tf:
                pending.record = _apply_teacher_forcing(scoring_url, pending)
        except Exception as exc:
            logger.exception("Teacher-forcing scoring baseline failed: %s", exc)
            return False
        finally:
            if scoring_cid:
                stop_scoring_baseline(
                    scoring_cid, state_dir, log_label=scoring_log_label
                )
        update_progress(phase="teacher_forcing_complete")
        return True

    # ------------------------------------------------------------------ #
    # Evaluate leader and runner-up (fresh scores on same hardware)
    # ------------------------------------------------------------------ #

    def _eval_incumbent(ci, label: str):
        """Run a leader or runner-up container; return record, fp events, and com."""
        if ci is None:
            return None, [], None
        com = CommitmentRecord(
            uid=ci.uid,
            hotkey=ci.hotkey,
            coldkey="",
            commit_block=ci.commit_block,
            image=ci.image,
            digest=ci.digest,
            raw="",
        )
        logger.info(
            "Re-evaluating %s UID %d (%s) image=%s",
            label,
            com.uid,
            com.hotkey[:16],
            com.image,
        )
        update_progress(phase=f"{label}_running")
        update_incumbent_status(label, status="evaluating")
        tf_output_texts: list[str] = []
        tf_miner_tokens: list[list[str]] = []
        fp_events: list[FingerprintCheckResult] = []
        record = evaluate_challenger(
            com,
            eval_prompts,
            eval_baseline,
            model_volume=model_volume,
            startup_timeout_s=validator_config.CHALLENGER_STARTUP_TIMEOUT_S,
            per_prompt_timeout_s=120,
            n_warmup=EVAL_N_WARMUP,
            current_block=block,
            state_dir=state_dir,
            log_label=f"{label}_{block}",
            collected_tf_output_texts=tf_output_texts,
            collected_tf_miner_tokens=tf_miner_tokens,
            max_model_len=mml,
            fingerprint_registry=fingerprint_registry,
            fingerprint_result=fp_events,
        )
        _queue_teacher_forcing(
            record,
            tf_output_texts,
            tf_miner_tokens,
            log_prefix=f"{label}:{com.hotkey[:16]}",
        )
        logger.info(
            "%s UID %d generation complete (score=%.4f, dq=%s)",
            label,
            record.uid,
            record.score,
            record.disqualify_reason or "no",
        )
        if record.disqualified:
            update_incumbent_status(
                label, status="dq", dq_reason=record.disqualify_reason
            )
        else:
            update_incumbent_status(label, status="awaiting_correctness")
        return record, fp_events, com

    leader_record: EvaluationRecord | None = None
    ru_record: EvaluationRecord | None = None
    challenger_persist: list[tuple[int, EvaluationRecord]] = []

    def _apply_fp_supersede(
        fp_events: list[FingerprintCheckResult],
        com: CommitmentRecord,
    ) -> None:
        nonlocal leader_record, ru_record, challenger_persist
        for fp_result in fp_events:
            leader_record, ru_record, challenger_persist, removed_keys = (
                apply_fingerprint_supersede_records(
                    fp_result,
                    com,
                    leader_record=leader_record,
                    ru_record=ru_record,
                    challenger_records=challenger_persist,
                )
            )
            if removed_keys:
                pending_tf[:] = [
                    p for p in pending_tf if p.record.eval_key not in removed_keys
                ]
            if fp_result.superseded_owner is not None:
                if leader_record is not None and leader_record.disqualified:
                    update_incumbent_status(
                        "leader",
                        status="dq",
                        dq_reason=leader_record.disqualify_reason,
                    )
                if ru_record is not None and ru_record.disqualified:
                    update_incumbent_status(
                        "runner_up",
                        status="dq",
                        dq_reason=ru_record.disqualify_reason,
                    )

    leader_record, leader_fp, leader_com = _eval_incumbent(eval_job.leader, "leader")
    if leader_com is not None:
        _apply_fp_supersede(leader_fp, leader_com)

    ru_record, ru_fp, ru_com = _eval_incumbent(eval_job.runner_up, "runner_up")
    if ru_com is not None:
        _apply_fp_supersede(ru_fp, ru_com)

    # ------------------------------------------------------------------ #
    # Evaluate challengers (generation + collect outputs for teacher-forcing)
    # ------------------------------------------------------------------ #

    challenger_records: list = []

    for challenger_idx, ci in enumerate(eval_job.challengers):
        if state.is_known(ci.hotkey, ci.commit_block):
            logger.info("Skipping UID %d (already evaluated)", ci.uid)
            update_challenger_status(
                challenger_idx, status="skipped", detail="already_known"
            )
            continue

        com = CommitmentRecord(
            uid=ci.uid,
            hotkey=ci.hotkey,
            coldkey="",
            commit_block=ci.commit_block,
            image=ci.image,
            digest=ci.digest,
            raw="",
        )
        logger.info(
            "Evaluating challenger UID %d (%s) image=%s",
            com.uid,
            com.hotkey[:16],
            com.image,
        )
        update_challenger_status(
            challenger_idx, status="pulling", detail="pulling_image"
        )

        def _on_challenger_pulled() -> None:
            update_challenger_status(
                challenger_idx, status="evaluating", detail="evaluating"
            )

        tf_output_texts: list[str] = []
        tf_miner_tokens: list[list[str]] = []
        fp_events: list[FingerprintCheckResult] = []
        record = evaluate_challenger(
            com,
            eval_prompts,
            eval_baseline,
            model_volume=model_volume,
            startup_timeout_s=validator_config.CHALLENGER_STARTUP_TIMEOUT_S,
            per_prompt_timeout_s=120,
            n_warmup=EVAL_N_WARMUP,
            current_block=block,
            state_dir=state_dir,
            collected_tf_output_texts=tf_output_texts,
            collected_tf_miner_tokens=tf_miner_tokens,
            max_model_len=mml,
            on_pulled=_on_challenger_pulled,
            fingerprint_registry=fingerprint_registry,
            fingerprint_result=fp_events,
        )
        _apply_fp_supersede(fp_events, com)
        _queue_teacher_forcing(
            record,
            tf_output_texts,
            tf_miner_tokens,
            log_prefix=f"{com.hotkey}:{com.commit_block}",
        )
        if record.disqualified:
            update_challenger_status(
                challenger_idx,
                status="dq",
                dq_reason=record.disqualify_reason,
                detail="disqualified",
            )
        else:
            update_challenger_status(
                challenger_idx,
                status="awaiting_correctness",
                detail="awaiting_correctness",
            )
        challenger_persist.append((challenger_idx, record))

    scoring_available = _batch_run_teacher_forcing()
    deferred_keys = (
        {p.record.eval_key for p in pending_tf} if not scoring_available else set()
    )
    if not scoring_available:
        logger.warning(
            "Teacher-forcing scoring unavailable; deferring %d participant(s)",
            len(deferred_keys),
        )

    if scoring_available:
        tf_by_key = {p.record.eval_key: p.record for p in pending_tf}
        challenger_persist = [
            (idx, tf_by_key.get(rec.eval_key, rec)) for idx, rec in challenger_persist
        ]
        if leader_record is not None:
            leader_record = tf_by_key.get(leader_record.eval_key, leader_record)
        if ru_record is not None:
            ru_record = tf_by_key.get(ru_record.eval_key, ru_record)

    for challenger_idx, record in challenger_persist:
        if record.eval_key in deferred_keys:
            update_challenger_status(
                challenger_idx,
                status="skipped",
                detail=TF_DEFER_DETAIL,
            )
            continue
        outcome = state.record_evaluation(record, current_block=block)
        icon = "❌" if outcome.stored.disqualify_reason else "📊"
        logger.info(
            "%s UID %d score=%.4f (dq=%s)",
            icon,
            outcome.stored.uid,
            outcome.stored.score,
            outcome.stored.disqualify_reason or "no",
        )
        if outcome.stored.disqualified:
            update_challenger_status(
                challenger_idx,
                status="dq",
                score=outcome.stored.score,
                dq_reason=outcome.stored.disqualify_reason,
                detail="disqualified",
            )
        else:
            update_challenger_status(
                challenger_idx,
                status="scored",
                score=outcome.stored.score,
                detail="scored",
            )
        challenger_records.append(outcome.stored)

        state.save()

    for label, rec in (("leader", leader_record), ("runner_up", ru_record)):
        if rec is None:
            continue
        if rec.eval_key in deferred_keys:
            update_incumbent_status(label, status="skipped", detail=TF_DEFER_DETAIL)
            continue
        outcome = state.record_evaluation(rec, current_block=block)
        stored = outcome.stored
        if label == "leader":
            leader_record = stored
        else:
            ru_record = stored
        icon = "❌" if stored.disqualified else "📊"
        logger.info(
            "%s %s UID %d score=%.4f (dq=%s) [after teacher-forcing]",
            icon,
            label,
            stored.uid,
            stored.score,
            stored.disqualify_reason or "no",
        )
        update_incumbent_status(
            label,
            status="dq" if stored.disqualified else "scored",
            score=stored.score,
            dq_reason=stored.disqualify_reason,
        )
    # ------------------------------------------------------------------ #
    # Rerank leader/RU fresh scores against challengers
    # ------------------------------------------------------------------ #

    if scoring_available:
        prev_winner = state.winner
        from .state import OVERTAKE_EPSILON

        new_winner, new_ru = state.rerank_round(
            leader_record=leader_record,
            ru_record=ru_record,
            challenger_records=challenger_records,
            current_block=block,
        )
        state.winner = new_winner
        state.runner_up_record = new_ru

        winner_changed = (new_winner is None) != (prev_winner is None) or (
            new_winner is not None
            and prev_winner is not None
            and (
                new_winner.hotkey != prev_winner.hotkey
                or new_winner.score != prev_winner.score
            )
        )
        if winner_changed:
            winner_ev = None
            if new_winner is not None:
                base_key = f"{new_winner.hotkey}:{new_winner.commit_block}"
                winner_ev = state.evaluations.get(
                    f"{base_key}:{block}"
                ) or state.evaluations.get(base_key)
            if winner_ev is not None:
                threshold = (
                    new_winner.score * (1.0 + OVERTAKE_EPSILON) if prev_winner else 0.0
                )
                append_leader_history(winner_ev, prev_winner, block, threshold)
            if new_winner is not None:
                logger.info(
                    "New winner: UID %d (score=%.4f)", new_winner.uid, new_winner.score
                )
            else:
                logger.info("No eligible winner this round")
    else:
        logger.warning("Scoring round void; preserving winner/runner-up")

    state.save()

    logger.info(
        "State saved. Winner: %s", f"UID {state.winner.uid}" if state.winner else "none"
    )
    logger.info("GPU eval complete")
    _complete_eval_job(block)
    complete_progress()
    _upload_eval_logs(state_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
