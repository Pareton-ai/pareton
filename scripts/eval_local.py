#!/usr/bin/env python3
"""Local eval test harness for two-pass validator evaluation.

Runs Pass 1 speed baseline + per-miner Pass 1/Pass 2 + inline correctness scoring against
manually-specified Docker images on a GPU pod. No bittensor, no S3, no
chain access -- just Docker and GPUs.

# Prerequisites:
# 1. (Optional) Set HuggingFace token:    export HF_TOKEN=...
# 2. Install deps if needed:              sudo apt-get install -y curl vim git
# 3. Setup env (run as root):             curl -fsSL "https://raw.githubusercontent.com/latent-to/cacheon/main/validator/setup-gpu.sh" | sudo -E bash
# 4. Activate your venv:                  cd ~/cacheon && source ~/venv-cacheon/bin/activate

Usage:
    python scripts/eval_local.py \
      --image xavierlyulatent/bad-miner-one:latest \
      --image xavierlyulatent/cacheon-example-miner:v1 \
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator import config as validator_config
from validator.chain import CommitmentRecord
from validator.docker_eval import (
    EVAL_N_STRESS_SCORED,
    EVAL_N_WARMUP,
    _detect_gpu_count,
    _max_model_len,
    ensure_eval_network,
    evaluate_challenger,
    run_baseline,
    score_challenger_teacher_forcing,
    start_baseline_for_scoring,
    stop_scoring_baseline,
)
from validator.prompts import sample_audit_prompts, sample_stress_prompts
from validator.state import EvaluationRecord

logger = logging.getLogger(__name__)

SCORING_BASELINE_STARTUP_TIMEOUT_S = 600


@dataclass
class _PendingPass2:
    record: EvaluationRecord
    audit_output_texts: list[str]
    audit_miner_tokens: list[list[str]]


BLOCK_HASH = "0" * 64


def _resolve_digest(image: str) -> str:
    pull = subprocess.run(
        ["docker", "pull", image], capture_output=True, text=True, timeout=600
    )
    if pull.returncode != 0:
        raise RuntimeError(
            f"docker pull failed for {image}: {pull.stderr.strip()[-400:]}"
        )
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0 and "@" in result.stdout:
        return result.stdout.strip().split("@", 1)[1]
    raise RuntimeError(f"Could not resolve digest for {image}")


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
        ttft_improvement=0.0,
        throughput_improvement=0.0,
        token_match_rate=record.token_match_rate,
        disqualified=True,
        disqualify_reason=f"{prefix}: " + "; ".join(fail_reasons),
        evaluated_at=record.evaluated_at,
        evaluation_block=record.evaluation_block,
        per_prompt=record.per_prompt,
    )


def _print_summary(records: list[tuple[str, EvaluationRecord]]) -> None:
    print("\n" + "=" * 70)
    print("LOCAL EVAL RESULTS")
    print("=" * 70)
    for image, record in records:
        print(f"\nImage: {image}")
        if record.disqualified:
            reason = record.disqualify_reason or "unknown"
            if reason.startswith("correctness_fail:"):
                print("  Correctness: FAIL (DQ)")
            elif reason.startswith("scoring_infra_fail:"):
                print("  Correctness: not run (scoring infra failed)")
            elif reason.startswith("pass1_match_fail:"):
                print("  Pass 1 match: FAIL (DQ)")
            elif reason.startswith("baseline_scoring_unavailable:"):
                print("  Correctness: not run (scoring infra failed)")
            else:
                print("  Correctness: not run (failed before scoring pass)")
            print(f"  Reason:      {reason}")
            print("  Score:       0.0000 (DQ)")
        else:
            print("  Correctness: PASS")
            print(f"  Score:       {record.score:.4f}")
            print(f"  TTFT imp:    {record.ttft_improvement:.4f}")
            print(f"  TPS imp:     {record.throughput_improvement:.4f}")
        print(f"  Match rate:  {record.token_match_rate:.4f}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local eval test harness")
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument(
        "--model-volume",
        default="/workspace/models/Qwen2.5-72B-Instruct",
    )
    parser.add_argument("--baseline-image", default=validator_config.BASELINE_IMAGE)
    parser.add_argument("--state-dir", default="/tmp/local-eval")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )

    state_dir = args.state_dir
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    (Path(state_dir) / "container_logs").mkdir(exist_ok=True)

    gpu_count = _detect_gpu_count()
    if gpu_count <= 0:
        logger.error("No GPUs detected via nvidia-smi")
        return 1
    logger.info("Detected %d GPU(s)", gpu_count)

    ensure_eval_network()
    baseline_digest = _resolve_digest(args.baseline_image)
    mml = _max_model_len(gpu_count, model_path=args.model_volume)
    stress_prompts = sample_stress_prompts(
        BLOCK_HASH, n=EVAL_N_STRESS_SCORED + EVAL_N_WARMUP, max_context_tokens=mml
    )
    audit_prompts = sample_audit_prompts(BLOCK_HASH)

    stress_baseline = run_baseline(
        stress_prompts,
        baseline_image=args.baseline_image,
        baseline_digest=baseline_digest,
        model_volume=args.model_volume,
        gpu_count=gpu_count,
        block_hash=BLOCK_HASH,
        state_dir=state_dir,
    )
    logger.info(
        "Pass 1 speed baseline ready: %d scored prompts", len(stress_baseline.results)
    )

    records: list[tuple[str, EvaluationRecord]] = []
    pending_pass2: list[_PendingPass2] = []
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scoring_log_label = f"baseline_scoring_0_{eval_ts}"

    for idx, image in enumerate(args.image):
        digest = _resolve_digest(image)
        com = CommitmentRecord(
            uid=idx,
            hotkey=f"local-test-{idx}",
            coldkey=f"local-coldkey-{idx}",
            commit_block=0,
            image=image,
            digest=digest,
            raw="",
        )
        logger.info("=== Evaluating [%d/%d]: %s ===", idx + 1, len(args.image), image)

        audit_output_texts: list[str] = []
        audit_miner_tokens: list[list[str]] = []
        record = evaluate_challenger(
            com,
            stress_prompts,
            audit_prompts,
            stress_baseline,
            model_volume=args.model_volume,
            startup_timeout_s=600,
            per_prompt_timeout_s=120,
            n_warmup=EVAL_N_WARMUP,
            current_block=0,
            state_dir=state_dir,
            collected_audit_output_texts=audit_output_texts,
            collected_audit_miner_tokens=audit_miner_tokens,
            max_model_len=mml,
        )
        if not record.disqualified and audit_output_texts:
            pending_pass2.append(
                _PendingPass2(record, audit_output_texts, audit_miner_tokens)
            )
        records.append((image, record))

    if pending_pass2:
        scoring_cid: str | None = None
        try:
            scoring_cid, scoring_url = start_baseline_for_scoring(
                model_volume=args.model_volume,
                gpu_count=gpu_count,
                state_dir=state_dir,
                startup_timeout_s=SCORING_BASELINE_STARTUP_TIMEOUT_S,
            )
            for pending in pending_pass2:
                try:
                    passed, fail_reasons, _ = score_challenger_teacher_forcing(
                        scoring_url,
                        audit_prompts,
                        pending.audit_output_texts,
                        pending.audit_miner_tokens,
                        log_prefix=pending.record.eval_key,
                    )
                except Exception as exc:
                    logger.error("Pass 2 scoring unavailable: %s", exc, exc_info=True)
                    pending.record = EvaluationRecord(
                        uid=pending.record.uid,
                        hotkey=pending.record.hotkey,
                        commit_block=pending.record.commit_block,
                        image=pending.record.image,
                        digest=pending.record.digest,
                        score=0.0,
                        ttft_improvement=0.0,
                        throughput_improvement=0.0,
                        token_match_rate=pending.record.token_match_rate,
                        disqualified=True,
                        disqualify_reason=f"baseline_scoring_unavailable: {exc}",
                        evaluated_at=pending.record.evaluated_at,
                        evaluation_block=pending.record.evaluation_block,
                        per_prompt=pending.record.per_prompt,
                    )
                    continue
                if not passed:
                    pending.record = _dq_from_scoring_fail(pending.record, fail_reasons)
        except Exception as exc:
            logger.exception("Pass 2 scoring baseline failed: %s", exc)
            for pending in pending_pass2:
                pending.record = EvaluationRecord(
                    uid=pending.record.uid,
                    hotkey=pending.record.hotkey,
                    commit_block=pending.record.commit_block,
                    image=pending.record.image,
                    digest=pending.record.digest,
                    score=0.0,
                    ttft_improvement=0.0,
                    throughput_improvement=0.0,
                    token_match_rate=pending.record.token_match_rate,
                    disqualified=True,
                    disqualify_reason=f"baseline_scoring_unavailable: {exc}",
                    evaluated_at=pending.record.evaluated_at,
                    evaluation_block=pending.record.evaluation_block,
                    per_prompt=pending.record.per_prompt,
                )
        finally:
            if scoring_cid:
                stop_scoring_baseline(
                    scoring_cid, state_dir, log_label=scoring_log_label
                )

    _print_summary(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
