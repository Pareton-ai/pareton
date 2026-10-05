#!/usr/bin/env bash
# Run once, after image publication, miner build verification and GPU calibration.
# PARETON_DATABASE_URL must be configured. This creates a public open campaign.
# Use the Pareton engine from ops/build-sglang-baseline.sh. The upstream
# lmsysorg/sglang runtime image lacks the trusted offline miner-build installer.
# The harness mounts pinned weights at /model and manages Docker networking,
# listen address, port and GPU allocation separately from these serving flags.
# Uses the v5 32-request workload at C4; engine capacity remains 32.
# Patches remain private with no timed public reveal.
# Qualify this TP1/FP8/MTP configuration on RTXPRO6000 before opening.
# Do not reuse TP4/NVFP4 qualification or its memory-fraction overrides.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: seed-sglang-qwen38-27b-pro6000.sh ENGINE_DIGEST_REF QUALIFIED_SAMPLING_RULE_JSON' >&2
  echo 'Creates a new campaign. For an existing campaign use python -m campaign.set_fee.' >&2
  exit 2
fi
engine_ref=$1
sampling_rule=$2
if [[ ! -f "$sampling_rule" || ! -r "$sampling_rule" ]]; then
  echo "Qualified sampling rule must be a readable file: $sampling_rule" >&2
  exit 2
fi
if [[ ! "$engine_ref" =~ ^ghcr\.io/pareton-ai/(pareton-engine|pareton-baseline)@sha256:[a-f0-9]{64}$ ]]; then
  echo 'Pass the full published SGLang engine reference by digest' >&2
  exit 2
fi

# Store the agreed 0.1 TAO initial fee with the open row, with no activation delay.
# seed validates whole RAO and the locally trusted recipient before insertion.
python -m campaign.seed \
  --submission-fee-tao 0.1 \
  --engine sglang \
  --patch-visibility private \
  --baseline-repo https://github.com/sgl-project/sglang.git \
  --baseline-commit 4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc \
  --base-image-digest "$engine_ref" \
  --baseline-engine-image-digest "$engine_ref" \
  --gpu-skus RTXPRO6000 --bench-gpu-count 1 \
  --bench-model-repo Qwen/Qwen3.8-27B-FP8 \
  --bench-model-revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --bench-dtype bfloat16 --bench-quantization fp8 --bench-max-model-len 262144 \
  --bench-serve-args=--tp --bench-serve-args=1 \
  --bench-serve-args=--tool-call-parser --bench-serve-args=qwen3_coder \
  --bench-serve-args=--reasoning-parser --bench-serve-args=qwen3 \
  --bench-serve-args=--context-length --bench-serve-args=262144 \
  --bench-serve-args=--max-running-requests --bench-serve-args=32 \
  --bench-serve-args=--mm-enable-dp-encoder \
  --bench-serve-args=--speculative-algorithm --bench-serve-args=EAGLE \
  --bench-serve-args=--speculative-num-steps --bench-serve-args=3 \
  --bench-serve-args=--speculative-eagle-topk --bench-serve-args=1 \
  --bench-serve-args=--speculative-num-draft-tokens --bench-serve-args=4 \
  --bench-correctness-num-prompts 32 \
  --bench-correctness-min-mean-logprob=-4 \
  --bench-correctness-min-token-logprob=-16 \
  --bench-correctness-min-token-quantile=0.001 \
  --bench-correctness-min-coverage-ratio=0.5 \
  --bench-correctness-max-mean-logprob-drop=2.5 \
  --sampling-rule-json "$sampling_rule" \
  --scoring-rule-json fixtures/campaigns/sglang_qwen38_27b_pro6000/scoring_rule.json \
  --status open --emission-start-weight 0.20 --emission-floor-weight 0 \
  --emission-decay-blocks 201600 --force
