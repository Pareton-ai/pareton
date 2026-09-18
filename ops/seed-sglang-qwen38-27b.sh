#!/usr/bin/env bash
# Run once, after image publication, miner build verification and GPU calibration.
# PARETON_DATABASE_URL must be configured. This creates a public open campaign.
# Use the Pareton engine from ops/build-sglang-baseline.sh. The upstream
# lmsysorg/sglang runtime image lacks the trusted offline miner-build installer.
# The harness mounts pinned weights at /model and manages Docker networking,
# listen address, port and GPU allocation separately from these serving flags.
# RadixArk's checkpoint uses mixed NVFP4/FP8 layers with a BF16 lm_head.
# SGLang loads its per-layer quantization with modelopt_mixed.
# Reserve scorer memory for full-input logprobs (logprob_start_len=0).
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: seed-sglang-qwen38-27b.sh PUBLISHED_ENGINE_DIGEST_REF INITIAL_FEE_TAO QUALIFIED_SAMPLING_RULE_JSON' >&2
  echo 'Creates a new campaign. For an existing campaign use python -m campaign.set_fee.' >&2
  exit 2
fi
engine_ref=$1
initial_fee_tao=$2
sampling_rule=$3
if [[ ! -f "$sampling_rule" || ! -r "$sampling_rule" ]]; then
  echo "Qualified sampling rule must be a readable file: $sampling_rule" >&2
  exit 2
fi
if [[ ! "$engine_ref" =~ ^ghcr\.io/pareton-ai/(pareton-engine|pareton-baseline)@sha256:[a-f0-9]{64}$ ]]; then
  echo 'Pass the full published SGLang engine reference by digest' >&2
  exit 2
fi

# Store the initial fee with the open row, with no delayed activation window.
# seed validates whole RAO and the locally trusted recipient before insertion.
python -m campaign.seed \
  --submission-fee-tao "$initial_fee_tao" \
  --engine sglang \
  --baseline-repo https://github.com/sgl-project/sglang.git \
  --baseline-commit 4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc \
  --base-image-digest "$engine_ref" \
  --baseline-engine-image-digest "$engine_ref" \
  --gpu-skus RTX5090 --bench-gpu-count 4 \
  --bench-model-repo RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead \
  --bench-model-revision 009632fef96dd349150baa780c984e62e70e91fe \
  --bench-dtype bfloat16 --bench-quantization modelopt_mixed --bench-max-model-len 262144 \
  --bench-serve-args=--trust-remote-code \
  --bench-serve-args=--served-model-name --bench-serve-args=qwen3.8-27b \
  --bench-serve-args=--tp-size --bench-serve-args=4 \
  --bench-serve-args=--kv-cache-dtype --bench-serve-args=bfloat16 \
  --bench-serve-args=--mem-fraction-static --bench-serve-args=0.85 \
  --bench-serve-args=--attention-backend --bench-serve-args=flashinfer \
  --bench-serve-args=--chunked-prefill-size --bench-serve-args=8192 \
  --bench-serve-args=--mamba-radix-cache-strategy --bench-serve-args=extra_buffer \
  --bench-serve-args=--max-running-requests --bench-serve-args=40 \
  --bench-serve-args=--reasoning-parser --bench-serve-args=qwen3 \
  --bench-serve-args=--tool-call-parser --bench-serve-args=qwen3_coder \
  --bench-serve-args=--enable-cache-report \
  --bench-correctness-num-prompts 32 \
  --bench-correctness-serve-args=--mem-fraction-static --bench-correctness-serve-args=0.4 \
  --bench-correctness-min-mean-logprob=-4 \
  --bench-correctness-min-token-logprob=-12 \
  --bench-correctness-min-token-quantile=0.001 \
  --bench-correctness-min-coverage-ratio=0.5 \
  --bench-correctness-max-mean-logprob-drop=1.5 \
  --sampling-rule-json "$sampling_rule" \
  --scoring-rule-json fixtures/campaigns/sglang_qwen38_27b/scoring_rule.json \
  --status open --emission-start-weight 0.20 --emission-floor-weight 0 \
  --emission-decay-blocks 201600 --force
