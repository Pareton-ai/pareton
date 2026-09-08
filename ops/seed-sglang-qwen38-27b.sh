#!/usr/bin/env bash
# Run once, after image publication, miner build verification and GPU smoke.
# PARETON_DATABASE_URL must be configured. This creates a public open campaign.
set -euo pipefail
engine_ref=${1:?Usage: seed-sglang-qwen38-27b.sh PUBLISHED_ENGINE_DIGEST_REF}
if [[ ! "$engine_ref" =~ ^ghcr\.io/pareton-ai/(pareton-engine|pareton-baseline)@sha256:[a-f0-9]{64}$ ]]; then
  echo 'Pass the full published SGLang engine reference by digest' >&2
  exit 2
fi

python -m campaign.seed \
  --engine sglang \
  --baseline-repo https://github.com/sgl-project/sglang.git \
  --baseline-commit 4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc \
  --base-image-digest "$engine_ref" \
  --baseline-engine-image-digest "$engine_ref" \
  --gpu-skus H200 --bench-gpu-count 1 \
  --bench-model-repo Qwen/Qwen3.8-27B-FP8 \
  --bench-model-revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --bench-dtype bfloat16 --bench-quantization fp8 --bench-max-model-len 8192 \
  --bench-serve-args=--tp-size --bench-serve-args=1 \
  --bench-serve-args=--mem-fraction-static --bench-serve-args=0.80 \
  --bench-serve-args=--max-running-requests --bench-serve-args=32 \
  --bench-correctness-num-prompts 32 \
  --bench-correctness-min-mean-logprob=-4 \
  --bench-correctness-min-token-logprob=-12 \
  --bench-correctness-min-token-quantile=0.001 \
  --bench-correctness-min-coverage-ratio=0.5 \
  --bench-correctness-max-mean-logprob-drop=1.5 \
  --sampling-rule-json fixtures/campaigns/sglang_qwen38_27b/sampling_rule.json \
  --status open --emission-start-weight 0 --emission-floor-weight 0 \
  --emission-decay-blocks 201600 --force
