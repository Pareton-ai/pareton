#!/usr/bin/env bash
# Seeds the Meta Muse Spark Open campaign that mirrors the Open Kernel Challenge
# serving track (fixture fixtures/campaigns/sglang_muse_spark_open_h200_okc),
# under OKC's rules of 2026-11-11: SGLang v0.5.23 (297e54fd) with its stock
# launch flags, 400 KB patches, lossless patches only, and at most 2 GB more
# GPU memory per GPU than stock (the scoring rule's max_extra_gpu_memory_mib).
# Run once, on the validator, with /opt/pareton/.env loaded. Running it twice
# creates two campaigns. PARETON_DATABASE_URL must be configured. This creates
# a public open campaign.
#
# Pass the rule qualified by ops/campaign_preflight.py at the fixture's
# settings (keep_special_tokens, 0.001 token quantile); seed refuses a rule
# whose qualification contract does not match.
# Both review gates (#214) are on: d1 obfuscation v1 and the d2 LLM review
# okc_serving_v1 with kimi-k3. pareton-worker needs PARETON_LLM_REVIEW_BASE_URL,
# PARETON_LLM_REVIEW_API_KEY and PARETON_LLM_REVIEW_RESPONSE_FORMAT=json_schema,
# or every submission is rejected at d2.
# Patches stay private with no timed public reveal.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'Usage: seed-sglang-muse-spark-open-h200-okc.sh ENGINE_DIGEST_REF QUALIFIED_SAMPLING_RULE_JSON SUBMISSION_FEE_TAO [SCORING_RULE_JSON]' >&2
  echo 'Creates a new campaign. For an existing campaign use python -m campaign.set_fee.' >&2
  exit 2
fi
engine_ref=$1
sampling_rule=$2
fee_tao=$3
scoring_rule=${4:-fixtures/campaigns/sglang_muse_spark_open_h200_okc/scoring_rule.json}
if [[ ! -f "$sampling_rule" || ! -r "$sampling_rule" ]]; then
  echo "Qualified sampling rule must be a readable file: $sampling_rule" >&2
  exit 2
fi
if [[ ! "$engine_ref" =~ ^ghcr\.io/pareton-ai/(pareton-engine|pareton-baseline)@sha256:[a-f0-9]{64}$ ]]; then
  echo 'Pass the full published SGLang engine reference by digest' >&2
  exit 2
fi
# The Muse engine took 739 s to become healthy on 8xH200, past the 600 s
# default. The workers forward this value to every GPU pod, so raise it in the
# validator's .env and restart both workers before the campaign opens.
min_health_timeout_s=3600
health_timeout_s=${PARETON_BENCH_HEALTH_TIMEOUT_S:-600}
if [[ ! "$health_timeout_s" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
  || ! awk -v v="$health_timeout_s" -v m="$min_health_timeout_s" 'BEGIN { exit !(v + 0 >= m) }'; then
  echo "PARETON_BENCH_HEALTH_TIMEOUT_S is $health_timeout_s; Muse Spark Open needs at least $min_health_timeout_s." >&2
  echo 'Set it in /opt/pareton/.env, restart pareton-round-worker and pareton-worker, load the .env and rerun.' >&2
  exit 2
fi

python -m campaign.seed \
  --submission-fee-tao "$fee_tao" \
  --engine sglang-okc \
  --priority-metric throughput \
  --success-threshold ">=10% completion speedup at C16 with TTFT within 2x stock" \
  --patch-visibility private \
  --baseline-repo https://github.com/sgl-project/sglang.git \
  --baseline-commit 297e54fd105e974cdf401e5b4d20253a22c16f90 \
  --base-image-digest "$engine_ref" \
  --baseline-engine-image-digest "$engine_ref" \
  --gpu-skus H200 --bench-gpu-count 8 \
  --bench-model-repo meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
  --bench-model-revision 4239e4db19376904f2e82c167e78f70abb5455fe \
  --bench-dtype bfloat16 --bench-quantization modelopt_fp4 \
  --bench-max-model-len 131072 \
  --bench-draft-model-repo RadixArk/Muse-Spark-Open-DFlash2 \
  --bench-draft-model-revision 46c90f412549c472941a68382d91a99e5e6372a0 \
  --bench-serve-args=--served-model-name --bench-serve-args=Muse-Spark-Open \
  --bench-serve-args=--speculative-draft-model-path --bench-serve-args=/draft \
  --bench-serve-args=--tp-size --bench-serve-args=8 \
  --bench-serve-args=--ep-size --bench-serve-args=8 \
  --bench-serve-args=--moe-runner-backend --bench-serve-args=marlin \
  --bench-serve-args=--mem-fraction-static --bench-serve-args=0.85 \
  --bench-serve-args=--speculative-algorithm --bench-serve-args=DFLASH \
  --bench-serve-args=--speculative-num-draft-tokens --bench-serve-args=8 \
  --bench-serve-args=--speculative-draft-attention-backend --bench-serve-args=fa4 \
  --bench-serve-args=--mm-feature-transport --bench-serve-args=cpu \
  --bench-serve-args=--enable-hierarchical-cache \
  --bench-serve-args=--hicache-size --bench-serve-args=160 \
  --bench-serve-args=--chunked-prefill-size --bench-serve-args=16384 \
  --bench-serve-args=--enable-cache-report \
  --bench-serve-args=--incremental-streaming-output \
  --bench-serve-args=--random-seed --bench-serve-args=1234 \
  --bench-serve-args=--max-running-requests --bench-serve-args=32 \
  --bench-serve-args=--cuda-graph-max-bs-decode --bench-serve-args=32 \
  --bench-correctness-num-prompts 32 \
  --bench-correctness-serve-args=--max-running-requests --bench-correctness-serve-args=1 \
  --bench-correctness-min-mean-logprob=-4 \
  --bench-correctness-min-token-logprob=-16 \
  --bench-correctness-min-token-quantile=0.001 \
  --bench-correctness-min-coverage-ratio=0.5 \
  --bench-correctness-max-mean-logprob-drop=2.5 \
  --obfuscation v1 \
  --llm-review okc_serving_v1 --llm-review-model kimi-k3 \
  --sampling-rule-json "$sampling_rule" \
  --scoring-rule-json "$scoring_rule" \
  --status open --emission-start-weight 0.50 --emission-floor-weight 0 \
  --emission-decay-blocks 201600 --force
