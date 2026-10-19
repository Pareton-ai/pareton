#!/usr/bin/env bash
# Do not run until the PR's B300 validation and correctness checks pass.
# Run once, after image publication, miner build verification and GPU calibration.
# PARETON_DATABASE_URL must be configured. This creates a public open campaign.
# Use the Pareton engine built at SGLang 11972e5 by ops/build-sglang-baseline.sh.
# The harness mounts pinned weights at /model and adds --model-path itself; the
# serving flags below must not repeat it. The pinned DSPARK draft is staged the
# same way and mounted read-only at /draft.
# Uses the v5 32-request 2k/4k/8k/16k workload (8 per tier) at C32, one group of 32; engine capacity remains 64.
# For the Exgentic v6 16k-128k campaign, pass its qualified rule and
# fixtures/campaigns/sglang_kimi_k3_b300_exgentic/scoring_rule.json.
# A rule qualified for the earlier 16-prompt C16 contract can be restamped without
# new GPU work: PYTHONPATH=. python ops/restamp_k3_c32.py OLD_RULE NEW_RULE.
# Patches remain private with no timed public reveal.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: seed-sglang-kimi-k3-b300.sh ENGINE_DIGEST_REF QUALIFIED_SAMPLING_RULE_JSON [SCORING_RULE_JSON]' >&2
  echo 'Creates a new campaign. For an existing campaign use python -m campaign.set_fee.' >&2
  exit 2
fi
engine_ref=$1
sampling_rule=$2
scoring_rule=${3:-fixtures/campaigns/sglang_kimi_k3_b300/scoring_rule.json}
if [[ ! -f "$sampling_rule" || ! -r "$sampling_rule" ]]; then
  echo "Qualified sampling rule must be a readable file: $sampling_rule" >&2
  exit 2
fi
if [[ ! "$engine_ref" =~ ^ghcr\.io/pareton-ai/(pareton-engine|pareton-baseline)@sha256:[a-f0-9]{64}$ ]]; then
  echo 'Pass the full published SGLang engine reference by digest' >&2
  exit 2
fi
# Kimi K3 engines took over 20 minutes to become healthy on 8xB300. The workers
# forward this value to every GPU pod, so raise it in the validator's .env and
# restart both workers before the campaign opens. Seed with that .env loaded.
min_health_timeout_s=3600
health_timeout_s=${PARETON_BENCH_HEALTH_TIMEOUT_S:-600}
if [[ ! "$health_timeout_s" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
  || ! awk -v v="$health_timeout_s" -v m="$min_health_timeout_s" 'BEGIN { exit !(v + 0 >= m) }'; then
  echo "PARETON_BENCH_HEALTH_TIMEOUT_S is $health_timeout_s; Kimi K3 needs at least $min_health_timeout_s." >&2
  echo 'Set it in /opt/pareton/.env, restart pareton-round-worker and pareton-worker, load the .env and rerun.' >&2
  exit 2
fi

# Store the agreed 0.35 TAO initial fee with the open row, with no activation delay.
# seed validates whole RAO and the locally trusted recipient before insertion.
python -m campaign.seed \
  --submission-fee-tao 0.35 \
  --engine sglang \
  --patch-visibility private \
  --baseline-repo https://github.com/sgl-project/sglang.git \
  --baseline-commit 11972e520709a76766f91744a672e92727fdcb09 \
  --base-image-digest "$engine_ref" \
  --baseline-engine-image-digest "$engine_ref" \
  --gpu-skus B300 --bench-gpu-count 8 \
  --bench-model-repo moonshotai/Kimi-K3 \
  --bench-model-revision f831ab66814297da540d832a5235f8e904f29d06 \
  --bench-dtype bfloat16 --bench-max-model-len 1048576 \
  --bench-draft-model-repo RadixArk/Kimi-K3-DSpark \
  --bench-draft-model-revision 3c5bac301d9cf392706189d82ed947feca6c2f0f \
  --bench-serve-args=--served-model-name --bench-serve-args=Kimi-K3 \
  --bench-serve-args=--tp --bench-serve-args=8 \
  --bench-serve-args=--context-length --bench-serve-args=1048576 \
  --bench-serve-args=--mem-fraction-static --bench-serve-args=0.88 \
  --bench-serve-args=--max-running-requests --bench-serve-args=64 \
  --bench-serve-args=--enable-cache-report \
  --bench-serve-args=--enable-metrics \
  --bench-serve-args=--trust-remote-code \
  --bench-serve-args=--tool-call-parser --bench-serve-args=kimi_k3 \
  --bench-serve-args=--dcp-size --bench-serve-args=8 \
  --bench-serve-args=--max-mamba-cache-size --bench-serve-args=320 \
  --bench-serve-args=--speculative-algorithm --bench-serve-args=DSPARK \
  --bench-serve-args=--speculative-draft-model-path --bench-serve-args=/draft \
  --bench-serve-args=--speculative-dspark-block-size --bench-serve-args=3 \
  --bench-serve-args=--enable-linear-replayssm-spec \
  --bench-serve-args=--watchdog-timeout --bench-serve-args=3600 \
  --bench-serve-args=--reasoning-parser --bench-serve-args=kimi_k3 \
  --bench-correctness-num-prompts 32 \
  --bench-correctness-serve-args=--mem-fraction-static --bench-correctness-serve-args=0.80 \
  --bench-correctness-serve-args=--max-mamba-cache-size --bench-correctness-serve-args=16 \
  --bench-correctness-serve-args=--max-running-requests --bench-correctness-serve-args=8 \
  --bench-correctness-min-mean-logprob=-4 \
  --bench-correctness-min-token-logprob=-16 \
  --bench-correctness-min-token-quantile=0.001 \
  --bench-correctness-min-coverage-ratio=0.5 \
  --bench-correctness-max-mean-logprob-drop=2.5 \
  --sampling-rule-json "$sampling_rule" \
  --scoring-rule-json "$scoring_rule" \
  --status open --emission-start-weight 0.50 --emission-floor-weight 0 \
  --emission-decay-blocks 201600 --force
