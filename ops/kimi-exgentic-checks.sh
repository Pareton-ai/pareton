#!/usr/bin/env bash
# Every pre-launch GPU check for the Kimi K3 16k-128k (Exgentic v6) campaign, in
# order, on one fresh 8xB300 host: host checks, CPU audit, scorer smoke, baseline
# qualification with the preflight, and the shadow round.
#
# From the repo root, with HF_TOKEN exported:
#   nohup bash ops/kimi-exgentic-checks.sh > /dev/null 2>&1 < /dev/null &
# Progress:  tail -f /workspace/kimi-exgentic/run.log   (current step: cat .../STATUS)
# Results:   /workspace/kimi-exgentic/RESULTS.md, and everything in
#            /workspace/kimi-exgentic-results.tgz when it ends.
# Running the same command again resumes: steps that passed are skipped, and a
# failed step's directory is kept under a timestamped name.
#
# Fallbacks it takes on its own, once each, so a failure does not cost a rental:
# - scorer smoke capacity failure: retry with --mem-fraction-static 0.84;
# - scorer smoke out of memory: retry with 0.76;
# - preflight that could not finish after the baseline stages: regrade the
#   saved outputs on one scorer start;
# - shadow round that could not finish: run it once more.
# A changed scorer fraction is part of the qualification contract: RESULTS.md
# says so, and #206's fixture and seed script must be changed to match.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
LAUNCH=${LAUNCH:-/workspace/kimi-exgentic}
HF_CACHE=${PARETON_BENCH_HF_CACHE_DIR:-/workspace/hf-cache}
MIN_SESSIONS=16       # qualification pool: twice the 8 prompts per tier
MIN_FREE_GB=${MIN_FREE_GB:-1800}  # Kimi K3 and its draft are about 1.6 TB
declare -A LAST=()
mkdir -p "$LAUNCH" "$HF_CACHE"
exec >>"$LAUNCH/run.log" 2>&1
echo $$ > "$LAUNCH/PID"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
step() { echo "$*" > "$LAUNCH/STATUS"; log "== $*"; }
status_of() { jq -r '.status // "missing"' "$1/summary.json" 2>/dev/null || echo missing; }
# Keep a failed attempt's evidence and free the name for the next one.
set_aside() { [[ -e $1 ]] && mv "$1" "$1.failed-$(date -u +%Y%m%dT%H%M%SZ)"; return 0; }
fraction() { jq -r '.bench.correctness.serve_args as $a
  | $a[($a | index("--mem-fraction-static")) + 1]' "$FIELDS"; }

finish() {
  local code=$1 message=$2
  step "finished: $message"
  {
    echo "# Kimi K3 16k-128k pre-launch checks"
    echo
    echo "- Result: **$message** (exit $code)"
    echo "- Finished: $(date -u +%FT%TZ)"
    echo "- Commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
    if [[ -n ${FIELDS:-} && -f $FIELDS ]]; then
      echo "- Scorer serve args: \`$(jq -c .bench.correctness.serve_args "$FIELDS")\`"
      if [[ $(fraction) != 0.80 ]]; then
        echo "- **The scorer fraction changed to $(fraction).** Set it in #206's"
        echo "  campaign-fields.json and ops/seed-sglang-kimi-k3-b300.sh before seeding."
      fi
    fi
    if [[ -f $LAUNCH/qualified_sampling_rule.json ]]; then
      echo "- Qualified rule: \`$LAUNCH/qualified_sampling_rule.json\`"
      echo "  ($(sha256sum "$LAUNCH/qualified_sampling_rule.json" | cut -d' ' -f1))"
    fi
    echo
    for name in audit smoke preflight shadow; do
      # The passing attempt, else the last one tried.
      local dir=$LAUNCH/$name
      [[ -n ${LAST[$name]:-} ]] && dir=${LAST[$name]}
      [[ -e $LAUNCH/$name.passed ]] && dir=$(readlink -f "$LAUNCH/$name.passed")
      [[ -f $dir/summary.json || -f $dir/audit.json ]] || continue
      echo "## $name ($dir)"
      echo
      echo '```json'
      case $name in
        audit) jq '{eligible_snapshots_by_tier, distinct_sessions_by_tier, preview_round_kv}' \
          "$LAUNCH/audit/audit.json" 2>/dev/null ;;
        smoke) jq '{status, failures, error, capacity, slowest_s, peak_used_mib, log_errors}' \
          "$dir/summary.json" ;;
        preflight) jq '{status, failures, error, input_tokens, by_temperature,
          below_min_output, greedy_below_floor: .greedy.below_floor, sla}' "$dir/summary.json" ;;
        shadow) jq '{status, failures, error, verdict, baseline_drift,
          baseline_tier_completion_s, entries}' "$dir/summary.json"
          jq '{excluded_baseline_prompts: .baseline.excluded_prompts}' \
            "$dir/round/bench_report.json" 2>/dev/null ;;
      esac
      echo '```'
      echo
    done
  } > "$LAUNCH/RESULTS.md"
  log "Results: $LAUNCH/RESULTS.md"
  tar czf "$LAUNCH-results.tgz" -C "$(dirname "$LAUNCH")" "$(basename "$LAUNCH")" \
    && log "Bundle: $LAUNCH-results.tgz"
  log "Exit code: $code ($message)"
  rm -f "$LAUNCH/PID"
  exit "$code"
}
trap 'finish 2 "stopped by a signal"' INT TERM HUP

# ---- 0. Host -----------------------------------------------------------------
step "0/4 host checks and Python setup"
for tool in jq git python3 docker nvidia-smi; do
  command -v "$tool" >/dev/null || finish 2 "missing $tool on the host"
done
[[ -n ${HF_TOKEN:-} ]] || finish 2 "HF_TOKEN is not exported"
gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader)
log "GPUs:"; echo "$gpus"
[[ $(grep -c B300 <<<"$gpus") == 8 && $(wc -l <<<"$gpus") == 8 ]] \
  || finish 2 "expected exactly 8 B300 GPUs"
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
[[ $busy == 0 ]] || finish 2 "$busy processes are already using the GPUs"
docker info >/dev/null || finish 2 "docker info failed"
free_gb=$(df -BG --output=avail "$HF_CACHE" | tail -1 | tr -dc 0-9)
[[ $free_gb -ge $MIN_FREE_GB || -d $HF_CACHE/models--moonshotai--Kimi-K3 ]] \
  || finish 2 "only ${free_gb}G free under $HF_CACHE; Kimi K3 needs about ${MIN_FREE_GB}G"
cd "$REPO" || finish 2 "cannot enter $REPO"
if [[ ! -f .venv/.installed ]]; then
  python3 -m venv .venv && .venv/bin/python -m pip install -q -r requirements.txt \
    && .venv/bin/python -m pip check && touch .venv/.installed \
    || finish 2 "Python setup failed"
fi
. .venv/bin/activate
export PYTHONPATH=$REPO PARETON_BENCH_HF_CACHE_DIR=$HF_CACHE PARETON_BENCH_HEALTH_TIMEOUT_S=3600
log "Commit $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
# The engine image is private on GHCR; pull it now so a missing docker login
# fails here rather than after the CPU audit.
image=$(jq -r .bench.baseline_engine_image_digest \
  fixtures/campaigns/sglang_kimi_k3_b300_exgentic/campaign-fields.json)
docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image" \
  || finish 2 "cannot pull $image; run docker login ghcr.io with a read:packages token"

# ---- 1. CPU audit --------------------------------------------------------------
step "1/4 CPU audit of the dataset"
if [[ ! -f $LAUNCH/audit/campaign-fields.json ]]; then
  set_aside "$LAUNCH/audit"
  python -u -m bench.qualify_exgentic --output-dir "$LAUNCH/audit" \
    || finish 1 "CPU audit failed"
fi
jq '{eligible_snapshots_by_tier, distinct_sessions_by_tier}' "$LAUNCH/audit/audit.json"
fewest=$(jq '[.distinct_sessions_by_tier[]] | min' "$LAUNCH/audit/audit.json")
[[ $fewest -ge $MIN_SESSIONS ]] \
  || finish 1 "a tier has only $fewest distinct sessions (needs $MIN_SESSIONS); the tiers in #206 must change"
# The working copy carries any scorer fallback across a resume.
FIELDS=$LAUNCH/campaign-fields.json
[[ -f $FIELDS ]] || cp "$LAUNCH/audit/campaign-fields.json" "$FIELDS"

# ---- 2. Scorer smoke -------------------------------------------------------------
step "2/4 scorer smoke at 128k"
if [[ $(status_of "$LAUNCH/smoke-prep") != prepared_only ]]; then
  set_aside "$LAUNCH/smoke-prep"
  python -u ops/scorer_smoke.py --campaign-fields "$FIELDS" \
    --output-dir "$LAUNCH/smoke-prep" --prepare-only || finish 1 "smoke prepare failed"
fi
run_smoke() {
  set_aside "$1"
  LAST[smoke]=$1
  log "Scorer serve args: $(jq -c .bench.correctness.serve_args "$FIELDS")"
  python -u ops/scorer_smoke.py --campaign-fields "$FIELDS" \
    --prepared "$LAUNCH/smoke-prep" --output-dir "$1"
  log "Smoke exit code: $?"
}
set_fraction() {
  local tmp
  tmp=$(mktemp) && jq --arg f "$1" '.bench.correctness.serve_args as $a
    | .bench.correctness.serve_args[($a | index("--mem-fraction-static")) + 1] = $f' \
    "$FIELDS" > "$tmp" && mv "$tmp" "$FIELDS"
}
if [[ $(status_of "$LAUNCH/smoke.passed") != passed ]]; then
  smoke=$LAUNCH/smoke
  [[ $(status_of "$smoke") == passed ]] || run_smoke "$smoke"
  if [[ $(status_of "$smoke") != passed && $(fraction) == 0.80 ]]; then
    if jq -e '[.failures[]?] | any(test("scorer accepts inputs below"))' \
      "$smoke/summary.json" >/dev/null 2>&1; then
      log "Capacity check failed at 0.80; retrying with 0.84"
      set_fraction 0.84
    elif grep -rqi "out of memory" "$smoke/summary.json" "$smoke/engine_logs" 2>/dev/null; then
      log "Scorer ran out of memory at 0.80; retrying with 0.76"
      set_fraction 0.76
    fi
    if [[ $(fraction) != 0.80 ]]; then
      smoke=$LAUNCH/smoke-$(fraction)
      run_smoke "$smoke"
    fi
  fi
  [[ $(status_of "$smoke") == passed ]] || finish 1 "scorer smoke failed; see $smoke/summary.json"
  ln -sfn "$(basename "$smoke")" "$LAUNCH/smoke.passed"
fi
jq '{capacity, slowest_s, peak_used_mib}' "$LAUNCH/smoke.passed/summary.json"

# ---- 3. Qualification and preflight -------------------------------------------
step "3/4 baseline qualification and preflight (temperatures 0.1 and 1.01, output floor)"
if [[ $(status_of "$LAUNCH/preflight.passed") != passed ]]; then
  pre=$LAUNCH/preflight
  if [[ $(status_of "$pre") != passed ]]; then
    set_aside "$pre"
    LAST[preflight]=$pre
    python -u -m ops.campaign_preflight --campaign-fields "$FIELDS" \
      --output-dir "$pre" --sla-advisory
    code=$?
    log "Preflight exit code: $code"
    # 2 and 3 mean it could not finish; with the baseline outputs saved, only
    # the scorer stage needs to run again.
    if [[ $code -ge 2 && -f $pre/outputs.json && -f $pre/qualification/sampling_rule.json ]]; then
      log "Regrading the saved outputs on one scorer start"
      set_aside "$pre-regrade"
      LAST[preflight]=$pre-regrade
      python -u -m ops.campaign_preflight --campaign-fields "$FIELDS" \
        --output-dir "$pre-regrade" --sla-advisory \
        --qualified-rule "$pre/qualification/sampling_rule.json" --outputs "$pre/outputs.json"
      log "Regrade exit code: $?"
      [[ $(status_of "$pre-regrade") == passed ]] && pre=$pre-regrade
    fi
  fi
  [[ $(status_of "$pre") == passed ]] || finish 1 "preflight failed; see $pre/summary.json"
  ln -sfn "$(basename "$pre")" "$LAUNCH/preflight.passed"
fi
cp "$LAUNCH/preflight.passed/qualification/sampling_rule.json" "$LAUNCH/qualified_sampling_rule.json"
jq '{by_temperature, below_min_output}' "$LAUNCH/preflight.passed/summary.json"
log "Qualified rule sha256: $(sha256sum "$LAUNCH/qualified_sampling_rule.json" | cut -d' ' -f1)"

# ---- 4. Shadow round ---------------------------------------------------------------
step "4/4 shadow round (one production round, baseline as the only candidate)"
if [[ $(status_of "$LAUNCH/shadow.passed") != passed ]]; then
  for attempt in 1 2; do
    shadow=$LAUNCH/shadow
    [[ $attempt == 2 ]] && shadow=$LAUNCH/shadow-2
    set_aside "$shadow"
    LAST[shadow]=$shadow
    python -u -m ops.shadow_round --campaign-fields "$FIELDS" \
      --qualified-rule "$LAUNCH/qualified_sampling_rule.json" --output-dir "$shadow"
    code=$?
    log "Shadow exit code: $code"
    # Only a run that could not finish is worth a second try.
    [[ $code -ge 2 ]] || break
  done
  [[ $(status_of "$shadow") == passed ]] || finish 1 "shadow round failed; see $shadow/summary.json"
  ln -sfn "$(basename "$shadow")" "$LAUNCH/shadow.passed"
fi

finish 0 "all checks passed"
