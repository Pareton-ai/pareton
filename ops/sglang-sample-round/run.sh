#!/usr/bin/env bash
set -Eeuo pipefail
phase=setup
log() {
    printf '[%s] [sample-round] [%ss] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SECONDS" "$*"
}
trap 'log "ERROR: phase=$phase line=$LINENO exit=$?" >&2' ERR
script_root="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_root/../.." && pwd)"
mkdir -p "${1:-/workspace/pareton-sample-round-nvfp4}"
run_root="$(cd "${1:-/workspace/pareton-sample-round-nvfp4}" && pwd)"
mkdir -p /opt/pareton
exec 9>/opt/pareton/.static-host.lock
log "Acquiring static-host lock; output directory: $run_root"
flock -n -E 75 9
log "Static-host lock acquired"

log "Creating Python environment and installing dependencies"
python3 -m venv "$run_root/venv"
source "$run_root/venv/bin/activate"
python -m pip install --progress-bar off -r "$repo_root/requirements.txt"
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PARETON_BENCH_HF_CACHE_DIR=/workspace/hf-cache
export PARETON_BENCH_ENGINE_CACHE_DIR=/workspace/engine-cache
export PARETON_BUILD_MAX_JOBS=6
export PARETON_BUILD_TIMEOUT_S=172800
export PARETON_BENCH_HEALTH_TIMEOUT_S=3600
cd "$repo_root"
log "Dependencies ready; weights cache=$PARETON_BENCH_HF_CACHE_DIR; engine cache=$PARETON_BENCH_ENGINE_CACHE_DIR"

# Same host lock as static-SSH orchestration; only Pareton bench leftovers.
phase=preflight
log "Removing stale Pareton containers/networks and checking for GPU compute processes"
python -c 'import logging; logging.basicConfig(level=logging.INFO); from gpu.static_host import cleanup_containers, check_idle_gpu; print(f"Removed {cleanup_containers()} containers", flush=True); check_idle_gpu(); print("GPU compute processes: none", flush=True)'
cleanup() {
    saved_status=$?
    trap - EXIT
    log "Cleanup starting; previous phase=$phase exit=$saved_status"
    phase=cleanup
    python -c 'import logging; logging.basicConfig(level=logging.INFO); from gpu.static_host import cleanup_containers, check_idle_gpu; print(f"Removed {cleanup_containers()} containers", flush=True); check_idle_gpu(); print("GPU compute processes: none", flush=True)' || saved_status=1
    log "Finished with exit=$saved_status; logs and reports retained at $run_root"
    exit "$saved_status"
}
trap cleanup EXIT

baseline='ghcr.io/pareton-ai/pareton-baseline@sha256:43d5d33c2d3f61923d7ff96b8c69b77b8ddee28f749c10bb876ed538169fd431'
log "Checking cached baseline image: $baseline"
docker image inspect "$baseline" >/dev/null
phase=build
log "Building patched SGLang; jobs=$PARETON_BUILD_MAX_JOBS timeout=${PARETON_BUILD_TIMEOUT_S}s; log=$run_root/build.log"
log "A cold native compiler cache can make this step take hours"
python -m builder \
    --engine sglang \
    --baseline-repo https://github.com/sgl-project/sglang.git \
    --baseline-commit 4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc \
    --base-image "$baseline" \
    --patch-file "$script_root/submission.diff" \
    --image-ref pareton-sample:sglang-minimal \
    --no-push --stream-build-logs \
    2>&1 | tee "$run_root/build.log"
log "Candidate build completed"

phase=sampling
log "Preparing sampled workload and benchmark request"
python "$script_root/prepare.py" "$run_root" "${2:-$repo_root/fixtures/campaigns/sglang_qwen38_27b/sampling_rule.json}"
output_dir="$run_root/output-$(date -u +%Y%m%dT%H%M%SZ)"
phase=benchmark
log "Starting baseline, candidate, correctness scorer, and baseline drift replay"
log "Live benchmark logs: $output_dir/harness.log; phase state: $output_dir/phase.json"
python -m bench --request "$run_root/bench_request.json" --output-dir "$output_dir"
log "Benchmark completed; inspect candidate outcomes in $output_dir/bench_report.json"
