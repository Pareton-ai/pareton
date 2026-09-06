#!/usr/bin/env bash
# Run on the GPU host using the bundled measurement helpers.
# Usage: bash reproduce-pareton.sh [all|bootstrap|serve|run|stop] [--existing-container NAME] [--run-name NAME]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/qwen38-27b-fp8-pareton.recipe"
STEP=all
EXISTING=false
CONTAINER=pareton-lmx-vllm
RUN_NAME="$RUN_NAME-$(date -u +%Y%m%dT%H%M%SZ)-$$"
while [ "$#" -gt 0 ]; do
  case "$1" in
    all|bootstrap|serve|run|stop) STEP=$1; shift ;;
    --existing-container) CONTAINER=${2:?container name required}; EXISTING=true; shift 2 ;;
    --run-name) RUN_NAME=${2:?run name required}; shift 2 ;;
    -h|--help)
      echo 'Usage: reproduce-pareton.sh [all|bootstrap|serve|run|stop] [--existing-container NAME] [--run-name NAME]'
      echo 'Default: install the benchmark client, start the recipe image, and benchmark.'
      echo '--existing-container: benchmark the named running container; never pull, start, stop, or replace it.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo 'Invalid run name' >&2; exit 2; }
if "$EXISTING" && { [ "$STEP" = serve ] || [ "$STEP" = stop ]; }; then
  echo 'serve and stop are unavailable with --existing-container' >&2; exit 2
fi
REFERENCE_COMMIT=e9792ebc2f60bdd9275f02f53f1d3eec34b86314
LMX_VERSION=v0.1.39
L=/workspace/lmx
LOGS=/workspace/logs
RUNS=/workspace/runs
VOLUME=pareton-hf-cache
export PATH=/workspace/bin:$PATH
mkdir -p "$L" "$LOGS" "$RUNS" /workspace/bin

# Use the same Docker identity for pulls, volumes, launch, and inspection.
DOCKER=(docker)
if [ "$(id -u)" -ne 0 ]; then DOCKER=(sudo docker); fi

stage_helpers() {
  for file in run_speed.sh power_sampler.sh power_window.py capture_meta.py submit_prep.py; do
    cp "$HERE/helpers/$file" "$L/$file"
  done
  cp "$HERE/helpers/prompt_reasoning-v1.txt" "$L/prompt_reasoning-v1.txt"
}

validate_hardware() {
  # These instances use H200 SXM; nvidia-smi may report only NVIDIA H200.
  python3 - "$L/hardware.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
hardware = json.loads(path.read_text())
if hardware.get('gpuName') == 'NVIDIA H200':
    hardware['gpuName'] = 'NVIDIA H200 SXM'
    path.write_text(json.dumps(hardware, indent=2) + '\n')
    print('Hardware name: NVIDIA H200 -> NVIDIA H200 SXM (assumed SXM)')
PY
  lmx hardware validate "$L/hardware.json"
}

bootstrap() {
  stage_helpers
  # Install only the benchmark client. The image supplies vLLM and its toolchain.
  local tmp
  tmp=$(mktemp -d)
  curl -fsSL "https://github.com/LottoLottoLotto/localmaxxing-cli/releases/download/$LMX_VERSION/lmx-linux-amd64.tar.gz" -o "$tmp/lmx-linux-amd64.tar.gz"
  curl -fsSL "https://github.com/LottoLottoLotto/localmaxxing-cli/releases/download/$LMX_VERSION/checksums.txt" -o "$tmp/checksums.txt"
  (cd "$tmp" && grep ' lmx-linux-amd64.tar.gz$' checksums.txt | sha256sum -c -)
  tar -xzf "$tmp/lmx-linux-amd64.tar.gz" -C /workspace/bin lmx
  chmod +x /workspace/bin/lmx
  rm -r "$tmp"
  lmx --version
  if ! "$EXISTING"; then
    "${DOCKER[@]}" pull "$IMAGE"
    "${DOCKER[@]}" volume create "$VOLUME" >/dev/null
  fi
  lmx hardware --out "$L/hardware.json"
  validate_hardware
}

serve() {
  # Fail on an existing name or occupied port, instead of replacing another run.
  if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    # Preserve only this variable through sudo; Docker reads it from its env.
    if [ "$(id -u)" -ne 0 ]; then DOCKER=(sudo --preserve-env=HF_TOKEN docker); fi
  fi
  local cmd=("${DOCKER[@]}" run -d --name "$CONTAINER" --gpus 'device=0'
    --ipc=host -p 127.0.0.1:8000:8000
    --mount "type=volume,source=$VOLUME,target=/hf-cache" -e HF_HOME=/hf-cache)
  if [ -n "${HF_TOKEN:-}" ]; then cmd+=(-e HF_TOKEN); fi
  cmd+=("$IMAGE" "${SERVE_ARGS[@]}")
  printf '%q ' "${cmd[@]}" > "$L/serve_current.txt"
  printf '\n' >> "$L/serve_current.txt"
  "${cmd[@]}"
  local deadline=$((SECONDS + 2400))
  while true; do
    if [ "$("${DOCKER[@]}" inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]; then
      "${DOCKER[@]}" logs "$CONTAINER" >&2; return 1
    fi
    if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
    if (( SECONDS >= deadline )); then
      echo 'Server readiness timed out; inspect docker logs pareton-lmx-vllm.' >&2; return 1
    fi
    sleep 5
  done
}

run() {
  stage_helpers
  local actual_image expected_image version notes sampler_pid rc
  actual_image=$("${DOCKER[@]}" inspect -f '{{.Image}}' "$CONTAINER")
  expected_image=$("${DOCKER[@]}" image inspect -f '{{.Id}}' "$IMAGE")
  [ "$actual_image" = "$expected_image" ] || { echo 'Container image mismatch' >&2; return 1; }
  [ "$("${DOCKER[@]}" inspect -f '{{.State.Running}}' "$CONTAINER")" = true ]
  "${DOCKER[@]}" inspect -f '{{json .Config.Cmd}}' "$CONTAINER" | python3 -c \
    'import json,sys; actual=json.load(sys.stdin); sys.exit(0 if actual == sys.argv[1:] else "Container arguments differ from the recipe")' "${SERVE_ARGS[@]}"
  "${DOCKER[@]}" inspect -f '{{json .NetworkSettings.Ports}}' "$CONTAINER" | python3 -c \
    'import json,sys; ports=json.load(sys.stdin) or {}; bindings=ports.get("8000/tcp") or []; sys.exit(0 if any(p["HostPort"] == "8000" and p["HostIp"] in ("127.0.0.1", "0.0.0.0") for p in bindings) else "Container must publish port 8000 on host localhost:8000")'
  curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null
  validate_hardware
  version=$("${DOCKER[@]}" exec "$CONTAINER" python -c 'import vllm; print(vllm.__version__)')
  notes="Custom Pareton image $IMAGE; model revision $REVISION. LocalMaxxing reasoning-v1, concurrency $CONCURRENCY, max-num-seqs $MAX_NUM_SEQS, 2 warmups, $ITERATIONS timed iterations, max_tokens $MAX_TOKENS."
  # Prevent an old successful run from masking a failed benchmark.
  mkdir "$RUNS/$RUN_NAME"
  printf '%s\n' "$IMAGE" > "$RUNS/$RUN_NAME/image-ref.txt"
  printf '%s\n' "$REFERENCE_COMMIT" > "$RUNS/$RUN_NAME/reference-commit.txt"
  "${DOCKER[@]}" inspect -f '{{json .Config.Cmd}}' "$CONTAINER" > "$RUNS/$RUN_NAME/server-args.json"
  # Capture the actual process command, including for manually launched containers.
  # Inspect only entrypoint/arguments, never the environment or registry credentials.
  "${DOCKER[@]}" inspect -f '{{json .Config.Entrypoint}}' "$CONTAINER" > "$RUNS/$RUN_NAME/server-entrypoint.json"
  python3 - "$RUNS/$RUN_NAME" <<'PY'
import json, pathlib, shlex, sys
directory = pathlib.Path(sys.argv[1])
entrypoint = json.loads((directory / 'server-entrypoint.json').read_text()) or []
arguments = json.loads((directory / 'server-args.json').read_text()) or []
(directory / 'serve-command.txt').write_text(shlex.join(entrypoint + arguments) + '\n')
PY
  "${DOCKER[@]}" exec "$CONTAINER" python -c 'import json,vllm,torch; print(json.dumps(dict(vllm=vllm.__version__,vllm_file=vllm.__file__,torch=torch.__version__,cuda=torch.version.cuda)))' > "$RUNS/$RUN_NAME/engine-build.json"
  touch "$LOGS/power_sampler.csv"
  bash "$L/power_sampler.sh" "$LOGS/power_sampler.csv" &
  sampler_pid=$!
  trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT
  # Retain the client status in the log alongside pipeline failure handling.
  rc=0
  bash "$L/run_speed.sh" "$RUN_NAME" vllm "$HF_ID" "$HF_ID" FP8 "$version" \
    "$CONCURRENCY" "$ITERATIONS" "$MAX_TOKENS" "$notes" "$RUNS/$RUN_NAME/serve-command.txt" \
    2>&1 | tee "$LOGS/run_$RUN_NAME.log" || rc=$?
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  trap - EXIT
  [ "$rc" -eq 0 ] && grep -qx 'lmx rc=0' "$LOGS/run_$RUN_NAME.log" || return 1
  for file in run.json meta.json power_window.json companion_response.json; do
    [ -s "$RUNS/$RUN_NAME/$file" ] || { echo "Missing result: $file" >&2; return 1; }
  done
  python3 - "$RUNS/$RUN_NAME/run.json" "$REVISION" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
data['modelRevision'] = sys.argv[2]
path.write_text(json.dumps(data, indent=2) + '\n')
PY
  python3 "$L/submit_prep.py" "$RUNS/$RUN_NAME"
  "${DOCKER[@]}" logs "$CONTAINER" > "$RUNS/$RUN_NAME/server.log" 2>&1
  echo "Results: $RUNS/$RUN_NAME/payload.json"
}

stop() {
  "${DOCKER[@]}" logs "$CONTAINER" > "$LOGS/pareton-lmx-server.log" 2>&1 || true
  "${DOCKER[@]}" rm -f "$CONTAINER"
}

case "$STEP" in
  bootstrap) bootstrap ;;
  serve) serve ;;
  run) run ;;
  stop) stop ;;
  all) bootstrap; if ! "$EXISTING"; then serve; fi; run ;;
  *) echo "Unknown step: $STEP" >&2; exit 2 ;;
esac
