#!/usr/bin/env bash
# Launch this entire controller under nohup; the runbook supplies redirections.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${PRO6000_RUN_DIR:?Restore the step 1 environment first}"
: "${PRO6000_FIELDS:?}"
: "${PRO6000_ENGINE_REF:?}"
: "${PRO6000_BASELINE_CONTAINER:?}"
: "${PRO6000_QUAL_NET:?}"
: "${PARETON_BENCH_HEALTH_TIMEOUT_S:?}"

# One attempt per run directory. Refuse duplicate starts without replacing logs
# or completion status. Retain the lock after failure as well as success.
mkdir "$PRO6000_RUN_DIR/step2.lock" || {
  echo 'Step 2 already claimed this run directory; inspect its PID, logs and exit code.' >&2
  exit 1
}
printf '%s\n' "$$" > "$PRO6000_RUN_DIR/step2.pid"
finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$PRO6000_RUN_DIR/step2.exit-code.tmp"
  mv "$PRO6000_RUN_DIR/step2.exit-code.tmp" "$PRO6000_RUN_DIR/step2.exit-code"
  if (( rc != 0 )); then
    printf 'Step 2 stopped (exit %s). Preserve this run; inspect logs before retrying.\n' "$rc" >&2
    for log in start-baseline.log startup.log qualification.log; do
      if [[ -f "$PRO6000_RUN_DIR/$log" ]]; then
        printf '\n--- %s ---\n' "$log" >&2
        tail -n 60 "$PRO6000_RUN_DIR/$log" >&2 || true
      fi
    done
  else
    echo 'Step 2 completed successfully; review qualification/sampling_rule.json before step 3.'
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo 'Staging weights, verifying the model volume and starting the baseline...'
python -u - <<'PYTHON' 2>&1 | tee "$PRO6000_RUN_DIR/start-baseline.log"
import json, os, subprocess
from pathlib import Path
from bench.schemas import ModelSpec
from bench.weights import stage_weights
from ops.pro6000_model_volume import DockerModelVolume

fields = json.loads(Path(os.environ["PRO6000_FIELDS"]).read_text())
bench = fields["bench"]
model = bench["model"]
assert fields["base_image_digest"] == bench["baseline_engine_image_digest"]
# A run directory owns one unique baseline name; never replace another container.
existing = subprocess.run(
    ["docker", "container", "inspect", os.environ["PRO6000_BASELINE_CONTAINER"]],
    capture_output=True, check=False,
)
if existing.returncode == 0:
    raise RuntimeError("baseline container already exists; inspect it before retrying")
staged = stage_weights(ModelSpec.from_dict(model))
volume = DockerModelVolume(Path(os.environ["PRO6000_RUN_DIR"]))
args = ["--model-path", "/model", "--dtype", model["dtype"],
        "--quantization", model["quantization"], *bench["serve_args"],
        "--host", "0.0.0.0", "--port", "30000"]
try:
    volume.prepare(staged.path, os.environ["PRO6000_ENGINE_REF"])
    subprocess.run([
        "docker", "run", "-d", "--name", os.environ["PRO6000_BASELINE_CONTAINER"],
        "--gpus", "device=0", "--ipc", "host", "--shm-size", "16g",
        "--network", os.environ["PRO6000_QUAL_NET"],
        "-p", "127.0.0.1:30000:30000", "--mount", volume.mount,
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "--entrypoint", "python3", os.environ["PRO6000_ENGINE_REF"],
        "-m", "sglang.launch_server", *args,
    ], check=True)
except BaseException:
    # Failed docker run can leave a created container holding the model volume.
    subprocess.run(["docker", "rm", "-f", os.environ["PRO6000_BASELINE_CONTAINER"]],
                   capture_output=True, check=False)
    volume.close()
    raise
PYTHON
PRO6000_HEALTH_DEADLINE=$((SECONDS + PARETON_BENCH_HEALTH_TIMEOUT_S))
while :; do
  # Fail immediately if startup did not create a running container.
  PRO6000_CONTAINER_STATE=$(docker inspect --format '{{.State.Status}}' "$PRO6000_BASELINE_CONTAINER")
  if [[ "$PRO6000_CONTAINER_STATE" != running ]]; then
    docker logs "$PRO6000_BASELINE_CONTAINER" > "$PRO6000_RUN_DIR/startup.log" 2>&1 || true
    printf 'Baseline is %s; inspect startup.log before retrying.\n' "$PRO6000_CONTAINER_STATE" >&2
    exit 1
  fi
  if curl -fsS --connect-timeout 2 --max-time 5 http://127.0.0.1:30000/v1/models \
      > "$PRO6000_RUN_DIR/models.json" 2> "$PRO6000_RUN_DIR/health-error.log"; then
    break
  fi
  if (( SECONDS >= PRO6000_HEALTH_DEADLINE )); then
    docker logs "$PRO6000_BASELINE_CONTAINER" > "$PRO6000_RUN_DIR/startup.log" 2>&1 || true
    echo 'Baseline health timeout; inspect startup.log and health-error.log before retrying.' >&2
    exit 1
  fi
  sleep 5
done
echo 'Baseline healthy; qualifying the source pool...'
python -u -m bench.qualify_longform \
  --campaign-fields "$PRO6000_FIELDS" \
  --base-url http://127.0.0.1:30000 \
  --container "$PRO6000_BASELINE_CONTAINER" --engine-ref "$PRO6000_ENGINE_REF" \
  --output-dir "$PRO6000_RUN_DIR/qualification" \
  --pool-size 64 --repetitions 2 --concurrency 4 --timeout 600 \
  2>&1 | tee "$PRO6000_RUN_DIR/qualification.log"
docker logs "$PRO6000_BASELINE_CONTAINER" > "$PRO6000_RUN_DIR/qualification-container.log" 2>&1
docker stop "$PRO6000_BASELINE_CONTAINER"
docker rm "$PRO6000_BASELINE_CONTAINER"
PRO6000_MODEL_VOLUME=$(jq -er '.name' "$PRO6000_RUN_DIR/model_volume.json")
docker volume rm "$PRO6000_MODEL_VOLUME"
docker network rm "$PRO6000_QUAL_NET"
