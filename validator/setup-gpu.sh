#!/usr/bin/env bash
# validator/setup.sh
#
# Single-machine validator host setup for Ubuntu 22.04 / 24.04 (GPU pod, e.g. Targon DinD / Lium).
# Installs Docker CE, NVIDIA Container Toolkit, Python venv, model weights, PG19 prefetch,
# and pulls the vLLM baseline image.
#
# Optional: export HF_TOKEN for huggingface-cli login (large model download).
#
# Usage:
#   bash validator/setup.sh          # full setup
#   bash validator/setup.sh --pull   # git pull + pip install only (no system installs, downloads, or Docker pulls)

set -euo pipefail

# -- config (same layout as legacy setup-cpu.sh) --
REPO_URL="https://github.com/latent-to/cacheon.git"
CACHEON_BRANCH="${CACHEON_BRANCH:-main}"
BASE="$HOME"
REPO_DIR="$BASE/cacheon"
VENV_DIR="$BASE/venv-cacheon"
MODEL_DIR="/workspace/models/Qwen2.5-72B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

PULL_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--pull" ]] && PULL_ONLY=true
done

if [[ "$PULL_ONLY" == true ]]; then
  echo "=== Pull-only (--pull): git pull + pip install ==="
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "ERROR: $REPO_DIR is not a git clone. Run full setup first (without --pull)."
    exit 1
  fi
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: venv missing at $VENV_DIR. Run full setup first (without --pull)."
    exit 1
  fi
  git -C "$REPO_DIR" pull
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -r "$REPO_DIR/validator/requirements-gpu.txt"
  echo "=== Pull-only run complete ==="
  exit 0
fi

# -- DNS (some DinD pods have flaky resolvers for huggingface.co) --
if ! getent hosts huggingface.co >/dev/null 2>&1; then
  echo ""
  echo "=== DNS workaround ==="
  echo "huggingface.co lookup failed; prepending public nameservers"
  if [[ -f /etc/resolv.conf ]] && ! grep -q '^nameserver 1\.1\.1\.1' /etc/resolv.conf; then
    cp /etc/resolv.conf /etc/resolv.conf.cacheon.bak
    { echo "nameserver 1.1.1.1"; echo "nameserver 8.8.8.8"; cat /etc/resolv.conf.cacheon.bak; } > /etc/resolv.conf
  elif [[ ! -f /etc/resolv.conf ]]; then
    printf 'nameserver 1.1.1.1\nnameserver 8.8.8.8\n' > /etc/resolv.conf
  fi
fi

# -- system deps --
echo ""
echo "=== System dependencies ==="
apt-get update -q
apt-get install -y --no-install-recommends \
  git curl ca-certificates tmux python3-venv jq gnupg

# -- Docker CE --
echo ""
echo "=== Docker CE ==="
if docker --version >/dev/null 2>&1; then
  echo "Docker already installed: $(docker --version)"
else
  echo "Installing Docker via get.docker.com..."
  curl -fsSL https://get.docker.com | sh
fi

# -- NVIDIA Container Toolkit --
echo ""
echo "=== NVIDIA Container Toolkit ==="
if nvidia-ctk --version >/dev/null 2>&1; then
  echo "nvidia-ctk already present: $(nvidia-ctk --version | head -n 1)"
else
  echo "Installing nvidia-container-toolkit from NVIDIA repository..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  apt-get update -q
  apt-get install -y --no-install-recommends nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker

mkdir -p /etc/cdi
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
echo "CDI specs generated at /etc/cdi/nvidia.yaml"

# DinD vs bare VM: Lium/Targon need nested fixes; Shadeform bare VMs need no-cgroups=false.
CACHEON_NESTED_DIN="${CACHEON_NESTED_DIN:-0}"

if [[ "$CACHEON_NESTED_DIN" == "1" ]]; then
  echo "Nested DinD mode (CACHEON_NESTED_DIN=1): enabling no-cgroups, runc 1.1.x"
  if grep -q '^#no-cgroups = false' /etc/nvidia-container-runtime/config.toml 2>/dev/null; then
    sed -i 's/^#no-cgroups = false/no-cgroups = true/' /etc/nvidia-container-runtime/config.toml
  elif ! grep -q '^no-cgroups = true' /etc/nvidia-container-runtime/config.toml 2>/dev/null; then
    echo "WARNING: could not find no-cgroups line in config.toml; GPU passthrough may fail in nested containers."
  fi
  RUNC_VER="$(runc --version 2>/dev/null | head -1 | grep -oP '[\d]+\.[\d]+' | head -1 || true)"
  if [[ "$RUNC_VER" == "1.2" || "$RUNC_VER" == "1.3" ]]; then
    echo "Downgrading runc from $RUNC_VER.x to 1.1.15 for nested container compatibility..."
    curl -fsSL -o /usr/sbin/runc https://github.com/opencontainers/runc/releases/download/v1.1.15/runc.amd64
    chmod +x /usr/sbin/runc
    echo "runc: $(runc --version | head -1)"
  fi
else
  echo "Bare VM mode (CACHEON_NESTED_DIN=0): keeping no-cgroups disabled"
  if grep -q '^no-cgroups = true' /etc/nvidia-container-runtime/config.toml 2>/dev/null; then
    sed -i 's/^no-cgroups = true/no-cgroups = false/' /etc/nvidia-container-runtime/config.toml
    nvidia-ctk runtime configure --runtime=docker
  fi
fi

# -- Docker data-root on /workspace (large block volume) --
echo ""
echo "=== Docker data-root ==="
DOCKER_DATA_ROOT="/workspace/docker"
if ! mountpoint -q /workspace 2>/dev/null && ! df -Pk /workspace >/dev/null 2>&1; then
  echo "ERROR: /workspace is not mounted. Attach the model volume before running setup."
  exit 1
fi
mkdir -p "$DOCKER_DATA_ROOT" /etc/docker
# max-concurrent-downloads=1 serializes layer extraction; concurrent extraction
# under nested overlay2 (DinD) is the trigger for "failed to register layer: file exists".
if [[ -f /etc/docker/daemon.json ]]; then
  jq '. + {"data-root": "'"$DOCKER_DATA_ROOT"'", "max-concurrent-downloads": 1}' /etc/docker/daemon.json \
    > /etc/docker/daemon.json.tmp
else
  echo '{"data-root":"'"$DOCKER_DATA_ROOT"'","max-concurrent-downloads":1}' > /etc/docker/daemon.json.tmp
fi
mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
echo "Configured data-root=$DOCKER_DATA_ROOT"

_need_docker_restart() {
  if ! docker info >/dev/null 2>&1; then
    return 0
  fi
  local root
  root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  [[ "$root" != "$DOCKER_DATA_ROOT" ]]
}

if _need_docker_restart; then
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active docker >/dev/null 2>&1; then
    systemctl stop docker
  elif pgrep -x dockerd >/dev/null 2>&1; then
    pkill -x dockerd && sleep 2
  fi
  echo ""
  echo "=== Starting dockerd ==="
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files docker.service >/dev/null 2>&1; then
    systemctl start docker
  else
    nohup dockerd > /var/log/dockerd.log 2>&1 &
    sleep 5
  fi
else
  echo "dockerd already using $DOCKER_DATA_ROOT, skipping restart"
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: dockerd failed to start. Check /var/log/dockerd.log"
  exit 1
fi
echo "dockerd running (PID $(pgrep -x dockerd || echo systemctl))"
docker info 2>/dev/null | grep -i "Docker Root Dir" || true

# -- clone or update repo --
echo ""
echo "=== Repo ==="
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "Repo exists, pulling latest..."
  git -C "$REPO_DIR" pull
else
  echo "Cloning into $REPO_DIR (branch: $CACHEON_BRANCH)"
  git clone -b "$CACHEON_BRANCH" "$REPO_URL" "$REPO_DIR"
fi

# -- Python venv --
echo ""
echo "=== Python dependencies ==="
echo "Python: $(python3 --version)"
if [[ ! -x "$VENV_DIR/bin/python" ]] || [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Creating venv at $VENV_DIR..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
export PATH="$VENV_DIR/bin:$PATH"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/validator/requirements-gpu.txt"

# Hugging Face cache (all HF operations below use this)
export HF_HOME="/workspace/.cache/huggingface"
mkdir -p "$HF_HOME"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo ""
  echo "=== Hugging Face CLI (HF_TOKEN set) ==="
  "$VENV_DIR/bin/hf" auth login --token "$HF_TOKEN"
fi

# -- disk space (before large downloads) --
echo ""
echo "=== Disk space ==="
df -h / /workspace "$DOCKER_DATA_ROOT" 2>/dev/null || df -h / /workspace 2>/dev/null || true
avail_kb="$(df -Pk /workspace 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -z "$avail_kb" ]] || ! [[ "$avail_kb" =~ ^[0-9]+$ ]]; then
  echo "WARNING: could not read free space on /workspace (df). Continuing."
else
  avail_gb=$((avail_kb / 1024 / 1024))
  if ((avail_gb < 100)); then
    echo "ERROR: need at least 100 GB free on the /workspace filesystem, found ${avail_gb} GB."
    exit 1
  fi
  if ((avail_gb < 200)); then
    echo "WARNING: only ${avail_gb} GB free on /workspace; recommend 200+ GB for model, dataset cache, and Docker layers."
  else
    echo "OK: ${avail_gb} GB free on /workspace."
  fi
fi

VLLM_COMPILE_CACHE="${CACHEON_VLLM_CACHE_DIR:-/workspace/vllm-cache}"
if [[ -n "$VLLM_COMPILE_CACHE" ]]; then
  mkdir -p "$VLLM_COMPILE_CACHE"
fi

# -- model weights --
echo ""
echo "=== Model weights ($MODEL_NAME) ==="
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/tokenizer.json" ] && \
   ls "$MODEL_DIR"/model-*.safetensors >/dev/null 2>&1; then
  echo "Model already present at $MODEL_DIR, skipping download."
else
  MAX_DL_ATTEMPTS=2
  DL_TIMEOUT=600
  for attempt in $(seq 1 $MAX_DL_ATTEMPTS); do
    echo "Download attempt $attempt/$MAX_DL_ATTEMPTS (timeout=${DL_TIMEOUT}s)..."
    if timeout "$DL_TIMEOUT" "$VENV_DIR/bin/hf" download "$MODEL_NAME" --local-dir "$MODEL_DIR"; then
      break
    fi
    if [ "$attempt" -eq "$MAX_DL_ATTEMPTS" ]; then
      echo "ERROR: model download failed after $MAX_DL_ATTEMPTS attempts"
      exit 1
    fi
    echo "Download interrupted, retrying in 10s..."
    sleep 10
  done
fi

# -- PG19 (matches validator/prompts.py DATASET_NAME) --
echo ""
echo "=== PG19 dataset ==="
PG19_CACHE="$HF_HOME/datasets/emozilla___pg19"
if [ -d "$PG19_CACHE" ] && find "$PG19_CACHE" -name "*.arrow" -print -quit 2>/dev/null | grep -q .; then
  echo "PG19 already cached at $PG19_CACHE, skipping."
else
  "$VENV_DIR/bin/python" -c "from datasets import load_dataset; load_dataset('emozilla/pg19', split='train'); print('PG19 cached')"
fi

BASELINE_IMAGE="${CACHEON_BASELINE_IMAGE:-vllm/vllm-openai:v0.22.0}"
SCORING_IMAGE="${CACHEON_SCORING_IMAGE:-vllm/vllm-openai:v0.9.2}"

# A failed pull leaves a partially-registered layer (orphaned
# layerdb/sha256/<hash> dir) behind. On ephemeral auto-rent volumes the whole
# volume is deleted after eval; on long-lived pods, broken layer state on
# /workspace can poison later Docker pulls until the data-root is reset.
# Removing only layerdb/tmp does not clear the orphaned destination dir, so the
# only reliable recovery is to wipe the whole data-root and restart dockerd.
# models/ and .cache/ live elsewhere on /workspace and are untouched.
_reset_docker_store() {
  echo "Resetting Docker layer store at $DOCKER_DATA_ROOT (clearing partial/broken layers)..."
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active docker >/dev/null 2>&1; then
    systemctl stop docker.socket docker || true
  elif pgrep -x dockerd >/dev/null 2>&1; then
    pkill -x dockerd || true
  fi
  sleep 3
  rm -rf "${DOCKER_DATA_ROOT:?}/"* 2>/dev/null || true
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files docker.service >/dev/null 2>&1; then
    systemctl start docker
  else
    nohup dockerd > /var/log/dockerd.log 2>&1 &
  fi
  local i
  for i in $(seq 1 15); do
    docker info >/dev/null 2>&1 && break
    sleep 1
  done
}

ensure_vllm_images() {
  local refs=("$BASELINE_IMAGE" "$SCORING_IMAGE")
  if docker image inspect "$BASELINE_IMAGE" >/dev/null 2>&1 \
    && docker image inspect "$SCORING_IMAGE" >/dev/null 2>&1; then
    echo "vLLM images already present, skipping pull."
    return 0
  fi
  local attempt ref
  for attempt in 1 2; do
    local failed=0
    for ref in "${refs[@]}"; do
      if docker image inspect "$ref" >/dev/null 2>&1; then
        echo "$ref already present, skipping pull."
        continue
      fi
      echo "Pulling $ref (attempt $attempt/2)..."
      if ! docker pull "$ref" 2>&1; then
        failed=1
        break
      fi
    done
    if [[ "$failed" -eq 0 ]]; then
      return 0
    fi
    if [[ "$attempt" -lt 2 ]]; then
      echo "Pull failed; resetting layer store and retrying both images in 15s..."
      _reset_docker_store
      sleep 15
    fi
  done
  echo "ERROR: failed to pull vLLM images after 2 attempts"
  return 1
}

# -- vLLM eval images (generation baseline + teacher-forcing) --
echo ""
echo "=== vLLM Docker images ==="
ensure_vllm_images
REPO_DIGEST="$(docker image inspect "$BASELINE_IMAGE" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
DIGEST=""
if [[ -n "$REPO_DIGEST" ]] && [[ "$REPO_DIGEST" == *"@"* ]]; then
  DIGEST="${REPO_DIGEST##*@}"
else
  echo "WARNING: could not read RepoDigests for $BASELINE_IMAGE; set CACHEON_BASELINE_DIGEST manually."
fi
echo ""
echo "# Add to .env (with CACHEON_MODEL_VOLUME, CACHEON_GPU_COUNT=8)"
echo "CACHEON_BASELINE_IMAGE=$BASELINE_IMAGE"
echo "CACHEON_BASELINE_DIGEST=${DIGEST:-sha256:REPLACE_WITH_ACTUAL_DIGEST}"
echo "CACHEON_SCORING_IMAGE=$SCORING_IMAGE"

# -- verification --
echo ""
echo "=== Verification ==="
nvidia-smi
docker --version
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi; then
  echo "GPU passthrough in nested container: OK"
else
  echo "WARNING: nested GPU passthrough check failed (non-fatal; eval may still work)"
fi
"$VENV_DIR/bin/python" -c "import datasets; print('python deps ok')"
test -d "$MODEL_DIR"

echo ""
echo "=== Setup complete ==="
echo "Next: copy validator/.env.example to your .env, set wallet vars, and paste CACHEON_BASELINE_DIGEST from above."
echo "Docs: https://cacheon.ai/docs/validators/overview"
echo "Run the validator: https://cacheon.ai/docs/architecture/validator"
