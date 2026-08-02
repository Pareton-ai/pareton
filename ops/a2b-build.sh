#!/usr/bin/env bash
# Overnight A2b: hermetic empty-patch → ghcr.io/pareton-ai/pareton-engine:baseline
#
# Usage (on a ≥32GB amd64 box — this recipe targets 4 vCPU / 32 GB):
#   # one-time: 64G swapfile (absorbs cicc peaks)
#   sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile \
#     && sudo mkswap /swapfile && sudo swapon /swapfile
#   export PARETON_GHCR_USERNAME='your-github-user'
#   export PARETON_GHCR_TOKEN='ghp_…'   # write:packages
#   # Rebuild A2 first whenever requirements-runtime.txt changes, then:
#   export BASE="$(docker inspect --format='{{index .RepoDigests 0}}' \
#     ghcr.io/pareton-ai/pareton-baseline:v0)"
#   # Real-box defaults (arch pin does most of the speedup; keep jobs modest):
#   export PARETON_BUILD_MAX_JOBS=2          # ≤ vCPUs and ≤ RAM/3GB on bigger hosts
#   export PARETON_TORCH_CUDA_ARCH_LIST=9.0  # match bench SKU
#   # optional if repo is private:
#   export PARETON_GIT_URL='https://x-access-token:${PARETON_GHCR_TOKEN}@github.com/Pareton-ai/pareton.git'
#   curl -fsSL https://raw.githubusercontent.com/Pareton-ai/pareton/main/ops/a2b-build.sh -o a2b-build.sh
#   # or scp this file up, then:
#   chmod +x a2b-build.sh
#   tmux new -s a2b './a2b-build.sh'
#
# Detach: Ctrl-b d. Reattach: tmux attach -t a2b
# Logs: /tmp/pareton-build-*/build.log
# Do NOT set MAX_JOBS=16 on 32GB — cicc peaks 6–12GB/job and will OOM.
set -euo pipefail

REPO_DIR="${PARETON_REPO_DIR:-$HOME/pareton}"
GIT_URL="${PARETON_GIT_URL:-https://github.com/Pareton-ai/pareton.git}"
# No hardcoded A2 digest: a stale default rebuilds engines FROM pre-cuda.txt
# images after runtime dep fixes land on main.
ENGINE_REF="${ENGINE_REF:-ghcr.io/pareton-ai/pareton-engine:baseline}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-ee0da84ab9e04ac7610e28580af62c365e898389}"
# 32GB + modest jobs often needs several hours; default CLI timeout is 7200s.
export PARETON_BUILD_TIMEOUT_S="${PARETON_BUILD_TIMEOUT_S:-28800}"
export PARETON_BUILD_MAX_JOBS="${PARETON_BUILD_MAX_JOBS:-2}"
export PARETON_TORCH_CUDA_ARCH_LIST="${PARETON_TORCH_CUDA_ARCH_LIST:-9.0}"

need_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: set $name" >&2
    exit 1
  fi
}

need_env PARETON_GHCR_USERNAME
need_env PARETON_GHCR_TOKEN
need_env BASE

if [[ "$BASE" != *@sha256:* ]]; then
  echo "error: BASE must be a RepoDigest (...@sha256:...), got: $BASE" >&2
  echo "  export BASE=\$(docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/pareton-ai/pareton-baseline:v0)" >&2
  exit 1
fi

echo "==> apt packages (docker.io + docker-buildx for BuildKit)"
sudo apt-get update
sudo apt-get install -y docker.io docker-buildx git python3-venv python3-pip

if ! groups | grep -qw docker; then
  echo "==> adding $USER to docker group (active for this script via sg)"
  sudo usermod -aG docker "$USER" || true
fi

# Fresh VPS: usermod does not update this shell's groups. Wrap any docker
# socket use (including python -m builder → docker build/push).
with_docker_group() {
  if docker info >/dev/null 2>&1; then
    "$@"
  else
    sg docker -c "$(printf '%q ' "$@")"
  fi
}

echo "==> clone / update repo at $REPO_DIR"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch --depth 1 origin main
  git -C "$REPO_DIR" checkout -B main origin/main
else
  git clone "$GIT_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> venv + deps"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "==> docker login ghcr.io"
echo "$PARETON_GHCR_TOKEN" | with_docker_group docker login ghcr.io -u "$PARETON_GHCR_USERNAME" --password-stdin

echo "==> pull A2 base $BASE"
with_docker_group docker pull "$BASE"

echo "==> A2 smoke (cuda runtime names used by EngineCore start)"
with_docker_group docker run --rm --entrypoint python "$BASE" -c \
  "import flashinfer, torchvision, torchaudio, numba; print('a2-ok', flashinfer.__version__, torchvision.__version__)"

echo "==> A2b build (timeout=${PARETON_BUILD_TIMEOUT_S}s" \
  "max_jobs=${PARETON_BUILD_MAX_JOBS}" \
  "cuda_arch=${PARETON_TORCH_CUDA_ARCH_LIST}). Walk away."
echo "    progress: ls -lt /tmp/pareton-build-*/build.log | head -1"
with_docker_group python -m builder \
  --baseline-repo "$VLLM_REPO" \
  --baseline-commit "$VLLM_COMMIT" \
  --base-image "$BASE" \
  --image-ref "$ENGINE_REF" \
  --empty-patch \
  --push

echo "==> engine smoke (CPU; does not replace Hopper B7-lite)"
with_docker_group docker run --rm --entrypoint python "$ENGINE_REF" -c \
  "import flashinfer, torchvision, torchaudio, vllm.entrypoints.openai.api_server; print('engine-ok')"

echo "==> done. Capture digest:"
echo "    docker pull $ENGINE_REF"
echo "    docker inspect --format='{{index .RepoDigests 0}}' $ENGINE_REF"
