#!/usr/bin/env bash
# Overnight A2b: hermetic empty-patch → ghcr.io/pareton-ai/pareton-engine:baseline
#
# Usage (on a ≥32GB amd64 box):
#   export PARETON_GHCR_USERNAME='your-github-user'
#   export PARETON_GHCR_TOKEN='ghp_…'   # write:packages
#   # optional if repo is private:
#   export PARETON_GIT_URL='https://x-access-token:${PARETON_GHCR_TOKEN}@github.com/Pareton-ai/pareton.git'
#   curl -fsSL https://raw.githubusercontent.com/Pareton-ai/pareton/main/ops/a2b-build.sh -o a2b-build.sh
#   # or scp this file up, then:
#   chmod +x a2b-build.sh
#   tmux new -s a2b './a2b-build.sh'
#
# Detach: Ctrl-b d. Reattach: tmux attach -t a2b
# Logs: /tmp/pareton-build-*/build.log
set -euo pipefail

REPO_DIR="${PARETON_REPO_DIR:-$HOME/pareton}"
GIT_URL="${PARETON_GIT_URL:-https://github.com/Pareton-ai/pareton.git}"
BASE="${BASE:-ghcr.io/pareton-ai/pareton-baseline@sha256:72b601e4314fa3c5e522e814305fad3a10f06eb174a5785e2729e655cb490986}"
ENGINE_REF="${ENGINE_REF:-ghcr.io/pareton-ai/pareton-engine:baseline}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-ee0da84ab9e04ac7610e28580af62c365e898389}"
# 32GB + MAX_JOBS=1 often needs 4–8h; default CLI timeout is 7200s.
export PARETON_BUILD_TIMEOUT_S="${PARETON_BUILD_TIMEOUT_S:-28800}"

need_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: set $name" >&2
    exit 1
  fi
}

need_env PARETON_GHCR_USERNAME
need_env PARETON_GHCR_TOKEN

echo "==> apt packages"
sudo apt-get update
sudo apt-get install -y docker.io git python3-venv python3-pip

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

echo "==> ensure MAX_JOBS=1 in hermetic Dockerfile (noop if already on main)"
python3 - <<'PY'
from pathlib import Path

p = Path("builder/hermetic.py")
t = p.read_text()
if "ENV MAX_JOBS=" in t:
    print("MAX_JOBS already present")
    raise SystemExit(0)
needle = '        "FROM ${BASE_IMAGE}",\n        "WORKDIR /src",'
insert = """        "FROM ${BASE_IMAGE}",
        "ENV MAX_JOBS=1",
        "ENV CMAKE_BUILD_PARALLEL_LEVEL=1",
        "ENV NVCC_THREADS=1",
        "WORKDIR /src","""
if needle not in t:
    raise SystemExit("FAIL: unexpected hermetic.py shape; patch manually")
p.write_text(t.replace(needle, insert, 1))
print("ok" if "ENV MAX_JOBS=" in p.read_text() else "FAIL")
PY

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

echo "==> A2b build (timeout=${PARETON_BUILD_TIMEOUT_S}s). Walk away."
echo "    progress: ls -lt /tmp/pareton-build-*/build.log | head -1"
with_docker_group python -m builder \
  --baseline-repo "$VLLM_REPO" \
  --baseline-commit "$VLLM_COMMIT" \
  --base-image "$BASE" \
  --image-ref "$ENGINE_REF" \
  --empty-patch \
  --push

echo "==> done. Capture digest:"
echo "    docker pull $ENGINE_REF"
echo "    docker inspect --format='{{index .RepoDigests 0}}' $ENGINE_REF"
