#!/usr/bin/env bash
# Hermetic empty-patch baseline build using the prepared Compose CLI image.
# Set BASE to an A2 RepoDigest and TORCH_CUDA_ARCH_LIST to the target GPU arch.
# Registry credentials may come from .env or exported PARETON_GHCR_* variables.
# Usage: bash ops/a2b-build.sh [--detach]
# With --detach, follow the returned container ID using docker logs -f ID.
# Build logs persist in PARETON_HOST_BUILD_LOG_DIR; Vector also collects stdout.
set -euo pipefail

if [[ "${1:-}" != --in-container ]]; then
  args=(--rm --no-deps)
  if [[ "${1:-}" == --detach && $# == 1 ]]; then
    args+=(--detach)
  elif [[ $# != 0 ]]; then
    echo "usage: bash ops/a2b-build.sh [--detach]" >&2
    exit 2
  fi
  cd "${PARETON_REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
  # Pass explicit nonempty shell overrides without replacing .env credentials.
  for name in BASE TORCH_CUDA_ARCH_LIST ENGINE_REF VLLM_REPO VLLM_COMMIT \
    PARETON_BUILD_TIMEOUT_S PARETON_BUILD_MAX_JOBS \
    PARETON_GHCR_USERNAME PARETON_GHCR_TOKEN; do
    if [[ -n "${!name:-}" ]]; then
      export "$name"
      args+=(-e "$name")
    fi
  done
  exec docker compose run "${args[@]}" cli \
    bash ops/a2b-build.sh --in-container
fi

ENGINE_REF="${ENGINE_REF:-ghcr.io/pareton-ai/pareton-engine:baseline}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-ee0da84ab9e04ac7610e28580af62c365e898389}"
export PARETON_BUILD_TIMEOUT_S="${PARETON_BUILD_TIMEOUT_S:-28800}"
export PARETON_BUILD_MAX_JOBS="${PARETON_BUILD_MAX_JOBS:-2}"

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
need_env TORCH_CUDA_ARCH_LIST
if [[ "$BASE" != *@sha256:* ]]; then
  echo "error: BASE must be a RepoDigest (...@sha256:...), got: $BASE" >&2
  exit 1
fi

python -m builder.preflight
echo "==> docker login ghcr.io"
printf '%s' "$PARETON_GHCR_TOKEN" | docker login ghcr.io \
  -u "$PARETON_GHCR_USERNAME" --password-stdin
echo "==> pull A2 base $BASE"
docker pull "$BASE"
echo "==> A2 smoke (CUDA runtime imports)"
docker run --rm --entrypoint python "$BASE" -c \
  "import flashinfer, torchvision, torchaudio, numba; print('a2-ok', flashinfer.__version__, torchvision.__version__)"

work_root=$(mktemp -d "${PARETON_WORK_DIR:-/var/lib/pareton/work}/a2b-XXXXXX")
trap 'rmdir "$work_root" 2>/dev/null || true' EXIT
echo "==> A2b build (timeout=${PARETON_BUILD_TIMEOUT_S}s max_jobs=${PARETON_BUILD_MAX_JOBS} cuda_arch=${TORCH_CUDA_ARCH_LIST})"
echo "    build logs: ${PARETON_BUILD_LOG_DIR:-/var/log/pareton/builds}"
python -m builder \
  --baseline-repo "$VLLM_REPO" \
  --baseline-commit "$VLLM_COMMIT" \
  --base-image "$BASE" \
  --image-ref "$ENGINE_REF" \
  --work-root "$work_root" \
  --empty-patch \
  --torch-cuda-arch-list "$TORCH_CUDA_ARCH_LIST" \
  --push

echo "==> engine smoke (CPU; does not replace GPU verification)"
docker run --rm --entrypoint python "$ENGINE_REF" -c \
  "import flashinfer, torchvision, torchaudio, vllm.entrypoints.openai.api_server; print('engine-ok')"
echo "==> published image digest"
docker pull "$ENGINE_REF"
docker inspect --format='{{index .RepoDigests 0}}' "$ENGINE_REF"
