#!/usr/bin/env bash
# Installed outside /src in the trusted image. Rebuild patched native packages.
set -euo pipefail
cd /src
export CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR=/opt/sglang-rust-target
export CARGO_BUILD_JOBS="${MAX_JOBS:-1}"
export SGLANG_BUILD_RUST_EXTS=all
export CCACHE_LOGFILE=/tmp/sglang-ccache.log
rm -f "$CCACHE_LOGFILE"
if [[ "${CCACHE_READONLY:-}" == 1 ]]; then
  # The immutable baseline carries its trusted cache to fresh validator hosts.
  # The shared host mount stays read-only; misses compile in this image layer.
  test -d /opt/sglang-ccache
  export CCACHE_DIR=/opt/sglang-ccache
  mkdir -p "${CCACHE_TEMPDIR:-/tmp/ccache-tmp}"
fi

# Persistent, private image build directories keep Rust/CMake artifacts available
# to the next image layer. Only ccache is shared, read-only for miner builds.
pip install --no-deps --no-build-isolation -e python/
pip install --no-deps --no-build-isolation --force-reinstall \
  --config-settings=build-dir=/opt/sglang-aot-build \
  --config-settings=cmake.args=-C/opt/sglang-deps/offline.cmake \
  --config-settings="cmake.define.CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  --config-settings=cmake.define.SGL_KERNEL_COMPILE_THREADS=1 \
  --config-settings=cmake.define.SGL_KERNEL_CXX_STANDARD=20 \
  --config-settings=cmake.define.ENABLE_BELOW_SM90=OFF \
  --config-settings=cmake.define.ENABLE_CCACHE=OFF \
  --config-settings=cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  --config-settings=cmake.define.CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  ./python/sglang/kernels/aot
pip check
if [[ "${CCACHE_READONLY:-}" != 1 ]]; then
  rm -rf /opt/sglang-ccache
  mkdir /opt/sglang-ccache
  cp -a "$CCACHE_DIR/." /opt/sglang-ccache/
fi

python - <<'PY'
import json
import re
from collections import Counter
from pathlib import Path

root = Path('/opt/sglang-build-evidence')
root.mkdir(exist_ok=True)
log = Path('/tmp/sglang-ccache.log')
counts = Counter(re.findall(r'Result: (\w+)', log.read_text()) if log.exists() else [])
(root / 'ccache.json').write_text(json.dumps(dict(counts), indent=2) + '\n')
print('Native build ccache results:', json.dumps(dict(counts)), flush=True)
PY
