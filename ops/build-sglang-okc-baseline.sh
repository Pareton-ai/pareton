#!/usr/bin/env bash
# Build the OKC-matched SGLang base, its empty-patch engine and a probe miner
# build, all through the offline hermetic builder. Run on an isolated
# linux/amd64 builder with a GHCR login and repo Python deps. Use a new suffix
# for each run. Publishes new tags; it never seeds a campaign.
set -euo pipefail

suffix=${1:?Usage: build-sglang-okc-baseline.sh UNIQUE_TAG_SUFFIX OUTPUT_DIR}
output_dir=${2:?An evidence output directory is required}
case "$suffix" in
  *[!a-zA-Z0-9_.-]*|'') echo 'Invalid image tag suffix' >&2; exit 2 ;;
esac
mkdir -p "$output_dir"
export PARETON_BUILD_MAX_JOBS="${PARETON_BUILD_MAX_JOBS:-8}"
export PARETON_BUILD_LOG_DIR="${PARETON_BUILD_LOG_DIR:-$output_dir/logs}"
build_tag="ghcr.io/pareton-ai/pareton-baseline:$suffix"
engine_tag="ghcr.io/pareton-ai/pareton-baseline:$suffix-engine"
probe_tag="ghcr.io/pareton-ai/pareton-baseline:$suffix-probe"
commit=a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
repo=https://github.com/sgl-project/sglang.git

docker buildx build --builder "${PARETON_BUILDER_NAME:-default}" --platform linux/amd64 --load \
  --build-arg MAX_JOBS="$PARETON_BUILD_MAX_JOBS" \
  --file images/baseline-sglang-okc/Dockerfile --tag "$build_tag" . \
  2>&1 | tee "$output_dir/base-build.txt"
docker push "$build_tag"
build_ref=$(docker inspect --format '{{index .RepoDigests 0}}' "$build_tag")
printf '%s\n' "$build_ref" > "$output_dir/build-base-image.txt"

python -m builder --engine sglang-okc --baseline-repo "$repo" \
  --baseline-commit "$commit" --base-image "$build_ref" \
  --image-ref "$engine_tag" --empty-patch --push --stream-build-logs \
  2>&1 | tee "$output_dir/baseline-build.txt"
engine_ref=$(docker inspect --format '{{index .RepoDigests 0}}' "$engine_tag")
printf '%s\n' "$engine_ref" > "$output_dir/engine-image.txt"

# A miner patch with a Python module and a JIT CUDA kernel, the two kinds of
# source OKC accepts under python/sglang/. The kernel is the native probe's
# (ops/make-sglang-native-probe.py) at its in-tree JIT path.
python3 - "$output_dir/probe.diff" <<'PY'
import difflib
import runpy
import sys

jit_source = runpy.run_path("ops/make-sglang-native-probe.py")["JIT_SOURCE"]
files = {
    "python/sglang/pareton_probe.py": "PARETON_OKC_PROBE = 41\n",
    "python/sglang/kernels/jit/csrc/pareton_probe.cuh": jit_source,
}
with open(sys.argv[1], "w") as out:
    for path, text in files.items():
        out.write(f"diff --git a/{path} b/{path}\nnew file mode 100644\n")
        out.writelines(
            difflib.unified_diff(
                [], text.splitlines(keepends=True), "/dev/null", f"b/{path}", n=0
            )
        )
PY
python -m builder --engine sglang-okc --baseline-repo "$repo" \
  --baseline-commit "$commit" --base-image "$build_ref" \
  --image-ref "$probe_tag" --patch-file "$output_dir/probe.diff" --push --stream-build-logs \
  2>&1 | tee "$output_dir/miner-build.txt"
docker run --rm --network none --entrypoint python3 "$probe_tag" -c \
  'import torch; from sglang.srt.mem_cache.rust_tree_core import mem_cache; import sglang.pareton_probe as p; assert p.PARETON_OKC_PROBE == 41; from sglang.srt.layers.moe.muse_spark_megamoe import require_muse_megamoe_patch; require_muse_megamoe_patch(); print("offline OKC miner build: OK")' \
  | tee "$output_dir/miner-import.txt"
# JIT kernels compile with nvcc on first use, which needs no GPU. Compile the
# probe kernel offline for H200 (sm_90) through SGLang's own load_jit, skipping
# only the final load, which would need a GPU.
docker run --rm --network none --entrypoint python3 "$probe_tag" -c \
  'from sglang.kernels.jit.utils.arch import override_jit_cuda_arch; from sglang.kernels.jit.utils.compile import load_jit, loader; loader._load = lambda library: library; m = override_jit_cuda_arch(9, 0); m.__enter__(); load_jit("pareton_okc_probe", cuda_files=["pareton_probe.cuh"], cuda_wrappers=[("run", "pareton::run")]); print("offline OKC JIT compile: OK")' \
  | tee "$output_dir/miner-jit-compile.txt"
