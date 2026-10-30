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

docker buildx build --platform linux/amd64 --load \
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

# A miner patch with a Python module and a JIT CUDA source, the two kinds of
# file OKC accepts under python/sglang/. JIT sources compile on first use on
# the GPU, so this offline check only proves the build ships the source.
cat > "$output_dir/probe.diff" <<'DIFF'
diff --git a/python/sglang/pareton_probe.py b/python/sglang/pareton_probe.py
new file mode 100644
--- /dev/null
+++ b/python/sglang/pareton_probe.py
@@ -0,0 +1 @@
+PARETON_OKC_PROBE = 41
diff --git a/python/sglang/pareton_probe.cuh b/python/sglang/pareton_probe.cuh
new file mode 100644
--- /dev/null
+++ b/python/sglang/pareton_probe.cuh
@@ -0,0 +1 @@
+__device__ inline float pareton_okc_probe(float x) { return x + 41.0f; }
DIFF
python -m builder --engine sglang-okc --baseline-repo "$repo" \
  --baseline-commit "$commit" --base-image "$build_ref" \
  --image-ref "$probe_tag" --patch-file "$output_dir/probe.diff" --push --stream-build-logs \
  2>&1 | tee "$output_dir/miner-build.txt"
docker run --rm --network none --entrypoint python3 "$probe_tag" -c \
  'import torch; from sglang.srt.mem_cache.rust_tree_core import mem_cache; import sglang.pareton_probe as p; assert p.PARETON_OKC_PROBE == 41; from pathlib import Path; assert "41.0f" in Path(p.__file__).with_suffix(".cuh").read_text(); from sglang.srt.layers.moe.muse_spark_megamoe import require_muse_megamoe_patch; require_muse_megamoe_patch(); print("offline OKC miner build: OK")' \
  | tee "$output_dir/miner-import.txt"
