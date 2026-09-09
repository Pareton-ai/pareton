#!/usr/bin/env bash
# Run on an isolated linux/amd64 builder with a GHCR login and repo Python deps.
# Use a new suffix for each run. Publishes new tags; it never seeds a campaign.
set -euo pipefail

suffix=${1:?Usage: build-sglang-baseline.sh UNIQUE_TAG_SUFFIX OUTPUT_DIR [PUBLISHED_BUILD_BASE_REF]}
output_dir=${2:?An evidence output directory is required}
reuse_build_ref=${3:-}
case "$suffix" in
  *[!a-zA-Z0-9_.-]*|'') echo 'Invalid image tag suffix' >&2; exit 2 ;;
esac
mkdir -p "$output_dir"
# A cold native build exceeded twelve hours on the validator VPS. This limit is
# ops-only; production miner builds retain their configured deadline.
export PARETON_BUILD_TIMEOUT_S="${PARETON_BUILD_TIMEOUT_S:-172800}"
# Operator-selected parallelism for this SGLang ops build. NVCC threads per job
# remain one; MAX_JOBS also controls CMake and Rust build concurrency.
export PARETON_BUILD_MAX_JOBS="${PARETON_BUILD_MAX_JOBS:-6}"
export PARETON_BUILD_LOG_DIR="${PARETON_BUILD_LOG_DIR:-$output_dir/logs}"
build_tag="ghcr.io/pareton-ai/pareton-baseline:$suffix"
engine_tag="ghcr.io/pareton-ai/pareton-baseline:$suffix-engine"
probe_tag="ghcr.io/pareton-ai/pareton-baseline:$suffix-probe"
commit=4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc
repo=https://github.com/sgl-project/sglang.git

python - "$build_tag" "$reuse_build_ref" "$commit" <<'PY'
import re
import subprocess
import sys
from builder.lock import builder_storage_lock

with builder_storage_lock(blocking=True):
    reuse = sys.argv[2]
    if reuse:
        if not re.fullmatch(r"ghcr\.io/pareton-ai/pareton-baseline@sha256:[a-f0-9]{64}", reuse):
            raise SystemExit("Reuse requires a digest-pinned Pareton build base")
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", reuse], check=True)
        source_pin = subprocess.check_output([
            "docker", "inspect", "--format", '{{index .Config.Labels "ai.pareton.sglang.commit"}}', reuse,
        ], text=True).strip()
        if source_pin != sys.argv[3]:
            raise SystemExit("Build base label does not match the SGLang source pin")
        native_build = subprocess.check_output([
            "docker", "inspect", "--format", '{{index .Config.Labels "ai.pareton.sglang.native-build"}}', reuse,
        ], text=True).strip()
        if native_build != "1":
            raise SystemExit("Build base does not support offline native SGLang builds")
    else:
        subprocess.run([
            "docker", "buildx", "build", "--platform", "linux/amd64", "--load",
            "--file", "images/baseline-sglang/Dockerfile", "--tag", sys.argv[1], ".",
        ], check=True)
        subprocess.run(["docker", "push", sys.argv[1]], check=True)
PY
build_ref=${reuse_build_ref:-$(docker inspect --format '{{index .RepoDigests 0}}' "$build_tag")}
printf '%s\n' "$build_ref" > "$output_dir/build-base-image.txt"
python -m builder --engine sglang --baseline-repo "$repo" \
  --baseline-commit "$commit" --base-image "$build_ref" \
  --image-ref "$engine_tag" --empty-patch --push --stream-build-logs \
  2>&1 | tee "$output_dir/baseline-build.txt"
engine_ref=$(docker inspect --format '{{index .RepoDigests 0}}' "$engine_tag")
printf '%s\n' "$engine_ref" > "$output_dir/engine-image.txt"

python ops/make-sglang-native-probe.py "$engine_ref" "$output_dir/probe.diff"
python -m builder --engine sglang --baseline-repo "$repo" \
  --baseline-commit "$commit" --base-image "$engine_ref" \
  --image-ref "$probe_tag" --patch-file "$output_dir/probe.diff" --push --stream-build-logs \
  2>&1 | tee "$output_dir/miner-build.txt"
docker run --rm --network none --entrypoint python "$probe_tag" -c \
  'import torch; from sglang.srt.mem_cache.rust_tree_core import mem_cache; assert mem_cache.PARETON_NATIVE_PROBE == 41; print("offline patched Rust extension: OK")' \
  | tee "$output_dir/miner-import.txt"
docker run --rm --network none --entrypoint cat "$probe_tag" /opt/sglang-build-evidence/ccache.json \
  > "$output_dir/miner-ccache.json"
probe_ref=$(docker inspect --format '{{index .RepoDigests 0}}' "$probe_tag")

python - "$build_ref" "$engine_ref" "$probe_ref" "$output_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[4])
cache = json.loads((root / 'miner-ccache.json').read_text())
assert cache.get('direct_cache_hit', 0) + cache.get('preprocessed_cache_hit', 0) > 0, cache
assert cache.get('cache_miss', 0) > 0, cache
pins = {
    "build_base_image": sys.argv[1], "engine_image": sys.argv[2],
    "probe_image": sys.argv[3],
    "probe_patch_sha256": hashlib.sha256((root / 'probe.diff').read_bytes()).hexdigest(),
}
(root / "image-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
print(json.dumps(pins, indent=2))
PY
