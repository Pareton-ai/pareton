#!/usr/bin/env bash
# Run on an isolated linux/amd64 builder with a GHCR login and repo Python deps.
# Use a new suffix for each run. Publishes new tags; it never seeds a campaign.
set -euo pipefail

suffix=${1:?Usage: build-sglang-baseline.sh UNIQUE_TAG_SUFFIX OUTPUT_DIR}
output_dir=${2:?An evidence output directory is required}
case "$suffix" in
  *[!a-zA-Z0-9_.-]*|'') echo 'Invalid image tag suffix' >&2; exit 2 ;;
esac
mkdir -p "$output_dir"
build_tag="ghcr.io/pareton-ai/pareton-baseline:$suffix"
engine_tag="ghcr.io/pareton-ai/pareton-engine:$suffix"
probe_tag="pareton-sglang-probe:$suffix"
commit=4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc
repo=https://github.com/sgl-project/sglang.git

python - "$build_tag" <<'PY'
import subprocess
import sys
from builder.lock import builder_storage_lock

with builder_storage_lock(blocking=True):
    subprocess.run([
        "docker", "buildx", "build", "--platform", "linux/amd64", "--load",
        "--file", "images/baseline-sglang/Dockerfile", "--tag", sys.argv[1], ".",
    ], check=True)
    subprocess.run(["docker", "push", sys.argv[1]], check=True)
PY
build_ref=$(docker inspect --format '{{index .RepoDigests 0}}' "$build_tag")
python -m builder --engine sglang --baseline-repo "$repo" \
  --baseline-commit "$commit" --base-image "$build_ref" \
  --image-ref "$engine_tag" --empty-patch --push \
  | tee "$output_dir/baseline-build.txt"
engine_ref=$(docker inspect --format '{{index .RepoDigests 0}}' "$engine_tag")

cat > "$output_dir/probe.diff" <<'PATCH'
diff --git a/python/sglang/pareton_build_probe.py b/python/sglang/pareton_build_probe.py
new file mode 100644
--- /dev/null
+++ b/python/sglang/pareton_build_probe.py
@@ -0,0 +1 @@
+PATCH_APPLIED = True
PATCH
python -m builder --engine sglang --baseline-repo "$repo" \
  --baseline-commit "$commit" --base-image "$engine_ref" \
  --image-ref "$probe_tag" --patch-file "$output_dir/probe.diff" --no-push \
  | tee "$output_dir/miner-build.txt"
docker run --rm --network none --entrypoint python "$probe_tag" -c \
  'from sglang.pareton_build_probe import PATCH_APPLIED; from pathlib import Path; assert PATCH_APPLIED; assert list(Path("/src/python/sglang").rglob("*.so")); print("offline patched import and Rust extensions: OK")' \
  | tee "$output_dir/miner-import.txt"

python - "$build_ref" "$engine_ref" "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

pins = {"build_base_image": sys.argv[1], "engine_image": sys.argv[2]}
Path(sys.argv[3], "image-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
print(json.dumps(pins, indent=2))
PY
