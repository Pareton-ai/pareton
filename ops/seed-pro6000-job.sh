#!/usr/bin/env bash
# Operator job wrapper. Launch ONLY after qualification/shadow/deployment review.
# This creates an open campaign; no automatic retries, including after a crash.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${PRO6000_SEED_DIR:?Set the protected seed job directory on the controller}"
: "${PRO6000_QUALIFIED_RULE:?Set the transferred qualified sampling rule path}"
mkdir "$PRO6000_SEED_DIR/seed.lock" || {
  echo 'Seed job already claimed this directory; inspect its log and database/API before any retry.' >&2
  exit 1
}
printf '%s\n' "$$" > "$PRO6000_SEED_DIR/seed.pid"
finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$PRO6000_SEED_DIR/seed.exit-code.tmp"
  mv "$PRO6000_SEED_DIR/seed.exit-code.tmp" "$PRO6000_SEED_DIR/seed.exit-code"
  printf 'Seed job finished (exit %s). Inspect seed.log and database/API state; never retry blindly.\n' "$rc"
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u -m ops.pro6000_preflight
engine_ref=$(python -u -c 'import json; print(json.load(open("fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json"))["base_image_digest"])')
export PYTHONUNBUFFERED=1
bash ops/seed-sglang-qwen38-27b-pro6000.sh "$engine_ref" "$PRO6000_QUALIFIED_RULE"
