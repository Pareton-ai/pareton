#!/usr/bin/env bash
# Host dependencies only; run under nohup before creating any GPU containers.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${PRO6000_SETUP_DIR:?Set a fresh setup evidence directory}"
mkdir "$PRO6000_SETUP_DIR/setup.lock"
printf '%s\n' "$$" > "$PRO6000_SETUP_DIR/setup.pid"
finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$PRO6000_SETUP_DIR/setup.exit-code.tmp"
  mv "$PRO6000_SETUP_DIR/setup.exit-code.tmp" "$PRO6000_SETUP_DIR/setup.exit-code"
  printf 'Environment setup finished (exit %s); inspect setup.log before proceeding.\n' "$rc"
  exit "$rc"
}
trap finish EXIT
python3 -m venv .venv
source .venv/bin/activate
python -u -m pip install -r requirements.txt
python -u -m pip check
python -u -m ops.pro6000_preflight
