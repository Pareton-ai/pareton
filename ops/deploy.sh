#!/usr/bin/env bash
# Thin wrapper (stage-2 spec section 4.5): the release coordinator owns the
# state machine, both coordination locks, and every deployment write path.
# Installed live at /usr/local/bin/pareton-deploy; ops/release.py installs to
# /usr/local/lib/pareton-ops/release.py via sync-config. PARETON_OPS_DIR and
# PARETON_RELEASE_BASE remap the paths for isolated acceptance, matching the
# stage-1 helpers' conventions.
set -euo pipefail
exec "${PARETON_OPS_DIR:-/usr/local/lib/pareton-ops}/release.py" tick "$@"
