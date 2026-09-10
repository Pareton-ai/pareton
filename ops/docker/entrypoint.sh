#!/bin/sh
set -eu
if [ -n "${PARETON_HEALTH_FILE:-}" ]; then
    rm -f "$PARETON_HEALTH_FILE"
fi
if [ "${PARETON_REQUIRE_BUILDER:-0}" = 1 ]; then
    python -m builder.preflight
fi
# Python must receive SIGTERM directly so its existing drain handlers run.
exec "$@"
