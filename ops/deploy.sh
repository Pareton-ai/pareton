#!/usr/bin/env bash
# Pull-based deploy for the Pareton VPS, run by pareton-deploy.timer.
#
# Behavior:
#   - Fetch origin/main every tick, but pull only while the worker is truly idle.
#   - pip install only when requirements.txt changed in the pull.
#   - A shared/exclusive host lock prevents a worker from claiming work between
#     the idle check and stop. The database probe also fails closed on any
#     running submission build or benchmark round, including orphaned state.
#   - The worker stops before the first checkout mutation and starts only after
#     every deploy step succeeds. A failed deploy therefore leaves it stopped
#     and the pending flag makes a later timer tick retry the whole transaction.
#   - Watcher and weights restarts are skipped if their units are not installed
#     yet so a first-ship tick cannot abort the deploy.
#   - pareton-gpu-reap needs no restart: it is a oneshot timer that re-reads
#     the code from disk on every 10-minute run.
#
# Installed live at /usr/local/bin/pareton-deploy (outside the repo, so a
# pull can never rewrite the script mid-execution). Keep this repo copy as
# the source of truth and re-install after changing it.
set -euo pipefail

REPO=/opt/pareton
PENDING_FLAG="$REPO/.deploy-pending"
DEPLOYED_FILE="$REPO/.deploy-done"
LOCK=/run/pareton-deploy.lock

exec 9>"$LOCK"
flock -n 9 || exit 0

cd "$REPO"

git fetch --quiet origin main
REMOTE=$(git rev-parse origin/main)
# Commit of the last deploy whose pull, pip and api restart all succeeded.
# Gating on this rather than HEAD is what lets a tick that died mid-deploy
# retry: git pull has already moved HEAD by then, so a HEAD-based check would
# skip the unfinished pip/api steps forever. Absent on first run, in which
# case HEAD is treated as already deployed.
DEPLOYED=$(cat "$DEPLOYED_FILE" 2>/dev/null || git rev-parse HEAD)

if [ "$DEPLOYED" = "$REMOTE" ] && [ ! -f "$PENDING_FLAG" ]; then
    exit 0
fi

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a
WORKER_ACTIVITY_LOCK=${PARETON_WORKER_ACTIVITY_LOCK_PATH:-/run/pareton-worker-activity.lock}

# Worker cycles hold this lock shared from immediately before claim through
# final cleanup. Holding it exclusively makes the idle decision atomic with
# stopping the worker. Do not wait: a later timer tick will retry.
exec 8>"$WORKER_ACTIVITY_LOCK"
if ! flock -n 8; then
    echo "deploy: worker activity lock held; update deferred"
    exit 0
fi

if [ -f "$PENDING_FLAG" ]; then
    # A prior attempt already passed the pre-mutation probe and stopped the
    # worker. Do not import code from the possibly partial checkout to probe
    # again; stop is idempotent and preserves the crash barrier.
    systemctl stop pareton-worker
else
    # Exit 0 = busy and 10 = confirmed idle. Every other code fails closed,
    # including Python's usual exit 1 for an import or syntax failure. The
    # exclusive lock closes the gap between this query and stopping the worker.
    "$REPO/.venv/bin/python" -m ops.deploy_probe && rc=0 || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "deploy: update deferred"
        exit 0
    elif [ "$rc" -ne 10 ]; then
        echo "deploy: worker probe failed (rc=$rc); update deferred"
        exit 0
    fi

    # Stopping before the first mutation is the crash barrier: if any later
    # command fails or this oneshot is killed, no old worker can resume on a
    # partial deploy.
    systemctl stop pareton-worker
fi

if [ "$DEPLOYED" != "$REMOTE" ]; then
    # Record the incomplete transaction before its first checkout mutation.
    if [ ! -f "$PENDING_FLAG" ]; then
        touch "$PENDING_FLAG"
    fi
    git pull --ff-only --quiet origin main
    if git diff --name-only "$DEPLOYED" HEAD | grep -qx requirements.txt; then
        "$REPO/.venv/bin/pip" install --quiet -r requirements.txt
        echo "deploy: requirements.txt changed, venv updated"
    fi
    systemctl restart pareton-api
    restarted="pareton-api"
    if systemctl cat pareton-watcher >/dev/null 2>&1; then
        systemctl restart pareton-watcher
        restarted="$restarted, pareton-watcher"
    fi
    if systemctl cat pareton-weights >/dev/null 2>&1; then
        systemctl restart pareton-weights
        restarted="$restarted, pareton-weights"
    fi
    echo "deploy: $DEPLOYED -> $(git rev-parse HEAD); $restarted restarted"
    git rev-parse HEAD > "$DEPLOYED_FILE"
fi

if [ -f "$PENDING_FLAG" ]; then
    systemctl start pareton-worker
    rm -f "$PENDING_FLAG"
    echo "deploy: pareton-worker started"
fi
