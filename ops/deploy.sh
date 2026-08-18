#!/usr/bin/env bash
# Pull-based deploy for the Pareton VPS, run by pareton-deploy.timer.
#
# Behavior:
#   - git pull --ff-only when origin/main has new commits.
#   - pip install only when requirements.txt changed in the pull.
#   - pareton-api and pareton-watcher restart on every new commit
#     (stateless, always safe). Watcher restart is skipped if the unit
#     is not installed yet so a first-ship tick cannot abort the deploy.
#   - pareton-worker restarts only when no submission_jobs are 'running'.
#     A running job killed mid-bench is never requeued (claim_next_job only
#     claims 'pending'), and its GPU pod burns money until the TTL reaper.
#     When busy, a pending flag defers the restart to a later idle tick.
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

worker_busy() {
    set -a
    # shellcheck disable=SC1091
    source "$REPO/.env"
    set +a
    # Exit 0 = busy, 1 = idle, 2 = probe error. Callers must treat 2 as busy
    # (fail closed) so a broken probe never restarts a worker mid-job.
    "$REPO/.venv/bin/python" -c "
import sys
try:
    from db.connection import db_connection
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(\"SELECT 1 FROM submission_jobs WHERE status = 'running' LIMIT 1\")
        sys.exit(0 if cur.fetchone() else 1)
except Exception:
    sys.exit(2)
"
}

git fetch --quiet origin main
REMOTE=$(git rev-parse origin/main)
# Commit of the last deploy whose pull, pip and api restart all succeeded.
# Gating on this rather than HEAD is what lets a tick that died mid-deploy
# retry: git pull has already moved HEAD by then, so a HEAD-based check would
# skip the unfinished pip/api steps forever. Absent on first run, in which
# case HEAD is treated as already deployed.
DEPLOYED=$(cat "$DEPLOYED_FILE" 2>/dev/null || git rev-parse HEAD)

if [ "$DEPLOYED" != "$REMOTE" ]; then
    # Mark the worker restart owed before the steps that can fail. If one does,
    # set -e aborts here and the pending block below never runs, so the worker
    # is not restarted onto a half-deployed tree.
    touch "$PENDING_FLAG"
    git pull --ff-only --quiet origin main
    if git diff --name-only "$DEPLOYED" HEAD | grep -qx requirements.txt; then
        "$REPO/.venv/bin/pip" install --quiet -r requirements.txt
        echo "deploy: requirements.txt changed, venv updated"
    fi
    systemctl restart pareton-api
    if systemctl cat pareton-watcher >/dev/null 2>&1; then
        systemctl restart pareton-watcher
        echo "deploy: $DEPLOYED -> $(git rev-parse HEAD); pareton-api and pareton-watcher restarted"
    else
        echo "deploy: $DEPLOYED -> $(git rev-parse HEAD); pareton-api restarted (pareton-watcher unit not installed)"
    fi
    git rev-parse HEAD > "$DEPLOYED_FILE"
fi

if [ -f "$PENDING_FLAG" ]; then
    worker_busy && rc=0 || rc=$?
    if [ "$rc" -eq 1 ]; then
        systemctl restart pareton-worker
        rm -f "$PENDING_FLAG"
        echo "deploy: pareton-worker restarted"
    elif [ "$rc" -eq 0 ]; then
        echo "deploy: worker has a running job; restart deferred"
    else
        echo "deploy: worker probe failed (rc=$rc); treating as busy, restart deferred"
    fi
fi
