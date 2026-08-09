#!/usr/bin/env bash
# Pull-based deploy for the Pareton VPS, run by pareton-deploy.timer.
#
# Behavior:
#   - git pull --ff-only when origin/main has new commits.
#   - pip install only when requirements.txt changed in the pull.
#   - pareton-api restarts on every new commit (stateless, always safe).
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
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" != "$REMOTE" ]; then
    # Mark pending BEFORE the steps that can fail. With set -e, a failed pip
    # install or api restart would otherwise abort after the pull with HEAD
    # already at origin/main, so no later tick would ever retry the rest.
    touch "$PENDING_FLAG"
    git pull --ff-only --quiet origin main
    if git diff --name-only "$LOCAL" HEAD | grep -qx requirements.txt; then
        "$REPO/.venv/bin/pip" install --quiet -r requirements.txt
        echo "deploy: requirements.txt changed, venv updated"
    fi
    systemctl restart pareton-api
    echo "deploy: $LOCAL -> $(git rev-parse HEAD); pareton-api restarted"
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
