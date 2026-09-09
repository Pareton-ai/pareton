#!/usr/bin/env bash
# Pull-based deploy for the Pareton VPS, run by pareton-deploy.timer.
#
# Behavior:
#   - git pull --ff-only when origin/main has new commits.
#   - pip install only when requirements.txt changed in the pull.
#   - pareton-api, pareton-watcher, and pareton-weights restart on every new
#     commit (stateless enough to be always safe). Watcher and weights restart
#     is skipped if the unit is not installed yet so a first-ship tick cannot
#     abort the deploy.
#   - Execution workers have separate pending restart flags. The round worker
#     waits only for rounds; the existing worker checks both queues to protect
#     legacy combined processes during migration.
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
ROUND_PENDING_FLAG="$REPO/.deploy-rounds-pending"
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
        cur.execute(\"SELECT 1 FROM rounds WHERE status = 'running' \"
                    \"UNION ALL SELECT 1 FROM submission_jobs \"
                    \"WHERE status = 'running' AND %s LIMIT 1\",
                    (sys.argv[1] != 'rounds',))
        sys.exit(0 if cur.fetchone() else 1)
except Exception:
    sys.exit(2)
" "$1"
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
    touch "$PENDING_FLAG" "$ROUND_PENDING_FLAG"
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

for unit in pareton-round-worker pareton-worker; do
    pending="$PENDING_FLAG"
    queue=all
    if [ "$unit" = pareton-round-worker ]; then
        pending="$ROUND_PENDING_FLAG"
        queue=rounds
    fi
    [ -f "$pending" ] || continue
    systemctl cat "$unit" >/dev/null 2>&1 || continue
    worker_busy "$queue" && rc=0 || rc=$?
    if [ "$rc" -eq 1 ]; then
        systemctl restart "$unit"
        rm -f "$pending"
        echo "deploy: $unit restarted"
    elif [ "$rc" -eq 0 ]; then
        echo "deploy: $unit has running work; restart deferred"
    else
        echo "deploy: $unit probe failed (rc=$rc); treating as busy, restart deferred"
    fi
done
