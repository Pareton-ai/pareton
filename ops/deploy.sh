#!/usr/bin/env bash
# Pull-based deploy for the Pareton VPS, run by pareton-deploy.timer.
#
# Behavior:
#   - git pull --ff-only when origin/main has new commits.
#   - pip install only when requirements.txt changed in the pull.
#   - Stage-1 config sync (ops/sync-config.py deploy-hook) runs on every tick
#     that holds the deploy lock: managed-file drift auto-converges, unknown
#     files/masks/broken alert prerequisites fail the deploy, and owed
#     daemon-reload/Vector restart actions are retried (spec section 5.3).
#     Config candidates are validated and installed BEFORE any service is
#     restarted onto them (spec section 6).
#   - install_ops atomically self-installs this script plus the ops helpers
#     from the just-pulled commit, so merging to main updates them without a
#     manual copy. Helpers are replaced before the main entry (spec 6.1).
#   - pareton-api, pareton-watcher, and pareton-weights restart on every new
#     commit (stateless enough to be always safe). Watcher and weights restart
#     is skipped if the unit is not installed yet so a first-ship tick cannot
#     abort the deploy. Units that only owe a config-driven restart are
#     restarted on no-change ticks until the debt is cleared.
#   - Execution workers have separate pending restart flags. The round worker
#     waits only for rounds; the existing worker checks both queues to protect
#     legacy combined processes during migration.
#     A running job killed mid-bench is never requeued (claim_next_job only
#     claims 'pending'), and its GPU pod burns money until the TTL reaper.
#     When busy, a pending flag defers the restart to a later idle tick.
#   - A failing worker busy-probe fails the whole deploy (OnFailure alert)
#     instead of silently skipping updates; a busy database is a legal deferral
#     (spec section 6.4).
#   - Progress is recorded to /var/lib/pareton-deploy/last-run.env for the
#     failure notifier; a fully successful tick calls record-success to clear
#     the active fault (spec sections 7.2-7.3).
#   - pareton-gpu-reap needs no restart: it is a oneshot timer that re-reads
#     the code from disk on every 10-minute run.
#
# Installed live at /usr/local/bin/pareton-deploy and self-installed from
# ops/deploy.sh on the success path. First activation needs a one-time manual
# bootstrap; see ops/runbook.md (spec section 6.1).
set -euo pipefail

REPO=${PARETON_DEPLOY_REPO:-/opt/pareton}
OPS_DIR=${PARETON_OPS_DIR:-/usr/local/lib/pareton-ops}
STATE_DIR=${PARETON_STATE_DIR:-/var/lib/pareton-deploy}
DEPLOY_BIN=${PARETON_DEPLOY_BIN:-/usr/local/bin/pareton-deploy}
PENDING_FLAG="$REPO/.deploy-pending"
ROUND_PENDING_FLAG="$REPO/.deploy-rounds-pending"
DEPLOYED_FILE="$REPO/.deploy-done"
LOCK=${PARETON_DEPLOY_LOCK:-/run/pareton-deploy.lock}
SYNC="$OPS_DIR/sync-config.py"
NOTIFY="$OPS_DIR/notify-deploy-failure.py"
RUN_STATE="$STATE_DIR/last-run.env"
INVOCATION=${INVOCATION_ID:-manual}
STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

exec 9>"$LOCK"
flock -n 9 || exit 0

cd "$REPO"
mkdir -p "$STATE_DIR"

record_step() {
  # KEY=VALUE lines parsed by the notifier with the same env-file parser.
  # Never log environment contents here; values are commits and step names.
  printf 'invocation_id=%s\nstarted_at=%s\nfrom_commit=%s\ntarget_commit=%s\nlast_step=%s\n' \
    "$INVOCATION" "$STARTED_UTC" "${FROM_COMMIT:-unknown}" "${TARGET_COMMIT:-unknown}" "$1" \
    > "$RUN_STATE"
}

install_ops() {
  [ -f "$REPO/ops/deploy.sh" ] || return 0
  mkdir -p "$OPS_DIR" "$(dirname "$DEPLOY_BIN")"
  local helper tmp
  for helper in ops_common.py sync-config.py notify-deploy-failure.py; do
    [ -f "$REPO/ops/$helper" ] || continue
    tmp="$OPS_DIR/.$helper.new.$$"
    install -m 0755 "$REPO/ops/$helper" "$tmp"
    mv -f "$tmp" "$OPS_DIR/$helper"
  done
  # The main entry goes last so a partial helper update can never leave the
  # installed deploy script calling programs that are not there yet.
  tmp="$(dirname "$DEPLOY_BIN")/.pareton-deploy.new.$$"
  install -m 0755 "$REPO/ops/deploy.sh" "$tmp"
  mv -f "$tmp" "$DEPLOY_BIN"
}

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

run_owed_restarts() {
  local owed unit
  owed=$("$SYNC" owed-restarts --repo "$REPO" 2>/dev/null || true)
  [ -n "$owed" ] || return 0
  for unit in $owed; do
    if systemctl cat "$unit" >/dev/null 2>&1; then
      systemctl restart "$unit"
      echo "deploy: $unit restarted (config change effectuation)"
    fi
  done
  "$SYNC" clear-restarts --repo "$REPO" $owed >/dev/null || true
}

record_step fetch
git fetch --quiet origin main
REMOTE=$(git rev-parse origin/main)
FROM_COMMIT=$(git rev-parse HEAD)
TARGET_COMMIT=$REMOTE
# Commit of the last deploy whose pull, pip, config sync and api restart all
# succeeded. Gating on this rather than HEAD is what lets a tick that died
# mid-deploy retry: git pull has already moved HEAD by then, so a HEAD-based
# check would skip the unfinished pip/config/api steps forever. Absent on
# first run, in which case HEAD is treated as already deployed.
DEPLOYED=$(cat "$DEPLOYED_FILE" 2>/dev/null || git rev-parse HEAD)

if [ "$DEPLOYED" != "$REMOTE" ]; then
    # Mark the worker restarts owed before the steps that can fail. If one
    # does, set -e aborts here and the pending block below never runs, so the
    # workers are not restarted onto a half-deployed tree.
    touch "$PENDING_FLAG" "$ROUND_PENDING_FLAG"
    record_step pull
    git pull --ff-only --quiet origin main
    if git diff --name-only "$DEPLOYED" HEAD | grep -qx requirements.txt; then
        record_step deps
        "$REPO/.venv/bin/pip" install --quiet -r requirements.txt
        echo "deploy: requirements.txt changed, venv updated"
    fi
    record_step install-ops
    install_ops
    record_step install-config
    "$SYNC" deploy-hook --repo "$REPO"
    record_step restart-app
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
    # The three app units were just restarted onto the new commit; whatever
    # restart debt pointed at them is settled.
    "$SYNC" clear-restarts --repo "$REPO" \
      pareton-api.service pareton-watcher.service pareton-weights.service \
      >/dev/null 2>&1 || true
    echo "deploy: $DEPLOYED -> $(git rev-parse HEAD); $restarted restarted"
    git rev-parse HEAD > "$DEPLOYED_FILE"
else
    record_step install-config
    "$SYNC" deploy-hook --repo "$REPO"
    run_owed_restarts
fi

probe_failed=0
for unit in pareton-round-worker pareton-worker; do
    pending="$PENDING_FLAG"
    queue=all
    if [ "$unit" = pareton-round-worker ]; then
        pending="$ROUND_PENDING_FLAG"
        queue=rounds
    fi
    [ -f "$pending" ] || continue
    systemctl cat "$unit" >/dev/null 2>&1 || continue
    record_step probe-worker
    worker_busy "$queue" && rc=0 || rc=$?
    if [ "$rc" -eq 1 ]; then
        systemctl restart "$unit"
        rm -f "$pending"
        echo "deploy: $unit restarted"
    elif [ "$rc" -eq 0 ]; then
        echo "deploy: $unit has running work; restart deferred"
    else
        # Spec 6.4: a broken probe is a deploy failure, not a silent skip.
        echo "deploy: $unit probe failed (rc=$rc); restart deferred, deploy fails" >&2
        probe_failed=1
    fi
done

if [ "$probe_failed" -ne 0 ]; then
    exit 1
fi

record_step done
# Notification bookkeeping only; a broken notifier must not fail deploys.
"$NOTIFY" record-success \
  --invocation "$INVOCATION" --from "$FROM_COMMIT" --to "$TARGET_COMMIT" \
  >/dev/null 2>&1 || echo "deploy: record-success failed (notifier unavailable)"
