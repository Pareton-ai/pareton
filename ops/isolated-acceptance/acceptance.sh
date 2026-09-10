#!/bin/bash
# Stage-1 isolated acceptance: A2 (install), A6 (restart failure + rollback),
# A8 (ops self-update), A19 (Vector restart continuity), plus the systemd side
# of A9/A17 (OnFailure chain + notifier state). Runs on REAL systemd inside a
# disposable container. Documented deviations from production: the Axiom sink
# is replaced by a file sink (no real token in isolation), and the webhook is
# an .invalid URL so the send path fails by design.
set -u
FAILED=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; FAILED=1; }
wait_for() { # wait_for <seconds> <file> <grep-pattern>
  local i
  for ((i = 0; i < $1; i++)); do
    grep -q -- "$3" "$2" 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

REPO=/opt/pareton
OPS=/usr/local/lib/pareton-ops
SHIPPED=/var/log/vector-shipped.log

echo "=== setup ==="
# Both path forms: direct commands validate the worktree, the clone/upload-pack
# transport validates the raw .git directory.
git config --system --add safe.directory "$REPO" 2>/dev/null
git config --system --add safe.directory "$REPO/.git" 2>/dev/null
git config --system --add safe.directory /srv/origin.git 2>/dev/null
# Baseline captured from HEAD before any container-only commit; later pushes
# move refs/remotes/origin/main, so this is the only reliable reference.
BASE_SHA=$(git -C "$REPO" rev-parse HEAD)
echo "baseline: $BASE_SHA"

# The production host runs Docker; builder-cleanup Requires=docker.service,
# and per-file verify resolves dependencies against the live system.
printf '[Unit]\nDescription=docker stub for isolated acceptance\n[Service]\nType=oneshot\nExecStart=/bin/true\n' \
  > /etc/systemd/system/docker.service
# vector validate requires an existing data_dir; the live host has one.
install -d /var/lib/vector

mkdir -p "$REPO/.venv/bin"
# Stand-in interpreter: long-running modules become a stable sleep (the real
# app deps are absent here), quick helpers succeed, and -c probes run for real.
cat > "$REPO/.venv/bin/python" <<'SH'
#!/bin/sh
if [ "$1" = "-m" ]; then
  case "$2" in
    worker.main|api|worker.watcher|weights*) exec /bin/sleep infinity ;;
    *) exit 0 ;;
  esac
fi
exec /usr/bin/python3 "$@"
SH
chmod 0755 "$REPO/.venv/bin/python"

cat > "$REPO/db/connection.py" <<'PY'
import os
import sqlite3
from contextlib import contextmanager

@contextmanager
def db_connection():
    if os.environ.get("TEST_DB_ERROR") == "1":
        raise RuntimeError("probe error injected")
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE submission_jobs (status TEXT)")
    conn.execute("CREATE TABLE rounds (status TEXT)")
    class Cursor:
        def execute(self, sql, args):
            self.result = conn.execute(sql.replace("%s", "?"), args)
        def fetchone(self):
            return self.result.fetchone()
    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()
    try:
        yield Connection()
    finally:
        conn.close()
PY

# The real db/__init__ imports names our fake does not define; the busy probe
# must import cleanly.
: > "$REPO/db/__init__.py"

cat > "$REPO/.env" <<'ENV'
PARETON_DISCORD_DEPLOY_WEBHOOK=https://pareton-isolated-test.invalid/webhook
PARETON_AXIOM_TOKEN=isolated-test-token
ENV
chown root:root "$REPO/.env"
chmod 0600 "$REPO/.env"

# Container-only TOML variant: file sink instead of Axiom, plus a transient
# unit for A19 markers. Committed so git-mode sources pick it up.
cat > "$REPO/ops/vector/vector.toml" <<'TOML'
data_dir = "/var/lib/vector"

[sources.journald]
type = "journald"
include_units = ["pareton-worker", "pareton-round-worker", "pareton-watcher", "pareton-api", "pareton-weights", "pareton-gpu-reap", "pareton-deploy", "pareton-deploy-failed", "pareton-a19"]

[transforms.parse_lifecycle]
type = "remap"
inputs = ["journald"]
source = ""

[sinks.shipped]
type = "file"
inputs = ["parse_lifecycle"]
path = "/var/log/vector-shipped.log"
encoding.codec = "json"
TOML

# Local bare origin so deploy.sh git operations work without GitHub access.
# Clone (not init) so the objects exist; pin main at the baseline.
git clone --bare -q "$REPO" /srv/origin.git
git --git-dir=/srv/origin.git update-ref refs/heads/main "$BASE_SHA"
git -C "$REPO" remote set-url origin /srv/origin.git
git -C "$REPO" fetch -q origin main
git -C "$REPO" -c user.email=isolated@test -c user.name=isolated \
  commit -qam "isolated: file-sink vector config for acceptance"

commit_repo() {
  git -C "$REPO" -c user.email=isolated@test -c user.name=isolated \
    commit -qam "isolated: $1"
  git -C "$REPO" push -q origin HEAD:refs/heads/main
}

quiet_units() {
  systemctl stop pareton-api pareton-watcher pareton-weights \
    pareton-worker pareton-round-worker 2>/dev/null || true
  systemctl reset-failed 2>/dev/null || true
}

echo "=== A2: bootstrap-order install (runbook steps 5+6) ==="
# Regression (Cursor review): production carries a deploy unit WITHOUT the
# OnFailure line; bootstrap must converge it, not block on it.
install -d /etc/systemd/system
grep -v "^OnFailure=" "$REPO/ops/systemd/pareton-deploy.service" \
  > /etc/systemd/system/pareton-deploy.service
chmod 0644 /etc/systemd/system/pareton-deploy.service
install -d -m 0755 "$OPS"
for f in ops_common.py sync-config.py notify-deploy-failure.py; do
  install -m 0755 "$REPO/ops/$f" "$OPS/$f"
done
install -m 0755 "$REPO/ops/deploy.sh" /usr/local/bin/pareton-deploy

if python3 "$OPS/sync-config.py" apply --repo "$REPO" > /tmp/a2-apply.json 2>/tmp/a2-apply.err; then
  pass "A2 apply exits 0"
else
  fail "A2 apply failed: $(cat /tmp/a2-apply.err)"
fi
python3 "$OPS/sync-config.py" check --repo "$REPO" > /tmp/a2-check.json 2>&1
[ $? -eq 0 ] && pass "A2 check=0 after apply" || fail "A2 check nonzero: $(cat /tmp/a2-check.json)"

DROPS=$(systemctl show pareton-worker -p DropInPaths --value)
case "$DROPS" in
  *queue.conf*timeout.conf*) pass "A2 systemd loaded both drop-ins: $DROPS" ;;
  *) fail "A2 drop-ins not loaded: '$DROPS'" ;;
esac
EFF=$(systemctl show pareton-worker -p TimeoutStopUSec --value)
case "$EFF" in
  4h*) pass "A2 effective TimeoutStopSec=4h (drop-in wins): $EFF" ;;
  *) fail "A2 effective stop budget wrong: $EFF" ;;
esac
grep -q "^OnFailure=pareton-deploy-failed.service" /etc/systemd/system/pareton-deploy.service \
  && pass "A2 OnFailure present in installed deploy unit" \
  || fail "A2 OnFailure missing"
[ "$(systemctl is-active vector)" = active ] && pass "A2 vector active" || fail "A2 vector not active"
[ -x /usr/local/bin/pareton-deploy ] && pass "A2 deploy script installed executable" || fail "A2 deploy script missing"
[ "$(stat -c %a /etc/vector/vector.toml)" = 600 ] && pass "A2 TOML mode 0600" || fail "A2 TOML mode $(stat -c %a /etc/vector/vector.toml)"

echo "=== A19: vector restart continuity (checkpoint) ==="
# Markers are real deploy-service runs: their stdout lands in the journal
# under pareton-deploy.service (an include_unit). A lost checkpoint would
# re-ship pre-restart entries after the restart, so exact counts prove both
# delivery and continuity.
systemctl start pareton-deploy.service
# The hook's JSON line lands inside the file-sink event with escaped quotes,
# so match the bare substring.
MARK="deploy-hook"
if wait_for 15 "$SHIPPED" "$MARK"; then
  pass "A19 marker-1 shipped before restart"
else
  fail "A19 marker-1 never reached sink; last lines: $(tail -3 "$SHIPPED" 2>/dev/null)"
fi
BEFORE=$(ls -A /var/lib/vector | wc -l)
C_BEFORE=$(grep -cF "$MARK" "$SHIPPED" || true)
printf '\n# isolated change to trigger restart\n' >> "$REPO/ops/vector/vector.toml"
commit_repo "touch toml for restart"
python3 "$OPS/sync-config.py" deploy-hook --repo "$REPO" >/tmp/a19-hook.json 2>&1 \
  && pass "A19 deploy-hook ok" || fail "A19 deploy-hook failed: $(cat /tmp/a19-hook.json)"
sleep 2
systemctl start pareton-deploy.service
sleep 3
C_TOTAL=$(grep -cF "$MARK" "$SHIPPED" || true)
[ "$C_TOTAL" -eq $((C_BEFORE + 1)) ] \
  && pass "A19 second run shipped, no duplicate re-ingest ($C_BEFORE->$C_TOTAL)" \
  || fail "A19 counts wrong: before=$C_BEFORE total=$C_TOTAL (duplication or missing)"
AFTER=$(ls -A /var/lib/vector | wc -l)
[ "$AFTER" -gt 0 ] && [ "$AFTER" -ge "$BEFORE" ] \
  && pass "A19 data_dir persisted ($BEFORE->$AFTER entries)" \
  || fail "A19 data_dir suspicious ($BEFORE->$AFTER)"

echo "=== A6: vector unit breaks -> fail, rollback, retry ==="
cp /etc/systemd/system/vector.service /tmp/vector.service.orig
sed -i 's|^ExecStart=.*|ExecStart=/bin/false|' "$REPO/ops/vector/vector.service"
commit_repo "break vector unit"
python3 "$OPS/sync-config.py" deploy-hook --repo "$REPO" >/tmp/a6.json 2>&1
RC=$?
[ "$RC" -ne 0 ] && pass "A6 broken vector candidate fails (rc=$RC)" || fail "A6 broken candidate accepted"
if diff -q /tmp/vector.service.orig /etc/systemd/system/vector.service >/dev/null; then
  pass "A6 vector.service rolled back byte-identical"
else
  fail "A6 rollback did not restore vector.service"
fi
sleep 2
for i in $(seq 1 10); do
  [ "$(systemctl is-active vector)" = active ] && break
  sleep 1
done
[ "$(systemctl is-active vector)" = active ] && pass "A6 vector active again after rollback" || fail "A6 vector down after rollback"
# The repo source must be healthy again before the retry can converge.
git -C "$REPO" checkout "$BASE_SHA" -- ops/vector/vector.service
[ -s "$REPO/ops/vector/vector.service" ] || fail "A6 restore produced empty file"
commit_repo "restore vector unit"
python3 "$OPS/sync-config.py" deploy-hook --repo "$REPO" >/tmp/a6b.json 2>&1 \
  && pass "A6 next tick converges cleanly" || fail "A6 retry failed: $(cat /tmp/a6b.json)"

echo "=== A8: deploy self-install via real deploy.sh ==="
echo "# self-update marker v1" >> "$REPO/ops/deploy.sh"
commit_repo "deploy.sh v1"
# install_ops only runs on the new-commit path; make this tick one.
git -C "$REPO" rev-parse HEAD~1 > "$REPO/.deploy-done"
systemctl start pareton-deploy.service
RC=$?
sleep 1
quiet_units
[ "$RC" = 0 ] && pass "A8 deploy tick exits 0" || fail "A8 deploy tick rc=$RC: $(journalctl -u pareton-deploy -n 20 --no-pager | tail -5)"
diff -q "$REPO/ops/deploy.sh" /usr/local/bin/pareton-deploy >/dev/null \
  && pass "A8 installed script matches repo copy" \
  || fail "A8 installed script differs from repo"
LEFT=$(ls -A /usr/local/lib/pareton-ops /usr/local/bin/.pareton-deploy.new.* 2>/dev/null | grep -c '\.new\.' || true)
[ "$LEFT" = 0 ] && pass "A8 no atomic-replace leftovers" || fail "A8 leftover temp files"
grep -q '"invocation"' /var/lib/pareton-deploy/alert-state.json 2>/dev/null \
  && pass "A8 record-success wrote state" || fail "A8 no success state"
echo "# self-update marker v2" >> "$REPO/ops/deploy.sh"
sed -i 's/self-update marker v1/self-update marker v1 superseded/' "$REPO/ops/deploy.sh"
commit_repo "deploy.sh v2"
systemctl start pareton-deploy.service
sleep 1
quiet_units
grep -q "marker v2" /usr/local/bin/pareton-deploy \
  && pass "A8 later run picked up new version" \
  || fail "A8 self-update did not converge"

echo "=== A9/A17 (systemd side): probe failure -> OnFailure -> notifier ==="
rm -f /var/lib/pareton-deploy/alert-state.json
echo "TEST_DB_ERROR=1" >> "$REPO/.env"
touch "$REPO/.deploy-pending" "$REPO/.deploy-rounds-pending"
systemctl start pareton-deploy.service
RC=$?
sleep 2
[ "$RC" -ne 0 ] && pass "A9 deploy fails on probe error (rc=$RC)" || fail "A9 probe error did not fail deploy"
NF_RESULT=$(systemctl show pareton-deploy-failed.service -p Result --value)
[ "$NF_RESULT" != inactive ] && [ -n "$NF_RESULT" ] \
  && pass "A9 OnFailure unit ran (result=$NF_RESULT)" \
  || fail "A9 OnFailure unit never started"
journalctl -u pareton-deploy-failed.service --no-pager 2>/dev/null | grep -q "notify" \
  && pass "A9 notifier executed and logged" \
  || fail "A9 no notifier output in journal"
python3 - <<'PY' && pass "A17 send-failure kept fault, no suppression window" \
|| fail "A17 state wrong"
import json, sys
state = json.load(open("/var/lib/pareton-deploy/alert-state.json"))
fault = state.get("fault") or {}
assert fault.get("count", 0) >= 1, fault
assert fault.get("last_notified") is None, fault
assert state.get("send_failures", 0) >= 1, state
# Owner-verified bug regression: the invocation must match the failed run
# (parsed by key, not by systemd's output order), so the alert facts and the
# fault key identify the actual deploy run.
key = fault.get("key") or {}
assert key.get("step") == "probe-worker", key
assert key.get("target_commit") not in ("", "unknown", None), key
PY
sed -i '/TEST_DB_ERROR/d' "$REPO/.env"
rm -f "$REPO/.deploy-pending" "$REPO/.deploy-rounds-pending"
quiet_units

echo
if [ "$FAILED" = 0 ]; then
  echo "ALL-ISOLATED-ACCEPTANCE-PASSED"
else
  echo "ISOLATED-ACCEPTANCE-HAD-FAILURES"
fi
exit "$FAILED"
