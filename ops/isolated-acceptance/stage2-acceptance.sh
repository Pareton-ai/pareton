#!/bin/bash
# Stage-2 isolated acceptance: real systemd + real Vector end-to-end release.
#
# Covers the systemd-facing slice of the stage-2 spec acceptance matrix:
#   S1  bootstrap via request reset (state machine starts from nothing)
#   S2  release-failure drill + notification-acceptance record + verify
#       (the mock Axiom serves the deploy-failed evidence)
#   S3  full release A->B through draining/quiescing/applying/re-exec/
#       verifying with real ExecCondition, ExecStartPre probes, in-process
#       probe threads (stand-in apps run the REAL observability.probe loop),
#       the one-shot GPU reap dispatch, and per-unit log checks
#   S4  gate matrix on real systemd: applying -> ExecCondition exit 1 skips
#       without failing; corrupt state -> 255 -> failed + release_gate_error
#   S5  worker activity lock: a live shared holder makes the tick report
#       active-work and exit 0 without writing the environment
#   S6  gpu-reap dispatch: consumed one-shot probe skips the real reap;
#       expired request runs it
#   S7  missing-source fault injection (B15): checker lists the missing unit
#
# Documented deviations from production: the Axiom sink keeps a fake token
# (delivery retries into the disk buffer harmlessly) while a file sink
# mirrors events for the in-container mock Axiom HTTP server that
# check-logs queries; the webhook is .invalid so the drill notifier's
# Discord send fails by design while its structured event still ships;
# app logic is sleep stand-ins, but coordination, probe, and HTTP surfaces
# run the real code; "reboot the host" is approximated by stopping all
# units and clearing /run coordination files.
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
wait_inactive() { # wait_inactive <seconds> <unit>
  local i
  for ((i = 0; i < $1; i++)); do
    [ "$(systemctl is-active "$2" 2>/dev/null)" != active ] && return 0
    sleep 1
  done
  return 1
}
state_json() { python3 -c "import json,sys;print(json.load(open('/var/lib/pareton-deploy/release-state.json'))$1)"; }

REPO=/opt/pareton
OPS=/usr/local/lib/pareton-ops
SHIPPED=/var/log/vector-shipped.log
STATE=/var/lib/pareton-deploy/release-state.json

echo "=== setup ==="
git config --system --add safe.directory "$REPO" 2>/dev/null
git config --system --add safe.directory "$REPO/.git" 2>/dev/null
git config --system --add safe.directory /srv/origin.git 2>/dev/null
BASE_SHA=$(git -C "$REPO" rev-parse HEAD)
echo "baseline: $BASE_SHA"

printf '[Unit]\nDescription=docker stub for isolated acceptance\n[Service]\nType=oneshot\nExecStart=/bin/true\n' \
  > /etc/systemd/system/docker.service
install -d /var/lib/vector
install -d "$REPO/.venv/bin" "$REPO/.venv/standins" "$REPO/.venv/lib/python3.12/site-packages"

# venv stand-in: long-running modules run real probe/http stand-ins, quick
# helpers exit 0, -c probes run the real interpreter against the sqlite fake.
cat > "$REPO/.venv/bin/python" <<'SH'
#!/bin/sh
if [ "$1" = "-m" ]; then
  case "$2" in
    worker.main) exec /usr/bin/python3 /opt/pareton/.venv/standins/worker.py pareton-worker ;;
    api) exec /usr/bin/python3 /opt/pareton/.venv/standins/api.py ;;
    worker.watcher) exec /usr/bin/python3 /opt/pareton/.venv/standins/worker.py pareton-watcher ;;
    weights*) exec /usr/bin/python3 /opt/pareton/.venv/standins/worker.py pareton-weights ;;
    gpu) : > /tmp/gpu-reap-ran; exit 0 ;;
    *) exit 0 ;;
  esac
fi
exec /usr/bin/python3 "$@"
SH
chmod 0755 "$REPO/.venv/bin/python"
printf '#!/bin/sh\nexit 0\n' > "$REPO/.venv/bin/pip"
chmod 0755 "$REPO/.venv/bin/pip"
printf 'home = /usr/bin\n' > "$REPO/.venv/pyvenv.cfg"

# Stand-ins run the REAL probe loop (stdlib only) so in-process probes,
# ExecCondition gating, and stop behavior are production code paths.
cat > "$REPO/.venv/standins/worker.py" <<'PY'
import logging
import sys

sys.path.insert(0, "/opt/pareton")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
from observability.probe import run_probe_loop

run_probe_loop(sys.argv[1])
PY
cat > "$REPO/.venv/standins/api.py" <<'PY'
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/opt/pareton")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
from observability.probe import run_probe_loop


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/health", "/v1/campaigns"):
            body = json.dumps({"ok": True, "campaigns": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


threading.Thread(target=run_probe_loop, args=("pareton-api",), daemon=True).start()
HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
PY

cat > "$REPO/db/connection.py" <<'PY'
import os
import sqlite3
from contextlib import contextmanager

@contextmanager
def db_connection():
    if os.environ.get("TEST_DB_ERROR") == "1":
        raise RuntimeError("probe error injected")
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE submission_jobs (id INT, submission_id TEXT,"
        " attempts INT, phase TEXT, heartbeat_at TEXT, status TEXT)")
    conn.execute(
        "CREATE TABLE rounds (id TEXT, ordinal INT, campaign_id TEXT,"
        " heartbeat_at TEXT, status TEXT)")
    class Cursor:
        def __init__(self):
            self.result = None
        def execute(self, sql, args=None):
            self.result = conn.execute(sql.replace("%s", "?"), args or [])
        def fetchone(self):
            return self.result.fetchone() if self.result else None
        def fetchall(self):
            return self.result.fetchall() if self.result else []
    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()
    try:
        yield Connection()
    finally:
        conn.close()
PY
: > "$REPO/db/__init__.py"

cat > "$REPO/.env" <<'ENV'
PARETON_DISCORD_DEPLOY_WEBHOOK=https://pareton-isolated-test.invalid/webhook
PARETON_AXIOM_TOKEN=isolated-test-token
ENV
chown root:root "$REPO/.env"
chmod 0600 "$REPO/.env"

# Container-only Vector config: keep the production axiom sink (fake token,
# disk-buffered retries) AND mirror every parsed lifecycle event to a file
# the mock Axiom server reads from.
cat > "$REPO/ops/vector/vector.toml" <<'TOML'
data_dir = "/var/lib/vector"

[sources.journald]
type = "journald"
include_units = ["pareton-worker", "pareton-round-worker", "pareton-watcher", "pareton-api", "pareton-weights", "pareton-gpu-reap", "pareton-deploy", "pareton-deploy-failed"]

[transforms.drop_noise]
type = "filter"
inputs = ["journald"]
condition = '''
  contains(string!(.message), "\"event\":") || !match(string!(.message), r' DEBUG  ')
'''

[transforms.parse_lifecycle]
type = "remap"
inputs = ["drop_noise"]
source = '''
  msg = string!(.message)
  if contains(msg, "\"event\":") {
    extracted, err = parse_regex(msg, r'(?P<json>\{.*\})$')
    if err == null {
      parsed, err = parse_json(extracted.json)
      if err == null && is_object(parsed) {
        . = merge!(., parsed)
      }
    }
  }
'''

[sinks.axiom]
type = "axiom"
inputs = ["parse_lifecycle"]
token = "${PARETON_AXIOM_TOKEN}"
dataset = "pareton-prod"

[sinks.axiom.buffer]
type = "disk"
max_size = 536870912

[sinks.mirror]
type = "file"
inputs = ["parse_lifecycle"]
path = "/var/log/vector-shipped.log"
encoding.codec = "json"
TOML

# Managed drop-in wiring the checker to the in-container mock Axiom.
install -d "$REPO/ops/systemd/pareton-deploy.service.d"
cat > "$REPO/ops/systemd/pareton-deploy.service.d/isolation.conf" <<'CONF'
[Service]
Environment=PARETON_AXIOM_API_URL=http://127.0.0.1:9421
Environment=PARETON_AXIOM_QUERY_TOKEN=isolated-test-token
Environment=PARETON_LOG_WAIT_BUDGET_S=90
CONF

# Mock Axiom: answers APL queries by replaying the file-sink mirror. An
# optional DROP_UNIT env removes one source to inject B15-style faults.
cat > /root/mock-axiom.py <<'PY'
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

SHIPPED = "/var/log/vector-shipped.log"
UNIT_KEYS = ("_SYSTEMD_UNIT", "systemd.unit", "_systemd_unit", "unit")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            # The checker must authenticate: a missing token is exactly the
            # class of bug the mock exists to catch.
            self.send_response(401)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        match = re.search(r"probe_id == '([^']+)'", body.get("apl", ""))
        invocation = re.search(r"invocation_id == '([^']+)'", body.get("apl", ""))
        drop = os.environ.get("DROP_UNIT", "")
        units = set()
        try:
            with open(SHIPPED) as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    unit = next(
                        (str(event[k]) for k in UNIT_KEYS if event.get(k)), ""
                    )
                    if invocation:
                        if str(event.get("invocation_id", "")) == invocation.group(1):
                            units.add(unit)
                        continue
                    if not match:
                        continue
                    if str(event.get("probe_id", "")) == match.group(1):
                        units.add(unit)
        except FileNotFoundError:
            pass
        units.discard("")
        if drop:
            units.discard(drop)
        payload = {
            "status": {"isPartial": False},
            "tables": [
                {
                    "columns": [{"name": "_SYSTEMD_UNIT"}],
                    "rows": [[u] for u in sorted(units)],
                }
            ],
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", 9421), Handler).serve_forever()
PY
nohup python3 /root/mock-axiom.py >/tmp/mock-axiom.log 2>&1 &

git clone --bare -q "$REPO" /srv/origin.git
git --git-dir=/srv/origin.git update-ref refs/heads/main "$BASE_SHA"
git -C "$REPO" remote add origin /srv/origin.git 2>/dev/null || git -C "$REPO" remote set-url origin /srv/origin.git
git -C "$REPO" fetch -q origin main
git -C "$REPO" add -A
git -C "$REPO" -c user.email=isolated@test -c user.name=isolated \
  commit -qm "isolated: stage2 stand-ins, mirror sink, checker drop-in"
STANDIN_SHA=$(git -C "$REPO" rev-parse HEAD)

# Install ops helpers + units BEFORE the first tick (bootstrap order).
install -d -m 0755 "$OPS"
for f in ops_common.py sync-config.py notify-deploy-failure.py release.py; do
  install -m 0755 "$REPO/ops/$f" "$OPS/$f"
done
install -m 0755 "$REPO/ops/deploy.sh" /usr/local/bin/pareton-deploy
python3 "$OPS/sync-config.py" apply --repo "$REPO" >/tmp/s2-apply.json 2>/tmp/s2-apply.err \
  && pass "setup sync apply ok" || fail "setup sync apply: $(cat /tmp/s2-apply.err)"
[ "$(systemctl is-active vector)" = active ] && pass "setup vector active" || fail "setup vector not active"

echo "=== S1: bootstrap via request reset ==="
"$OPS/release.py" request reset \
  --baseline-commit "$STANDIN_SHA" \
  --confirm-evidence "isolated bootstrap: clean checkout at standin commit" \
  --operator isolated >/dev/null \
  && pass "S1 reset registered" || fail "S1 reset registration refused"
systemctl start pareton-deploy.service
RC=$?
# The first verify must fail on notification-acceptance (no drill yet);
# business units must nevertheless be up and healthy.
[ "$RC" -ne 0 ] && pass "S1 bootstrap verify failed without drill (rc=$RC)" || fail "S1 bootstrap unexpectedly passed without drill"
python3 - <<'PY' && pass "S1 state stuck at verifying/log-accepted=false" || fail "S1 state wrong"
import json
state = json.load(open("/var/lib/pareton-deploy/release-state.json"))
assert state["phase"] == "verifying", state
assert state["startup_complete"] is True, state
assert state["log_accepted"] is False, state
assert state["hold"] is not None, state
PY
for u in pareton-api pareton-watcher pareton-weights pareton-worker pareton-round-worker; do
  [ "$(systemctl is-active "$u")" = active ] && pass "S1 $u active after bootstrap verify" \
    || fail "S1 $u not active"
done
http_ok() { python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000$1', timeout=5).status == 200 else 1)"; }
http_ok /health && pass "S1 api /health ok" || fail "S1 api /health failed"
http_ok /v1/campaigns && pass "S1 api /v1/campaigns ok" || fail "S1 api /v1/campaigns failed"

echo "=== S2: failure drill + acceptance record + verify ==="
# The drill must be a REAL deploy failure; under hold the tick is read-only,
# so induce it with a corrupt state file (B21 path) and restore afterwards.
cp "$STATE" /tmp/state.s2
printf '{"schema_version": 1}' > "$STATE"
systemctl start pareton-deploy.service
RC=$?
[ "$RC" -ne 0 ] && pass "S2 drill deploy failed (rc=$RC)" || fail "S2 drill deploy unexpectedly passed"
DRILL_INV=$(systemctl show pareton-deploy -p InvocationID --value)
[ -n "$DRILL_INV" ] || fail "S2 no drill invocation id"
sleep 3
journalctl -u pareton-deploy-failed.service --no-pager | grep -q "notify" \
  && pass "S2 OnFailure notifier ran" || fail "S2 notifier never ran"
wait_for 30 "$SHIPPED" "deploy_failure_notified" \
  && pass "S2 structured notify event reached sink" \
  || fail "S2 structured notify event never shipped"
cp /tmp/state.s2 "$STATE"
PARETON_AXIOM_API_URL=http://127.0.0.1:9421 "$OPS/release.py" record-notification-acceptance \
  --invocation "$DRILL_INV" --message-id iso-drill-0001 --confirmed-by isolated \
  && pass "S2 acceptance recorded with Axiom evidence" \
  || fail "S2 acceptance recording refused"
"$OPS/release.py" request verify --operator isolated >/dev/null \
  && pass "S2 verify registered" || fail "S2 verify registration failed"
systemctl start pareton-deploy.service
RC=$?
[ "$RC" = 0 ] && pass "S2 verify completed the release (rc=0)" || fail "S2 verify deploy rc=$RC"
python3 - <<'PY' && pass "S2 state idle, baseline verified, hold kept" || fail "S2 final state wrong"
import json
state = json.load(open("/var/lib/pareton-deploy/release-state.json"))
assert state["phase"] == "idle", state
assert state["verified_commit"] == state["target_commit"], state
assert state["hold"] is not None, state  # verify never clears hold
PY

echo "=== S3: unpause, then full release A->B ==="
echo "# stage2 target change" >> "$REPO/requirements.txt"
printf 'print("stage2 B")\n' > "$REPO/stage2_marker.py"
git -C "$REPO" add stage2_marker.py
git -C "$REPO" -c user.email=isolated@test -c user.name=isolated \
  commit -qam "isolated: stage2 target B"
git -C "$REPO" push -q origin HEAD:refs/heads/main
TARGET_B=$(git -C "$REPO" rev-parse HEAD)
# A-era venv marker: it must ride INTO the recovery copy the release saves,
# so S8 can prove the rollback restored the copy's contents (not "nothing").
touch "$REPO/.venv/bin/stage2-a-era-marker"
chmod 0755 "$REPO/.venv/bin/stage2-a-era-marker"
"$OPS/release.py" request unpause --main-commit "$TARGET_B" \
  --operator isolated >/dev/null && pass "S3 unpause registered" || fail "S3 unpause refused"
systemctl start pareton-deploy.service
[ "$?" = 0 ] && pass "S3 unpause executed" || fail "S3 unpause deploy rc!=0"
systemctl start pareton-deploy.service
RC=$?
[ "$RC" = 0 ] && pass "S3 full release A->B (rc=0)" || fail "S3 release rc=$RC: $(journalctl -u pareton-deploy -n 40 --no-pager | tail -8)"
TARGET_B="$TARGET_B" python3 - <<'PY' && pass "S3 state verified at B" || fail "S3 state wrong after release"
import json
import os
state = json.load(open("/var/lib/pareton-deploy/release-state.json"))
assert state["phase"] == "idle", state
assert state["verified_commit"] == os.environ["TARGET_B"], state
assert state["hold"] is None, state
PY
[ "$(cat "$REPO/.deploy-done")" = "$TARGET_B" ] \
  && pass "S3 alias matches state" || fail "S3 alias != verified commit"
[ "$(cat "$REPO/stage2_marker.py")" = 'print("stage2 B")' ] \
  && pass "S3 checkout moved to B" || fail "S3 checkout not at B"
ls "$REPO/../pareton-deploy/recovery" >/dev/null 2>&1 || true
[ -d /var/lib/pareton-deploy/recovery ] && pass "S3 recovery copy kept" || fail "S3 no recovery copy"
wait_for 30 "$SHIPPED" "maintenance_finished" \
  && pass "S3 maintenance_finished event shipped" \
  || fail "S3 maintenance_finished never shipped"

echo "=== S4: gate matrix on real systemd ==="
systemctl stop pareton-api 2>/dev/null || true
python3 - <<'PY'
import json
path = "/var/lib/pareton-deploy/release-state.json"
state = json.load(open(path))
state["phase"] = "applying"
json.dump(state, open(path, "w"))
PY
systemctl start pareton-api 2>/dev/null
sleep 2
ACT=$(systemctl is-active pareton-api)
RESULT=$(systemctl show pareton-api -p Result --value)
[ "$ACT" != active ] && [ "$RESULT" != failed ] \
  && pass "S4 applying: ExecCondition=1 skips without failing ($ACT/$RESULT)" \
  || fail "S4 applying: unit active=$ACT result=$RESULT"
python3 - <<'PY'
import json
path = "/var/lib/pareton-deploy/release-state.json"
state = json.load(open(path))
state["phase"] = "verifying"
state["startup_complete"] = True
json.dump(state, open(path, "w"))
PY
systemctl start pareton-api
sleep 2
[ "$(systemctl is-active pareton-api)" = active ] \
  && pass "S4 verifying(started): startup allowed" || fail "S4 verifying: api did not start"
cp "$STATE" /tmp/state.bak
printf '{"schema_version": 99}' > "$STATE"
systemctl stop pareton-api 2>/dev/null || true
systemctl reset-failed pareton-api 2>/dev/null || true
systemctl start pareton-api 2>/dev/null
sleep 2
ACT=$(systemctl is-active pareton-api)
[ "$ACT" = failed ] \
  && pass "S4 corrupt state: gate exits 255, unit failed" \
  || fail "S4 corrupt state: is-active=$ACT"
journalctl -u pareton-api -n 10 --no-pager | grep -q "release_gate_error" \
  && pass "S4 release_gate_error in journal" || fail "S4 no release_gate_error event"
cp /tmp/state.bak "$STATE"
systemctl reset-failed pareton-api 2>/dev/null || true
systemctl start pareton-api
sleep 1

echo "=== S5: worker activity lock holds the tick (B1/B2 slice) ==="
cat > /tmp/holder.py <<'PY'
import os
import sys
import time
sys.path.insert(0, "/opt/pareton")
os.environ["PARETON_COORDINATION"] = "1"
os.environ["PARETON_ACTIVITY_LOCK"] = "/run/pareton-activity.lock"
os.environ["PARETON_RELEASE_STATE"] = "/var/lib/pareton-deploy/release-state.json"
from worker import coordination

with coordination.claim_guard():
    open("/tmp/holder-ready", "w").write("1")
    time.sleep(45)
PY
rm -f /tmp/holder-ready
nohup python3 /tmp/holder.py >/tmp/holder.log 2>&1 &
HOLDER_PID=$!
for i in $(seq 1 10); do [ -f /tmp/holder-ready ] && break; sleep 1; done
[ -f /tmp/holder-ready ] && pass "S5 holder acquired shared lock" || fail "S5 holder never started"
python3 - <<'PY'
import json
path = "/var/lib/pareton-deploy/release-state.json"
state = json.load(open(path))
import datetime
state.update({"phase": "draining", "phase_since": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "target_commit": "deadbeef", "scope": "full"})
json.dump(state, open(path, "w"))
PY
MARKER_BEFORE=$(cat "$REPO/.deploy-done")
systemctl start pareton-deploy.service
RC=$?
[ "$RC" = 0 ] && pass "S5 tick exits 0 while work is active (<30min)" || fail "S5 tick rc=$RC"
grep -q "last_step=active-work" /var/lib/pareton-deploy/last-run.env \
  && pass "S5 last-run reports active-work" || fail "S5 step: $(grep last_step /var/lib/pareton-deploy/last-run.env)"
[ "$(cat "$REPO/.deploy-done")" = "$MARKER_BEFORE" ] \
  && pass "S5 no environment write during active work" || fail "S5 environment moved!"
kill "$HOLDER_PID" 2>/dev/null || true
wait "$HOLDER_PID" 2>/dev/null || true
cp /tmp/state.bak "$STATE"
python3 - <<'PY'
import json
path = "/var/lib/pareton-deploy/release-state.json"
state = json.load(open(path))
state["phase"] = "idle"
json.dump(state, open(path, "w"))
PY

echo "=== S6: gpu-reap one-shot dispatch (B18) ==="
python3 - <<'PY'
import json
path = "/var/lib/pareton-deploy/release-state.json"
state = json.load(open(path))
state["phase"] = "verifying"
state["startup_complete"] = True
json.dump(state, open(path, "w"))
PY
INV=$(systemctl show pareton-deploy -p InvocationID --value)
python3 - "$INV" <<'PY'
import json
import sys
import datetime
request = {
    "invocation_id": sys.argv[1],
    "op_id": json.load(open("/var/lib/pareton-deploy/release-state.json"))["op_id"],
    "probe_id": "gpu-probe-1",
    "issued_at": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"),
    "consumed": False,
}
json.dump(request, open("/run/pareton-deploy/gpu-reap-request.json", "w"))
PY
rm -f /tmp/gpu-reap-ran
systemctl start pareton-gpu-reap.service
RC=$?
sleep 2
[ "$RC" = 0 ] && [ ! -e /tmp/gpu-reap-ran ] \
  && pass "S6 valid request consumed: probe without real reap" \
  || fail "S6 valid request path ran the real reap (rc=$RC)"
journalctl -u pareton-gpu-reap -n 10 --no-pager | grep -q "gpu_reap_probe_warning" \
  && pass "S6 dispatcher warning logged" || fail "S6 no warning event"
systemctl start pareton-gpu-reap.service
sleep 2
[ -e /tmp/gpu-reap-ran ] \
  && pass "S6 second run (consumed) performs the real reap" \
  || fail "S6 consumed request still swallowed the reap"
cp /tmp/state.bak "$STATE"
python3 - <<'PY'
import json
path = "/var/lib/pareton-deploy/release-state.json"
state = json.load(open(path))
state["phase"] = "idle"
json.dump(state, open(path, "w"))
PY

echo "=== S7: missing-source fault injection (B15) ==="
pkill -f mock-axiom.py 2>/dev/null || true
sleep 1
DROP_UNIT=pareton-gpu-reap.service nohup python3 /root/mock-axiom.py >/tmp/mock-axiom2.log 2>&1 &
PARETON_AXIOM_API_URL=http://127.0.0.1:9421 "$OPS/release.py" check-logs --probe-id gpu-probe-1 --target "$(git -C "$REPO" rev-parse HEAD)" \
  >/tmp/s7-report.json 2>/tmp/s7-err
RC=$?
[ "$RC" = 1 ] && pass "S7 missing source -> exit 1" || fail "S7 exit=$RC"
grep -q "pareton-gpu-reap.service" /tmp/s7-report.json \
  && pass "S7 report names the dropped unit" \
  || fail "S7 report: $(cat /tmp/s7-report.json)"
pkill -f mock-axiom.py 2>/dev/null || true
sleep 1
nohup python3 /root/mock-axiom.py >/tmp/mock-axiom3.log 2>&1 &

echo "=== S8: rollback with a real recovery copy (B12/B23 slice) ==="
# The pip stand-in changes nothing, so simulate B-era venv drift with a
# marker only the live .venv has; a correct rollback must remove it by
# restoring the recovery copy (not by re-resolving deps).
touch "$REPO/.venv/bin/stage2-b-only-marker"
chmod 0755 "$REPO/.venv/bin/stage2-b-only-marker"
"$OPS/release.py" request rollback --reason "isolated B12 drill" --operator isolated \
  && pass "S8 rollback registered" || fail "S8 rollback refused"
systemctl start pareton-deploy.service
RC=$?
[ "$RC" = 0 ] && pass "S8 rollback completed (rc=0)" \
  || fail "S8 rollback rc=$RC: $(journalctl -u pareton-deploy -n 40 --no-pager | tail -8)"
STANDIN_SHA="$STANDIN_SHA" python3 - <<'PY' && pass "S8 state verified back at A, hold kept" || fail "S8 state wrong after rollback"
import json
import os

state = json.load(open("/var/lib/pareton-deploy/release-state.json"))
assert state["phase"] == "idle", state
assert state["verified_commit"] == os.environ["STANDIN_SHA"], state
assert state["hold"] is not None, state  # rollback keeps the pause
PY
[ "$(git -C "$REPO" rev-parse HEAD)" = "$STANDIN_SHA" ] \
  && pass "S8 checkout back at A" || fail "S8 checkout not at A"
[ "$(cat "$REPO/.deploy-done")" = "$STANDIN_SHA" ] \
  && pass "S8 alias back at A" || fail "S8 alias != A"
[ ! -e "$REPO/.venv/bin/stage2-b-only-marker" ] \
  && pass "S8 venv restored from recovery copy (B-era marker gone)" \
  || fail "S8 venv not restored (B-era marker still present)"
[ -e "$REPO/.venv/bin/stage2-a-era-marker" ] \
  && pass "S8 venv contents came from the copy (A-era marker present)" \
  || fail "S8 A-era marker missing: copy not actually restored"
[ ! -e "$REPO/stage2_marker.py" ] \
  && pass "S8 B-era file gone from checkout" || fail "S8 B-era file still in checkout"

echo
if [ "$FAILED" = 0 ]; then
  echo "ALL-STAGE2-ISOLATED-ACCEPTANCE-PASSED"
else
  echo "STAGE2-ISOLATED-ACCEPTANCE-HAD-FAILURES"
fi
exit "$FAILED"
