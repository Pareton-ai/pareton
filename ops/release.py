#!/usr/bin/env python3
"""Pareton stage-2 release coordinator (spec: docs/第二阶段发布安全-spec.md).

Standard library only (system Python >= 3.11 for tomllib): it must run
without the application venv. Owns the release state machine, both
coordination locks, and every deployment write path. ops/deploy.sh is a
thin wrapper around ``release.py tick``.

Subcommands and exit codes are the normative contract of spec section 4.5:
  tick            state-machine advance driven by pareton-deploy.timer
  gate            ExecCondition checker: 0 allow, 1 expected block, 255 corrupt
  emit-probe      ExecStartPre probe emitter
  gpu-reap-dispatch  one-shot log-only probe dispatch, else exec real command
  status          read-only state dump
  request         register hold/unpause/reset/resume/cancel/verify/rollback/
                  vector-repair (spec 6.3)
  record-notification-acceptance  drill evidence registrar (spec 7.4)
  check-logs      per-unit Axiom ingestion check (spec 7.3)

PARETON_RELEASE_BASE remaps every absolute target path under a prefix for
tests and isolated acceptance, following the stage-1 helpers' convention.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops_common import now_iso, parse_env_file, parse_iso, read_json, write_json_atomic

SCHEMA_VERSION = 2
PHASES = ("idle", "draining", "quiescing", "applying", "verifying", "verified")
SCOPES = ("full", "vector-only")
DIRECTIONS = ("forward", "rollback", "reset")

RESIDENT_UNITS = ("pareton-api", "pareton-watcher", "pareton-weights")
WORKER_UNITS = ("pareton-worker", "pareton-round-worker")
ALL_RESIDENT = RESIDENT_UNITS + WORKER_UNITS
MAINT_TIMERS = ("pareton-gpu-reap.timer", "pareton-builder-cleanup.timer")
ONESHOT_UNITS = ("pareton-gpu-reap.service", "pareton-builder-cleanup.service")
DEPLOY_TIMER = "pareton-deploy.timer"
DEPLOY_UNIT = "pareton-deploy.service"

# The one include_units source with no per-release probe: real OnFailure
# drills cover it (spec 7.1). Everything else must have a probe method.
PROBE_EXEMPT_UNITS = ("pareton-deploy-failed.service",)
PROBE_KNOWN_UNITS = (
    tuple(f"{u}.service" for u in (*ALL_RESIDENT, "pareton-deploy", "pareton-gpu-reap"))
    + PROBE_EXEMPT_UNITS
)

# Files whose content change requires a new real failure drill (spec 7.4).
NOTIFICATION_CHAIN_FILES = (
    "ops/notify-deploy-failure.py",
    "ops/deploy.sh",
    "ops/release.py",
    "ops/ops_common.py",
    "ops/systemd/pareton-deploy.service",
    "ops/systemd/pareton-deploy-failed.service",
    "ops/vector/vector.service",
)

STOP_ALERT_S = {  # service-stopping thresholds (spec 5.2)
    "pareton-api": 600,
    "pareton-watcher": 600,
    "pareton-weights": 1800,
    "pareton-worker": 1800,
    "pareton-round-worker": 1800,
}
DB_PROBE_TIMEOUT_S = 30
API_HEALTH_TIMEOUT_S = 60


def p(absolute: str) -> Path:
    base = os.environ.get("PARETON_RELEASE_BASE", "")
    return Path(base + absolute) if base else Path(absolute)


def repo() -> Path:
    return Path(os.environ.get("PARETON_REPO", "/opt/pareton"))


def state_dir() -> Path:
    return p(os.environ.get("PARETON_STATE_DIR", "/var/lib/pareton-deploy"))


def state_path() -> Path:
    return state_dir() / "release-state.json"


def request_path() -> Path:
    return state_dir() / "release-request.json"


def state_lock_path() -> Path:
    return state_dir() / "release-state.lock"


def deploy_lock_path() -> Path:
    return p(os.environ.get("PARETON_DEPLOY_LOCK", "/run/pareton-deploy.lock"))


def activity_lock_path() -> Path:
    return p(os.environ.get("PARETON_ACTIVITY_LOCK", "/run/pareton-activity.lock"))


def coord_dir() -> Path:
    return p("/run/pareton-deploy")


def probe_path() -> Path:
    return coord_dir() / "probe.json"


def gpu_request_path() -> Path:
    return coord_dir() / "gpu-reap-request.json"


def acceptance_path() -> Path:
    return state_dir() / "notification-acceptance.json"


def alias_path() -> Path:
    return repo() / ".deploy-done"


def ops_dir() -> Path:
    return Path(os.environ.get("PARETON_OPS_DIR", "/usr/local/lib/pareton-ops"))


def venv_python() -> Path:
    return repo() / ".venv" / "bin" / "python"


def drain_wait_s() -> int:
    return int(os.environ.get("PARETON_DRAIN_WAIT_S", "1800"))


def log_budget_s() -> int:
    return int(os.environ.get("PARETON_LOG_WAIT_BUDGET_S", "120"))


class Fail(Exception):
    def __init__(self, code: int, reason: str, **extra):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.extra = extra


def emit_event(event: str, **fields) -> None:
    """Single-line JSON lifecycle event on stdout (journald -> Vector)."""
    payload = {"event": event, **{k: v for k, v in fields.items() if v is not None}}
    print(json.dumps(payload, default=str, separators=(",", ":")), flush=True)


# ---------------------------------------------------------------------------
# Command choke point (monkeypatched in tests)


def run_cmd(
    argv: list[str],
    env: dict | None = None,
    timeout: int = 300,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
    )


def http_post_json(
    url: str, payload: dict, timeout: int
) -> tuple[int, dict | None, str | None]:
    """POST JSON; returns (status_code, parsed_body_or_None, error_category)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode() or "{}")
            return response.status, body if isinstance(body, dict) else None, None
    except urllib.error.HTTPError as exc:
        return exc.code, None, f"http-{exc.code}"
    except urllib.error.URLError as exc:
        return 0, None, type(exc.reason).__name__ if exc.reason else "URLError"
    except (ValueError, OSError) as exc:
        return 0, None, type(exc).__name__


# ---------------------------------------------------------------------------
# State


def validate_state(data) -> dict | None:
    """Return the state dict when it satisfies the spec 4.5 schema, else None."""
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    if data.get("phase") not in PHASES or data.get("scope") not in SCOPES:
        return None
    for key in ("op_id", "from_commit", "target_commit"):
        if not isinstance(data.get(key), str) or not data[key]:
            return None
    if data.get("direction", "forward") not in DIRECTIONS:
        return None
    verified = data.get("verified_commit")
    if not isinstance(verified, str) or not verified:
        return None
    return data


def load_state() -> dict | None:
    return validate_state(read_json(state_path()))


@contextmanager
def state_lock():
    import ops_common

    with ops_common.locked(state_lock_path()):
        yield


def mutate_state(mutator, *, required: bool = True) -> dict:
    """Read-modify-write the state under the short state lock, from disk."""
    with state_lock():
        raw = read_json(state_path())
        state = validate_state(raw) if raw is not None else None
        if state is None:
            if required:
                raise Fail(2, "state-corrupt")
            state = raw if isinstance(raw, dict) else {}
        mutator(state)
        state["updated_at"] = now_iso()
        write_json_atomic(state_path(), state)
        return state


_TICK_STARTED_AT: str | None = None


def record_step(step: str, extra: dict | None = None) -> None:
    """Progress lines for the failure notifier (same env-file format).

    started_at is this tick's own start time: the notifier treats
    last_success > started_at as "already recovered" and stays silent, so a
    state-derived timestamp would swallow early-tick failures (CR P1-5).
    """
    state = read_json(state_path()) or {}
    lines = {
        "invocation_id": os.environ.get("INVOCATION_ID", "manual"),
        "started_at": _TICK_STARTED_AT or now_iso(),
        "from_commit": state.get("from_commit", "unknown"),
        "target_commit": state.get("target_commit", "unknown"),
        "last_step": step,
    }
    if extra:
        lines.update(extra)
    path = state_dir() / "last-run.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{k}={v}\n" for k, v in lines.items())
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".last-run.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def gate_open(state: dict | None) -> tuple[bool, str]:
    """Claim-gate half of the spec 4.5 matrix."""
    if state is None:
        return False, "state-corrupt"
    if state["scope"] == "vector-only":
        return True, "vector-only"
    phase = state["phase"]
    if phase in ("idle", "verified"):
        return True, phase
    if phase == "verifying":
        if state.get("startup_complete") is True:
            return True, "verifying-started"
        return False, "phase=verifying"
    return False, f"phase={phase}"


def exec_allow(state: dict | None) -> tuple[bool, str]:
    """ExecCondition half of the spec 4.5 matrix."""
    if state is None:
        return False, "state-corrupt"
    if state["scope"] == "vector-only":
        return True, "vector-only"
    phase = state["phase"]
    if phase in ("idle", "draining", "verifying", "verified"):
        return True, phase
    return False, f"phase={phase}"


# ---------------------------------------------------------------------------
# systemctl / git / venv helpers


_STOPPED_IS_ACTIVE = ("inactive", "failed")


def unit_is_stopped(unit: str) -> bool:
    """systemctl is-active says stopped only for inactive/failed.

    "deactivating"/"activating"/"reloading" are still running: treating them
    as stopped let applies start while a stop was mid-flight (CR P1-3).
    """
    result = run_cmd(["systemctl", "is-active", unit], timeout=30)
    return result.stdout.strip() in _STOPPED_IS_ACTIVE


def unit_is_active(unit: str) -> bool:
    return not unit_is_stopped(unit)


def unit_is_enabled(unit: str) -> bool:
    result = run_cmd(["systemctl", "is-enabled", unit], timeout=30)
    return result.stdout.strip() == "enabled"


def snapshot_units() -> dict:
    snapshot: dict = {}
    for unit in (*ALL_RESIDENT, *MAINT_TIMERS, DEPLOY_TIMER):
        snapshot[unit] = {
            "active": unit_is_active(unit),
            "enabled": unit_is_enabled(unit),
        }
    return snapshot


def stop_unit(unit: str) -> None:
    run_cmd(["systemctl", "stop", "--no-block", unit], timeout=60)


def start_unit(unit: str) -> None:
    run_cmd(["systemctl", "start", unit], timeout=120)


def git(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return run_cmd(["git", "-C", str(repo()), *args], timeout=timeout)


def git_out(*args: str) -> str:
    result = git(*args)
    if result.returncode != 0:
        raise Fail(2, f"git-{args[0]}-failed", stderr=result.stderr.strip()[:400])
    return result.stdout.strip()


def db_running_records() -> dict:
    """Running rounds/submissions via the application venv (stdlib caller)."""
    probe = (
        "import json\n"
        "try:\n"
        "    from db.connection import db_connection\n"
        "    out={'error':None,'rounds':[],'submissions':[]}\n"
        "    with db_connection() as conn:\n"
        "        with conn.cursor() as cur:\n"
        '            cur.execute("SELECT id, ordinal, campaign_id, heartbeat_at"\n'
        "                        \" FROM rounds WHERE status='running'\")\n"
        "            out['rounds']=[{'id':str(r[0]),'ordinal':r[1],\n"
        "                           'campaign_id':str(r[2]),'heartbeat_at':str(r[3])}\n"
        "                          for r in cur.fetchall()]\n"
        '            cur.execute("SELECT j.id, j.submission_id, j.attempts, j.phase,"\n'
        '                        " j.heartbeat_at FROM submission_jobs j"\n'
        "                        \" WHERE j.status='running'\")\n"
        "            out['submissions']=[{'id':r[0],'submission_id':str(r[1]),\n"
        "                                'attempts':r[2],'phase':r[3],\n"
        "                                'heartbeat_at':str(r[4])} for r in cur.fetchall()]\n"
        "except Exception as exc:\n"
        "    out={'error':type(exc).__name__}\n"
        "print(json.dumps(out))\n"
    )
    try:
        result = run_cmd(
            [str(venv_python()), "-c", probe],
            timeout=DB_PROBE_TIMEOUT_S,
            cwd=repo(),
            env={"PYTHONPATH": str(repo())},
        )
    except subprocess.TimeoutExpired:
        return {"error": "db-probe-timeout"}
    if result.returncode != 0:
        return {"error": "db-probe-exit", "detail": result.returncode}
    try:
        data = json.loads(result.stdout)
    except ValueError:
        return {"error": "db-probe-output"}
    return data if isinstance(data, dict) else {"error": "db-probe-output"}


# ---------------------------------------------------------------------------
# Locks


class DeployLock:
    """The deploy mutual-exclusion lock; inherited across re-exec."""

    def __init__(self, fd: int):
        self.fd = fd

    @classmethod
    def acquire(cls) -> DeployLock | None:
        try:
            fd = os.open(str(deploy_lock_path()), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            raise Fail(2, "deploy-lock-open", detail=type(exc).__name__) from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None  # a peer deploy is running; tick exits silently
        except OSError as exc:
            os.close(fd)
            raise Fail(2, "deploy-lock-flock", detail=type(exc).__name__) from exc
        os.set_inheritable(fd, True)
        return cls(fd)

    @classmethod
    def adopt(cls, fd: int) -> DeployLock:
        lock = cls(fd)
        lock.assert_held()
        return lock

    def assert_held(self) -> None:
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # same OFD: no-op

    def release(self) -> None:
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


class ActivityLock:
    """The worker activity lock, exclusive side (spec 5.1)."""

    def __init__(self, fd: int):
        self.fd = fd

    @classmethod
    def try_acquire(cls) -> ActivityLock | None:
        coord_dir().mkdir(parents=True, exist_ok=True)
        fd = os.open(str(activity_lock_path()), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        os.set_inheritable(fd, True)
        return cls(fd)

    @classmethod
    def adopt(cls, fd: int) -> ActivityLock:
        lock = cls(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock

    def release(self) -> None:
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


# ---------------------------------------------------------------------------
# Probes (spec 7.2 / 4.5)


def write_probe(probe_id: str, target: str) -> dict:
    coord_dir().mkdir(parents=True, exist_ok=True)
    probe = {
        "probe_id": probe_id,
        "target_commit": target,
        "host": platform.node(),
        "issued_at": now_iso(),
    }
    with state_lock():
        write_json_atomic(probe_path(), probe)
    return probe


def clear_coordination_files() -> None:
    for path in (probe_path(), gpu_request_path()):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def emit_deploy_probe(probe: dict) -> None:
    emit_event(
        "deployment_probe",
        probe_id=probe["probe_id"],
        unit="pareton-deploy.service",
        target_commit=probe["target_commit"],
        host=probe["host"],
    )


def gpu_reap_wait_s() -> int:
    return int(os.environ.get("PARETON_GPU_REAP_WAIT_S", "1800"))


def gpu_probe_flow(op_id: str, probe: dict) -> None:
    """One-shot GPU reap probe: pause timer, request, run, consume (7.2).

    Waiting for an in-flight reap is capped at 30 minutes (spec 4.4): on
    timeout the verification fails with a preparation-step report, the
    timer is restored, the target stays unaccepted, and business continues.
    """
    # The timer was stopped at quiescing for full releases; the vector-only
    # path stops it here.
    if unit_is_active("pareton-gpu-reap.timer"):
        stop_unit("pareton-gpu-reap.timer")
    pending = _wait_units_inactive(
        ONESHOT_UNITS[:1], budget_s=gpu_reap_wait_s(), step="gpu-reap-wait"
    )
    if pending:
        _restore_maint_timers(load_state())
        raise Fail(1, "gpu-reap-wait-timeout", units=pending)
    request = {
        "invocation_id": os.environ.get("INVOCATION_ID", "manual"),
        "op_id": op_id,
        "probe_id": probe["probe_id"],
        "issued_at": now_iso(),
        "consumed": False,
    }
    with state_lock():
        write_json_atomic(gpu_request_path(), request)
    start_unit("pareton-gpu-reap.service")
    _wait_units_inactive(
        ("pareton-gpu-reap.service",), budget_s=120, step="gpu-reap-wait"
    )


def _wait_units_inactive(units, budget_s: int, step: str) -> list[str]:
    deadline = time.monotonic() + budget_s
    while True:
        pending = [u for u in units if unit_is_active(u)]
        if not pending:
            return []
        if time.monotonic() >= deadline:
            return pending
        time.sleep(2)


# ---------------------------------------------------------------------------
# Axiom check-logs (spec 7.3) and notification acceptance (spec 7.4)


def axiom_query_token() -> tuple[str | None, str | None]:
    explicit = os.environ.get("PARETON_AXIOM_QUERY_TOKEN")
    if explicit:
        return explicit, None
    values, problems = parse_env_file(p("/opt/pareton/.env"))
    if problems:
        return None, ",".join(problems)
    token = values.get("PARETON_AXIOM_TOKEN", "")
    if not token:
        return None, "token-missing"
    return token, None


def installed_include_units() -> tuple[list[str], str | None]:
    import tomllib

    toml_path = p("/etc/vector/vector.toml")
    try:
        data = tomllib.loads(toml_path.read_text())
    except (OSError, ValueError) as exc:
        return [], f"toml-{type(exc).__name__}"
    try:
        units = data["sources"]["journald"]["include_units"]
        data["sinks"]["axiom"]["dataset"]  # presence check
    except (KeyError, TypeError):
        return [], "toml-missing-fields"
    normalized = sorted(u if u.endswith(".service") else f"{u}.service" for u in units)
    return normalized, None


def run_log_check(probe: dict, *, skip_acceptance: bool = False) -> tuple[int, dict]:
    """Poll Axiom until every expected source answered this probe_id.

    Returns (exit_code, report); 0 complete, 1 missing/acceptance-required,
    2 query/permission/parse failure (spec 7.3).
    """
    units, problem = installed_include_units()
    if problem:
        return 2, {"error": problem}
    expected = [u for u in units if u not in PROBE_EXEMPT_UNITS]
    unknown = [u for u in expected if u not in PROBE_KNOWN_UNITS]
    if unknown:
        return 2, {"error": "no-probe-method", "units": unknown}

    token, token_problem = axiom_query_token()
    if token is None:
        return 2, {"error": "axiom-token", "detail": token_problem}
    import tomllib

    dataset = tomllib.loads(p("/etc/vector/vector.toml").read_text())["sinks"]["axiom"][
        "dataset"
    ]
    base_url = os.environ.get("PARETON_AXIOM_API_URL", "https://api.axiom.co")
    url = f"{base_url}/v1/datasets/_apl?format=tabular"
    apl = (
        f"['{dataset}'] | where probe_id == '{probe['probe_id']}' "
        f"| where ['_SYSTEMD_UNIT'] != ''"
    )
    start = parse_iso(probe["issued_at"]) or parse_iso(now_iso())
    # Clock-skew margin on the query window (spec 7.3).
    import calendar

    start_epoch = calendar.timegm(start.utctimetuple()) - 300
    body = {
        "apl": apl,
        "startTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_epoch)),
        "endTime": None,  # filled per request below
    }

    report = {"probe_id": probe["probe_id"], "expected": expected, "received": []}
    deadline = time.monotonic() + log_budget_s()
    last_error: dict | None = None
    while True:
        remaining = max(1, int(deadline - time.monotonic()))
        request_timeout = min(10, remaining)
        body["endTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status, data, error = http_post_json(url, body, request_timeout)
        if status == 200 and data is not None:
            if _query_is_partial(data):
                last_error = {"error": "partial-result"}
            else:
                received = _units_in_response(data)
                report["received"] = sorted(set(received))
                last_error = None
                if set(expected) <= set(received):
                    break
        else:
            last_error = {"error": error or "bad-response", "status": status}
        if time.monotonic() >= deadline:
            break
        time.sleep(5)

    if last_error and set(expected) - set(report["received"]):
        # Only a query failure when nothing new arrived; distinguish per spec.
        if not report["received"]:
            report.update(last_error)
            report["missing"] = sorted(set(expected) - set(report["received"]))
            return 2, report
    report["missing"] = sorted(set(expected) - set(report["received"]))
    if report["missing"]:
        return 1, report
    if skip_acceptance:
        return 0, report
    status, detail = acceptance_status(probe.get("target_commit", ""))
    report["notification_acceptance"] = detail
    if status != "ok":
        report["missing"] = []
        return 1, report
    return 0, report


def _query_is_partial(data: dict) -> bool:
    status = data.get("status")
    return bool(isinstance(status, dict) and status.get("isPartial"))


def _units_in_response(data: dict) -> list[str]:
    units: list[str] = []
    for table in data.get("tables", []):
        columns = [c.get("name") for c in table.get("columns", [])]
        unit_idx = next(
            (
                i
                for i, name in enumerate(columns)
                if name in ("_SYSTEMD_UNIT", "systemd.unit", "_systemd_unit")
            ),
            None,
        )
        if unit_idx is None:
            continue
        for row in table.get("rows", []):
            if unit_idx < len(row) and row[unit_idx]:
                units.append(str(row[unit_idx]))
    return units


def git_blob(ref: str, rel: str) -> bytes | None:
    result = git("cat-file", "-p", f"{ref}:{rel}")
    if result.returncode != 0:
        return None
    return result.stdout.encode()


def _drill_unit_files(unit: str) -> list[str]:
    """The deploy/deploy-failed unit plus any managed drop-ins for it."""
    files = []
    result = git("ls-tree", "-r", "--name-only", "HEAD", "ops/systemd", "ops/gpu")
    names = [
        line
        for line in (result.stdout.splitlines() if result.returncode == 0 else [])
        if line
    ]
    for name in names:
        if name == f"ops/systemd/{unit}" or name.startswith(f"ops/systemd/{unit}.d/"):
            files.append(name)
    if (
        unit == "pareton-deploy.service"
        and "ops/systemd/pareton-deploy.service" not in files
    ):
        files.append(f"ops/systemd/{unit}")
    return files


def acceptance_status(target_commit: str) -> tuple[str, dict]:
    """Evaluate the spec 7.4 drill-evidence validity for this target."""
    record = read_json(acceptance_path())
    if not isinstance(record, dict) or not record.get("commit"):
        return "required", {"reason": "record-missing"}
    if record.get("host") != platform.node():
        return "required", {"reason": "host-changed"}
    detail = {"acceptance_commit": record["commit"]}
    if not target_commit:
        return "required", {**detail, "reason": "target-unknown"}
    for rel in NOTIFICATION_CHAIN_FILES:
        if git_blob(record["commit"], rel) != git_blob(target_commit, rel):
            return "required", {**detail, "reason": f"changed:{rel}"}
    for unit in ("pareton-deploy.service", "pareton-deploy-failed.service"):
        for rel in _drill_unit_files(unit):
            if rel in NOTIFICATION_CHAIN_FILES:
                continue
            if git_blob(record["commit"], rel) != git_blob(target_commit, rel):
                return "required", {**detail, "reason": f"changed:{rel}"}
    vector_result = _vector_toml_drift(record["commit"], target_commit)
    if vector_result is not None:
        return "required", {**detail, "reason": vector_result}
    return "ok", detail


def _vector_toml_drift(acceptance_commit: str, target_commit: str) -> str | None:
    """None when the TOML difference stays inside the single 7.4 exemption."""
    import tomllib

    def parsed(ref: str) -> dict | None:
        blob = git_blob(ref, "ops/vector/vector.toml")
        if blob is None:
            return None
        try:
            return tomllib.loads(blob.decode())
        except ValueError:
            return None

    accepted = parsed(acceptance_commit)
    target = parsed(target_commit)
    if accepted is None or target is None:
        return "vector-toml-unparseable"
    units_key = ("sources", "journald", "include_units")

    def without_units(doc: dict) -> str:
        copy = json.loads(json.dumps(doc, default=str))
        node = copy
        for key in units_key[:-1]:
            node = node.get(key, {})
        node.pop(units_key[-1], None)
        return json.dumps(copy, sort_keys=True, default=str)

    if without_units(accepted) != without_units(target):
        return "vector-toml-changed"
    target_units = set(
        target.get("sources", {}).get("journald", {}).get("include_units", [])
    )
    normalized = {u if u.endswith(".service") else f"{u}.service" for u in target_units}
    if "pareton-deploy-failed.service" not in normalized:
        return "exempt-unit-removed"
    return None


# ---------------------------------------------------------------------------
# Recovery copy (spec 6.2)


def venv_entry_ok(venv: Path) -> bool:
    if not (venv / "pyvenv.cfg").is_file():
        return False
    if not (venv / "bin" / "python").exists():
        return False
    return any(venv.glob("lib/python*/site-packages"))


def save_recovery_copy(state: dict) -> str:
    """Copy the quiesced venv aside; only promoted when complete (spec 6.2)."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    parent = state_dir() / "recovery"
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".{stamp}.tmp"
    final = parent / stamp
    if tmp.exists() or final.exists():
        raise Fail(2, "recovery-copy-collision", stamp=stamp)
    try:
        shutil.copytree(repo() / ".venv", tmp, symlinks=True)
        if not venv_entry_ok(tmp):
            raise Fail(2, "recovery-copy-incomplete")
        meta = {
            "op_id": state["op_id"],
            "from_commit": state["from_commit"],
            "target_commit": state["target_commit"],
            "created_at": now_iso(),
        }
        write_json_atomic(
            tmp / "meta.json" if False else tmp / "recovery-meta.json", meta
        )
        os.rename(tmp, final)
    except Fail:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    except (OSError, shutil.Error) as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise Fail(2, "recovery-copy-failed", detail=type(exc).__name__) from exc
    return str(final)


def prune_recovery_copies(keep: int = 2) -> None:
    parent = state_dir() / "recovery"
    if not parent.is_dir():
        return
    dirs = sorted(
        d for d in parent.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    for old in dirs[:-keep] if len(dirs) > keep else []:
        shutil.rmtree(old, ignore_errors=True)


def restore_recovery_venv(copy_dir: Path) -> None:
    """Put the saved venv back at its original absolute path (spec 6.2)."""
    source = copy_dir / "venv" if (copy_dir / "venv").is_dir() else copy_dir
    if not venv_entry_ok(source):
        raise Fail(2, "recovery-copy-unusable", copy_dir=str(copy_dir))
    target = repo() / ".venv"
    stage = repo() / f".venv.restore.{os.getpid()}"
    shutil.copytree(source, stage, symlinks=True)
    displaced = repo() / f".venv.displaced.{os.getpid()}"
    if not target.exists():
        # A previous swap died between the two renames: prefer the leftover
        # displaced copy as the thing to keep, and place the stage directly.
        leftovers = sorted(repo().glob(".venv.displaced.*"))
        for leftover in leftovers:
            try:
                os.rename(leftover, displaced)
                break
            except OSError:
                continue
        try:
            os.rename(stage, target)
        except OSError as exc:
            raise Fail(2, "venv-restore-failed", detail=type(exc).__name__) from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return
    os.rename(target, displaced)
    try:
        os.rename(stage, target)
    except OSError as exc:
        # Put the original back before raising: deleting the displaced copy
        # here would destroy the only good venv (CR P3).
        os.rename(displaced, target)
        shutil.rmtree(stage, ignore_errors=True)
        raise Fail(2, "venv-restore-failed", detail=type(exc).__name__) from exc
    shutil.rmtree(displaced, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tick state machine (spec 4.2)


def tick(argv: list[str]) -> int:
    global _DEPLOY_LOCK_FD
    parser = argparse.ArgumentParser(prog="release.py tick")
    parser.add_argument("--continue-op", default=None)
    args = parser.parse_args(argv)
    global _TICK_STARTED_AT
    _TICK_STARTED_AT = now_iso()
    if args.continue_op:
        return tick_continue(args.continue_op)

    lock = DeployLock.acquire()
    if lock is None:
        return 0  # peer deploy holds the mutex; silent per spec 4.1
    _DEPLOY_LOCK_FD = lock.fd
    try:
        return tick_locked()
    finally:
        lock.release()


def tick_locked() -> int:
    state = load_state()
    request = read_json(request_path())
    active = (
        request
        if isinstance(request, dict) and request.get("status") in ("pending", "running")
        else None
    )

    if state is None:
        if active and active.get("type") == "reset":
            return execute_reset(active)
        record_step("state-corrupt")
        print(
            "release: state corrupt or missing; bootstrap via request reset",
            file=sys.stderr,
        )
        return 2

    if active:
        if active.get("status") == "pending":
            return execute_request(state, active)
        kind = active["type"]
        # Running requests: cancel re-enters its stop-wait loop; verify
        # re-runs idempotently from idle; drain-first recovery (rollback/
        # reset/resume) continues through the normal phase dispatch below
        # so a busy worker cannot strand the operation (CR P1-4).
        if kind == "cancel" or (kind == "verify" and state["phase"] == "idle"):
            return execute_request(state, active)

    if state.get("hold") and active is None:
        return tick_held(state)

    phase = state["phase"]
    if phase == "idle":
        return tick_idle(state)
    if phase == "draining":
        return tick_draining(state)
    if phase == "quiescing":
        return tick_quiescing(state)
    if phase == "applying":
        # applying is only valid inside a single coordinated call that ends
        # with the re-exec; reaching here means that call died (spec 4.2).
        record_step("applying-interrupted")
        print(
            "release: applying was interrupted; use request resume or rollback",
            file=sys.stderr,
        )
        return 2
    if phase == "verifying":
        return tick_verifying_resume(state)
    record_step(f"unexpected-phase-{phase}")
    return 2


def tick_held(state: dict) -> int:
    # Read-only checks only: fetch is allowed for reporting, never pull/apply
    # (spec 4.1). Config check must not report errors under hold either way.
    try:
        git_out("fetch", "--quiet", "origin", "main")
    except Fail:
        record_step("held-fetch-error")
        return 2
    result = run_cmd(
        [
            sys.executable,
            str(repo() / "ops" / "sync-config.py"),
            "check",
            "--repo",
            str(repo()),
            "--source",
            "worktree",
        ],
        timeout=300,
    )
    if result.returncode not in (0, 1):
        record_step("held-check-error", {"detail": f"sync-exit-{result.returncode}"})
        return 2
    record_step("held")
    print(f"release: held (baseline {state['verified_commit'][:12]})")
    return 0


def sync_check_exit() -> int:
    result = run_cmd(
        [
            sys.executable,
            str(repo() / "ops" / "sync-config.py"),
            "check",
            "--repo",
            str(repo()),
            "--source",
            "worktree",
        ],
        timeout=300,
    )
    return result.returncode


def tick_idle(state: dict) -> int:
    # Alias bookkeeping: state is the authority, the file is a compat alias.
    alias = alias_path()
    try:
        current_alias = alias.read_text().strip()
    except OSError:
        current_alias = ""
    if state["verified_commit"] and current_alias != state["verified_commit"]:
        alias.write_text(state["verified_commit"] + "\n")
        print("release: .deploy-done alias rewritten from state")

    try:
        git_out("fetch", "--quiet", "origin", "main")
        target = git_out("rev-parse", "origin/main")
        head = git_out("rev-parse", "HEAD")
    except Fail as failure:
        record_step("git-error")
        print(f"release: {failure.reason}", file=sys.stderr)
        return failure.code

    drift_exit = sync_check_exit()
    if drift_exit not in (0, 1):
        record_step("config-check-error", {"detail": f"sync-exit-{drift_exit}"})
        print(f"release: config check failed (exit {drift_exit})", file=sys.stderr)
        return 2

    change_needed = target != state["verified_commit"] or drift_exit == 1
    if not change_needed:
        record_step("done")
        return 0

    # Classify the change set (spec 4.3): the only fast path is a complete
    # change set touching nothing but ops/vector/vector.toml with no owed
    # restart debt.
    changed_files = _changed_files(state["verified_commit"], target)
    vector_only = (
        all(f == "ops/vector/vector.toml" for f in changed_files)
        and drift_exit == 1
        and _drift_only_vector()
    )
    if vector_only:
        return vector_fast_path(state, target)

    op_id = str(uuid.uuid4())
    mutate_state(
        lambda s: s.update(
            {
                "op_id": op_id,
                "phase": "draining",
                "scope": "full",
                "direction": "forward",
                "from_commit": head,
                "target_commit": target,
                "startup_complete": False,
                "log_accepted": False,
                "recovery_copy": None,
                "failure_step": None,
                "phase_since": now_iso(),
                "original_units": snapshot_units(),
            }
        )
    )
    emit_event("maintenance_started", op_id=op_id, target_commit=target)
    return tick_draining(mutate_state(lambda s: None))


def _changed_files(from_commit: str, target: str) -> list[str]:
    """Full-tree diff: release-scope decisions must see business commits too.

    An ops/-only listing made the vector-only classification vacuously true
    for business commits (CR P1-2) and hid requirements.txt changes from the
    pip decision (CR P1-1).
    """
    if from_commit == target:
        return []
    result = git("diff", "--name-only", from_commit, target)
    if result.returncode != 0:
        return ["<diff-error>"]
    return [line for line in result.stdout.splitlines() if line]


def _requirements_changed(changed: list[str]) -> bool:
    return any(f in ("requirements.txt", "api/requirements.txt") for f in changed)


def _drift_only_vector() -> bool:
    result = run_cmd(
        [
            sys.executable,
            str(repo() / "ops" / "sync-config.py"),
            "check",
            "--repo",
            str(repo()),
            "--source",
            "worktree",
        ],
        timeout=300,
    )
    try:
        payload = json.loads(result.stdout)
        findings = payload.get("findings", [])
    except ValueError:
        return False
    return all(f.get("target") == "/etc/vector/vector.toml" for f in findings)


def vector_fast_path(state: dict, target: str) -> int:
    op_id = str(uuid.uuid4())
    mutate_state(
        lambda s: s.update(
            {
                "op_id": op_id,
                "phase": "applying",
                "scope": "vector-only",
                "direction": "forward",
                "from_commit": s["verified_commit"],
                "target_commit": target,
                "startup_complete": True,
                "log_accepted": False,
                "failure_step": None,
                "phase_since": now_iso(),
                "original_units": snapshot_units(),
            }
        )
    )
    try:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        if dirty.stdout.strip():
            raise Fail(2, "worktree-dirty")
        git_out("reset", "--hard", target)
        result = run_cmd(
            [
                sys.executable,
                str(repo() / "ops" / "sync-config.py"),
                "apply",
                "--install-only",
                "--repo",
                str(repo()),
                "--source",
                "worktree",
            ],
            timeout=600,
        )
        if result.returncode != 0:
            raise Fail(
                result.returncode or 2,
                "vector-install-failed",
                detail=result.stdout.strip()[:400],
            )
        probe = write_probe(str(uuid.uuid4()), target)
        gpu_probe_flow(op_id, probe)
        emit_deploy_probe(probe)
        code, report = run_log_check(probe)
    except Fail as failure:
        reason, code = failure.reason, failure.code
        mutate_state(
            lambda s: s.update(
                {"phase": "idle", "scope": "full", "failure_step": reason}
            )
        )
        _restore_maint_timers(load_state())
        record_step(reason)
        print(f"release: {reason}", file=sys.stderr)
        clear_coordination_files()
        return code or 2
    _restore_maint_timers(load_state())
    if code != 0:
        mutate_state(
            lambda s: s.update(
                {
                    "phase": "idle",
                    "scope": "full",
                    "log_accepted": False,
                    "failure_step": "log-ingestion" if code == 1 else "axiom-query",
                }
            )
        )
        record_step(
            "log-ingestion" if code == 1 else "axiom-query",
            {"detail": json.dumps(report.get("missing", []))},
        )
        print(f"release: log check failed: {json.dumps(report)[:400]}", file=sys.stderr)
        clear_coordination_files()
        return 1 if code == 1 else 2
    clear_coordination_files()
    mutate_state(
        lambda s: s.update(
            {
                "phase": "idle",
                "scope": "full",
                "log_accepted": True,
                "verified_commit": target,
                "failure_step": None,
            }
        )
    )
    alias_path().write_text(target + "\n")
    record_step("done")
    print(f"release: vector-only {state['verified_commit'][:12]} -> {target[:12]}")
    return 0


def tick_draining(state: dict) -> int:
    activity = ActivityLock.try_acquire()
    if activity is None:
        # Workers hold the shared lock: report live work, never force.
        records = db_running_records()
        detail = {
            "active-work": True,
            "rounds": records.get("rounds", []) if not records.get("error") else None,
            "submissions": (
                records.get("submissions", []) if not records.get("error") else None
            ),
        }
        step = (
            "active-work" if not records.get("error") else "active-work-db-unavailable"
        )
        since = parse_iso(state.get("phase_since") or state.get("updated_at") or "")
        waited = (parse_iso(now_iso()) - since).total_seconds() if since else 0
        mutate_state(lambda s: s.update({"failure_step": None}))
        record_step(
            step,
            {
                "detail": json.dumps(detail, default=str)[:600],
                "draining_since": state.get("phase_since", ""),
            },
        )
        if waited > drain_wait_s():
            record_step("drain-wait")
            print(
                f"release: drain wait exceeded {drain_wait_s()}s; still waiting",
                file=sys.stderr,
            )
            return 1
        return 0

    records = db_running_records()
    if records.get("error"):
        record_step("db-probe-error", {"detail": records["error"]})
        print(f"release: db probe failed ({records['error']})", file=sys.stderr)
        return 2
    if records.get("rounds"):
        record_step(
            "round-record-unresolved", {"detail": json.dumps(records["rounds"])[:400]}
        )
        print(
            f"release: running rounds with idle activity lock: "
            f"{[r['id'] for r in records['rounds']]}; see recover tooling",
            file=sys.stderr,
        )
        return 2
    if records.get("submissions"):
        record_step(
            "submission-record-unresolved",
            {"detail": json.dumps(records["submissions"])[:400]},
        )
        print(
            f"release: running submissions with idle activity lock: "
            f"{[r['id'] for r in records['submissions']]}; "
            f"run ops/recover_submission.py",
            file=sys.stderr,
        )
        return 2
    mutate_state(lambda s: s.update({"phase": "quiescing", "phase_since": now_iso()}))
    _stow_activity_fd(activity)
    return tick_quiescing(mutate_state(lambda s: None))
    # Every return above releases the exclusive lock (the process exits and
    # the fd closes); only the quiescing handoff stows it for applying.


_ACTIVITY_STASH: ActivityLock | None = None
_DEPLOY_LOCK_FD: int | None = None


def _stow_activity_fd(activity: ActivityLock) -> None:
    global _ACTIVITY_STASH
    _ACTIVITY_STASH = activity


def _ensure_activity_exclusive() -> ActivityLock | None:
    """Activity exclusive from this process, or a fresh acquire attempt.

    Re-acquiring in the same process through a new fd would self-conflict,
    so a stashed lock is reused (spec 4.5 re-exec contract).
    """
    if _ACTIVITY_STASH is not None:
        return _ACTIVITY_STASH
    return ActivityLock.try_acquire()


def tick_quiescing(state: dict) -> int:
    activity = _ensure_activity_exclusive()
    if activity is None:
        # A late in-flight task approved before the gate closed still holds
        # the shared lock; wait for it instead of stopping services (5.1).
        record_step("active-work", {"detail": "quiescing blocked by shared lock"})
        return 0
    _stow_activity_fd(activity)
    for timer in MAINT_TIMERS:
        if unit_is_active(timer):
            stop_unit(timer)
    pending = _wait_units_inactive(ONESHOT_UNITS, budget_s=60, step="oneshot-wait")
    if pending:
        record_step("service-stopping", {"detail": ",".join(pending)})
        return _quiesce_wait_exit(state, ONESHOT_UNITS, pending)

    for unit in WORKER_UNITS:
        if unit_is_active(unit):
            stop_unit(f"{unit}.service")
    pending = _wait_units_inactive(
        [f"{u}.service" for u in WORKER_UNITS], budget_s=60, step="worker-stop"
    )
    if pending:
        record_step("service-stopping", {"detail": ",".join(pending)})
        return _quiesce_wait_exit(
            state, [f"{u}.service" for u in WORKER_UNITS], pending
        )

    for unit in RESIDENT_UNITS:
        if unit_is_active(unit):
            stop_unit(f"{unit}.service")
    pending = _wait_units_inactive(
        [f"{u}.service" for u in RESIDENT_UNITS], budget_s=60, step="resident-stop"
    )
    if pending:
        record_step("service-stopping", {"detail": ",".join(pending)})
        return _quiesce_wait_exit(
            state, [f"{u}.service" for u in RESIDENT_UNITS], pending
        )

    mutate_state(lambda s: s.update({"phase": "applying", "phase_since": now_iso()}))
    return tick_applying(mutate_state(lambda s: None))


def _quiesce_wait_exit(state: dict, all_units: list[str], pending: list[str]) -> int:
    """Still quiescing: exit 0 within thresholds, alert beyond (spec 5.2)."""
    since = parse_iso(state.get("phase_since") or state.get("updated_at") or "")
    waited = (parse_iso(now_iso()) - since).total_seconds() if since else 0
    for unit in pending:
        base = unit.removesuffix(".service")
        if waited > STOP_ALERT_S.get(base, 1800):
            record_step("service-stopping", {"detail": unit})
            print(f"release: {unit} stopping for {waited:.0f}s", file=sys.stderr)
            return 1
    return 0


def tick_applying(state: dict) -> int:
    op_id = state["op_id"]
    direction = state.get("direction", "forward")

    if state.get("recovery_copy") and direction in ("rollback", "reset"):
        restore_recovery_venv(Path(state["recovery_copy"]))
    elif direction == "forward" and state.get("recovery_copy") is None:
        copy_dir = save_recovery_copy(state)
        mutate_state(lambda s: s.update({"recovery_copy": copy_dir}))
    # reset without a copy: the operator's evidence vouches for the env.

    status = git("status", "--porcelain", "--untracked-files=no")
    if status.stdout.strip():
        record_step("worktree-dirty")
        print("release: tracked worktree modifications present", file=sys.stderr)
        return 2
    try:
        git_out("reset", "--hard", state["target_commit"])
    except Fail as failure:
        record_step(failure.reason)
        return failure.code

    # Rollback never re-resolves dependencies: the restored venv copy IS
    # the environment (spec 6.2; un-pinned requirements make pip a mutation).
    changed = _changed_files(state["from_commit"], state["target_commit"])
    if _requirements_changed(changed) and direction == "forward":
        result = run_cmd(
            [
                str(venv_python().parent / "pip"),
                "install",
                "--quiet",
                "-r",
                str(repo() / "requirements.txt"),
            ],
            timeout=3600,
        )
        if result.returncode != 0:
            record_step("deps-failed")
            print("release: pip install failed", file=sys.stderr)
            return 2

    result = run_cmd(
        [
            sys.executable,
            str(repo() / "ops" / "sync-config.py"),
            "apply",
            "--install-only",
            "--repo",
            str(repo()),
            "--source",
            "worktree",
        ],
        timeout=600,
    )
    if result.returncode != 0:
        record_step("install-failed", {"detail": result.stdout.strip()[:400]})
        print(
            f"release: sync-config apply failed ({result.returncode})", file=sys.stderr
        )
        return 2

    # Hand over to the freshly installed entrypoint; the deploy mutex and
    # activity lock fds travel via the environment (spec 6.2-3).
    entry = ops_dir() / "release.py"
    env = {
        **os.environ,
        "PARETON_INHERIT_DEPLOY_LOCK_FD": str(_DEPLOY_LOCK_FD),
        "PARETON_INHERIT_ACTIVITY_LOCK_FD": str(_ACTIVITY_STASH.fd)
        if _ACTIVITY_STASH
        else "",
    }
    record_step("reexec")
    os.execve(
        str(sys.executable),
        [str(sys.executable), str(entry), "tick", "--continue-op", op_id],
        env,
    )
    return 0  # unreachable


# ---------------------------------------------------------------------------
# Verification (spec 4.2 verifying / 7)


def tick_continue(op_id: str) -> int:
    global _DEPLOY_LOCK_FD, _TICK_STARTED_AT
    _TICK_STARTED_AT = _TICK_STARTED_AT or now_iso()
    try:
        deploy_fd = int(os.environ.get("PARETON_INHERIT_DEPLOY_LOCK_FD", "0"))
        activity_fd = int(os.environ.get("PARETON_INHERIT_ACTIVITY_LOCK_FD", "0"))
    except ValueError:
        record_step("reexec-fd-invalid")
        return 2
    if not deploy_fd:
        record_step("reexec-fd-missing")
        return 2
    lock = DeployLock.adopt(deploy_fd)
    _DEPLOY_LOCK_FD = deploy_fd
    try:
        if activity_fd:
            _stow_activity_fd(ActivityLock.adopt(activity_fd))
        state = load_state()
        if state is None or state["op_id"] != op_id:
            record_step("reexec-op-mismatch")
            return 2
        if state["phase"] != "applying":
            record_step("reexec-phase-mismatch")
            return 2
        mutate_state(
            lambda s: s.update(
                {
                    "phase": "verifying",
                    "startup_complete": False,
                    "log_accepted": False,
                    "phase_since": now_iso(),
                }
            )
        )
        return verify_flow(mutate_state(lambda s: None), fresh_start=True)
    finally:
        lock.release()


def tick_verifying_resume(state: dict) -> int:
    """A tick found phase=verifying with startup complete."""
    if state.get("startup_complete") is not True:
        record_step("verifying-partial-startup")
        print(
            "release: target partially started; use request resume or rollback",
            file=sys.stderr,
        )
        return 2
    if state.get("failure_step") in (
        "log-ingestion",
        "axiom-query",
        "notification-acceptance-required",
    ):
        # A finished-but-failed verification is the "running, unaccepted"
        # steady state (spec 4.2): business continues, automatic re-verify
        # would re-run the GPU probe and Axiom loop every tick; wait for the
        # explicit verify / rollback / vector-repair instead.
        record_step("log-unaccepted", {"detail": state["failure_step"]})
        return 0
    # The verification was interrupted mid-flight; continue it.
    return verify_flow(load_state(), fresh_start=False)


def api_healthy() -> bool:
    base = os.environ.get("PARETON_API_BASE", "http://127.0.0.1:8000")
    for path in ("/health", "/v1/campaigns"):
        try:
            with urllib.request.urlopen(f"{base}{path}", timeout=5) as response:
                if response.status != 200:
                    return False
        except (urllib.error.URLError, OSError, ValueError):
            return False
    return True


def _wait_unit_healthy(unit: str, budget_s: int) -> bool:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        if unit == "pareton-api":
            if api_healthy():
                return True
        elif unit_is_active(unit):
            return True
        time.sleep(2)
    return api_healthy() if unit == "pareton-api" else unit_is_active(unit)


def verify_flow(state: dict, *, fresh_start: bool) -> int:
    global _ACTIVITY_STASH
    target = state["target_commit"]
    if fresh_start:
        probe = write_probe(str(uuid.uuid4()), target)
        for unit in RESIDENT_UNITS:
            start_unit(f"{unit}.service")
        for unit in RESIDENT_UNITS:
            if not _wait_unit_healthy(unit, API_HEALTH_TIMEOUT_S):
                record_step("start-failed", {"detail": unit})
                print(f"release: {unit} failed to start/health", file=sys.stderr)
                return 2
        mutate_state(lambda s: s.update({"startup_complete": True}))
        # The activity lock guards claims; workers may start once released.
        if _ACTIVITY_STASH is not None:
            _ACTIVITY_STASH.release()
            _ACTIVITY_STASH = None
        for unit in WORKER_UNITS:
            start_unit(f"{unit}.service")
        for unit in WORKER_UNITS:
            if not _wait_unit_healthy(unit, API_HEALTH_TIMEOUT_S):
                record_step("start-failed", {"detail": unit})
                print(f"release: {unit} failed to start", file=sys.stderr)
                return 2
    else:
        probe = write_probe(str(uuid.uuid4()), target)

    gpu_probe_flow(state["op_id"], probe)
    emit_deploy_probe(probe)
    code, report = run_log_check(probe)
    return finish_verification(load_state(), code, report)


def finish_verification(state: dict, code: int, report: dict) -> int:
    target = state["target_commit"]
    if code == 0:
        mutate_state(
            lambda s: s.update(
                {
                    "phase": "idle",
                    "scope": "full",
                    "log_accepted": True,
                    "verified_commit": target,
                    "failure_step": None,
                    "last_release": {
                        "op_id": state["op_id"],
                        "target": target,
                        "completed_at": now_iso(),
                    },
                }
            )
        )
        alias_path().write_text(target + "\n")
        run_cmd(
            [
                sys.executable,
                str(repo() / "ops" / "sync-config.py"),
                "effectuate-restarts",
                "--repo",
                str(repo()),
            ],
            timeout=300,
        )
        _restore_maint_timers(state)
        prune_recovery_copies()
        _notify_success(state)
        clear_coordination_files()
        _finish_request("done", {"target": target})
        emit_event("maintenance_finished", op_id=state["op_id"], target_commit=target)
        record_step("done")
        print(f"release: verified {target[:12]}")
        return 0

    step = {
        1: "notification-acceptance-required"
        if report.get("notification_acceptance", {}).get("reason")
        else "log-ingestion",
        2: "axiom-query",
    }.get(code, "log-ingestion")
    mutate_state(
        lambda s: s.update(
            {
                "log_accepted": False,
                "failure_step": step,
                "phase": "verifying",
                "startup_complete": True,
            }
        )
    )
    record_step(step, {"detail": json.dumps(report)[:600]})
    print(
        f"release: log verification failed: {json.dumps(report)[:400]}", file=sys.stderr
    )
    _restore_maint_timers(state)
    clear_coordination_files()
    _finish_request("failed", {"step": step})
    return 1 if code == 1 else 2


def _restore_maint_timers(state: dict | None = None) -> None:
    snapshot = (state or {}).get("original_units") or {}
    for timer in MAINT_TIMERS:
        if snapshot.get(timer, {}).get("active", True):
            start_unit(timer)
    if snapshot and not (state or {}).get("hold"):
        if snapshot.get(DEPLOY_TIMER, {}).get("active", True):
            start_unit(DEPLOY_TIMER)


def _notify_success(state: dict) -> None:
    notifier = ops_dir() / "notify-deploy-failure.py"
    run_cmd(
        [
            sys.executable,
            str(notifier),
            "record-success",
            "--invocation",
            os.environ.get("INVOCATION_ID", "manual"),
            "--from",
            state.get("from_commit", "unknown"),
            "--to",
            state["target_commit"],
        ],
        timeout=60,
    )


def _finish_request(status: str, result: dict) -> None:
    request = read_json(request_path())
    if not isinstance(request, dict) or request.get("status") not in (
        "pending",
        "running",
    ):
        return
    request["status"] = status
    request["result"] = result
    request["finished_at"] = now_iso()
    with state_lock():
        write_json_atomic(request_path(), request)


# ---------------------------------------------------------------------------
# Requests (spec 6.3)


REQUEST_TYPES = (
    "hold",
    "unpause",
    "reset",
    "resume",
    "cancel",
    "verify",
    "rollback",
    "vector-repair",
)


def cmd_request(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="release.py request")
    parser.add_argument("type", choices=REQUEST_TYPES)
    parser.add_argument("--reason", default=None)
    parser.add_argument("--operator", default=None)
    parser.add_argument("--target", default=None, help="vector-repair target commit")
    parser.add_argument("--baseline-commit", default=None, help="reset evidence")
    parser.add_argument("--confirm-evidence", default=None, help="reset evidence")
    parser.add_argument("--recovery-copy", default=None, help="reset venv source")
    parser.add_argument("--main-commit", default=None, help="unpause checked commit")
    args = parser.parse_args(argv)

    if args.type == "reset":
        if not (args.baseline_commit and args.confirm_evidence and args.operator):
            print(
                "request reset: --baseline-commit, --confirm-evidence and "
                "--operator are all required (spec 6.3)",
                file=sys.stderr,
            )
            return 2
    if args.type == "vector-repair" and not args.target:
        print("request vector-repair: --target COMMIT required", file=sys.stderr)
        return 2
    if args.type == "unpause" and not (args.main_commit and args.operator):
        print(
            "request unpause: --main-commit and --operator required (spec 6.3)",
            file=sys.stderr,
        )
        return 2
    if not args.operator:
        args.operator = os.environ.get("USER", "unknown")

    if args.type == "hold":
        # Registration takes effect immediately under the short state lock;
        # a running install is never killed and finishes first (spec 6.3).
        state = load_state()
        if state is None:
            print("request hold: state corrupt; use request reset", file=sys.stderr)
            return 2

        def set_hold(s: dict) -> None:
            s["hold"] = {
                "reason": args.reason or "manual",
                "operator": args.operator,
                "at": now_iso(),
                "baseline_commit": s.get("verified_commit"),
            }

        mutate_state(set_hold)
        result = run_cmd(["systemctl", "disable", "--now", DEPLOY_TIMER], timeout=60)
        note = "" if result.returncode == 0 else " (timer disable failed)"
        print(f"request hold: registered{note}")
        return 0

    # All other requests are registered under the deploy mutex and executed
    # by pareton-deploy.service ticks (spec 6.3).
    lock = DeployLock.acquire()
    if lock is None:
        print("request: a deploy is running; retry when it finishes", file=sys.stderr)
        return 1
    try:
        existing = read_json(request_path())
        if isinstance(existing, dict) and existing.get("status") in (
            "pending",
            "running",
        ):
            print(
                f"request: {existing['type']} still {existing['status']}; "
                "one request at a time",
                file=sys.stderr,
            )
            return 1
        payload = {
            "type": args.type,
            "reason": args.reason,
            "operator": args.operator,
            "registered_at": now_iso(),
            "status": "pending",
            "target": args.target,
            "baseline_commit": args.baseline_commit,
            "confirm_evidence": args.confirm_evidence,
            "recovery_copy": args.recovery_copy,
            "main_commit": args.main_commit,
        }
        with state_lock():
            write_json_atomic(request_path(), payload)
        print(f"request {args.type}: registered; run pareton-deploy.service to execute")
        return 0
    finally:
        lock.release()


def execute_request(state: dict, request: dict) -> int:
    kind = request["type"]
    handler = {
        "unpause": _request_unpause,
        "resume": _request_resume,
        "cancel": _request_cancel,
        "verify": _request_verify,
        "rollback": _request_rollback,
        "vector-repair": _request_vector_repair,
        "reset": _execute_reset_request,
    }.get(kind)
    if handler is None:
        record_step(f"request-unknown-{kind}")
        return 2
    return handler(state, request)


def _refuse(request: dict, step: str) -> None:
    """Refused requests end as failed instead of retrying every tick
    and blocking new registrations (spec 6.3, CR P1-4)."""
    record_step(step)
    _finish_request("failed", {"step": step})


def _mark_request_running(request: dict) -> None:
    request["status"] = "running"
    with state_lock():
        current = read_json(request_path())
        if isinstance(current, dict) and current.get("type") == request["type"]:
            write_json_atomic(request_path(), request)


def _start_recovery_operation(
    state: dict, request: dict, direction: str, target: str
) -> int:
    """Common drain-first entry for rollback/resume/reset (spec 6.3)."""

    def begin(s: dict) -> None:
        s.update(
            {
                "phase": "draining",
                "scope": "full",
                "direction": direction,
                "target_commit": target,
                "startup_complete": False,
                "log_accepted": False,
                "phase_since": now_iso(),
                "failure_step": None,
            }
        )
        if s.get("original_units") is None:
            s["original_units"] = snapshot_units()

    mutate_state(begin)
    _mark_request_running(request)
    emit_event("maintenance_started", op_id=state["op_id"], target_commit=target)
    return tick_draining(mutate_state(lambda s: None))


def _request_rollback(state: dict, request: dict) -> int:
    if state.get("hold") is None:

        def set_hold(s: dict) -> None:
            s["hold"] = {
                "reason": request.get("reason") or "rollback",
                "operator": request.get("operator", "unknown"),
                "at": now_iso(),
                "baseline_commit": s.get("verified_commit"),
            }

        mutate_state(set_hold)
    run_cmd(["systemctl", "disable", "--now", DEPLOY_TIMER], timeout=60)
    target = state["verified_commit"]
    if not state.get("recovery_copy"):
        _refuse(request, "rollback-no-copy")
        print(
            "request rollback: no recovery copy for the current op; "
            "venv restores only from a saved copy",
            file=sys.stderr,
        )
        return 2
    return _start_recovery_operation(state, request, "rollback", target)


def _request_resume(state: dict, request: dict) -> int:
    if state["phase"] not in ("applying", "verifying"):
        _refuse(request, "resume-not-applicable")
        print(
            f"request resume: phase is {state['phase']}, not interrupted work",
            file=sys.stderr,
        )
        return 1
    return _start_recovery_operation(
        state, request, state.get("direction", "forward"), state["target_commit"]
    )


def _request_cancel(state: dict, request: dict) -> int:
    if state["phase"] not in ("draining", "quiescing"):
        _refuse(request, "cancel-not-applicable")
        print(
            f"request cancel: phase is {state['phase']}; only pre-write phases",
            file=sys.stderr,
        )
        return 1

    def set_hold(s: dict) -> None:
        s["hold"] = {
            "reason": request.get("reason") or "cancelled release",
            "operator": request.get("operator", "unknown"),
            "at": now_iso(),
            "baseline_commit": s.get("verified_commit"),
        }

    mutate_state(set_hold)
    run_cmd(["systemctl", "disable", "--now", DEPLOY_TIMER], timeout=60)
    if request.get("status") != "running":
        _mark_request_running(request)
    stopping = [f"{u}.service" for u in ALL_RESIDENT if unit_is_active(f"{u}.service")]
    if stopping:
        pending = _wait_units_inactive(stopping, budget_s=60, step="cancel-wait")
        if pending:
            record_step("cancel-wait", {"detail": ",".join(pending)})
            return 0  # next tick continues the cancel

    # Phase goes back to idle BEFORE any start: ExecCondition re-reads the
    # state file when systemd starts the unit, and quiescing denies startup
    # (CR P1-6). Snapshot keys are bare unit names, not FQNs.
    snapshot = state.get("original_units") or {}
    mutate_state(
        lambda s: s.update(
            {
                "phase": "idle",
                "scope": "full",
                "failure_step": None,
                "recovery_copy": None,
            }
        )
    )
    for unit in ALL_RESIDENT:
        if snapshot.get(unit, {}).get("active", True):
            start_unit(f"{unit}.service")
    for timer in MAINT_TIMERS:
        if snapshot.get(timer, {}).get("active", True):
            start_unit(timer)
    request["status"] = "done"
    request["result"] = {"cancelled_target": state["target_commit"]}
    request["finished_at"] = now_iso()
    with state_lock():
        write_json_atomic(request_path(), request)
    record_step("cancel-done")
    print("request cancel: environment unchanged, services restored, hold kept")
    return 0


def _request_verify(state: dict, request: dict) -> int:
    if state["phase"] not in ("verifying", "idle"):
        _refuse(request, "verify-not-applicable")
        print(f"request verify: phase is {state['phase']}", file=sys.stderr)
        return 1
    if state["phase"] == "verifying" and state.get("startup_complete") is not True:
        _refuse(request, "verify-partial-startup")
        return 1
    if state["phase"] == "idle":
        # The GPU dispatch only consumes one-shot probe requests while the
        # phase is verifying (spec 7.2); enter it before checking logs.
        mutate_state(
            lambda s: s.update(
                {
                    "phase": "verifying",
                    "startup_complete": True,
                    "phase_since": now_iso(),
                }
            )
        )
    _mark_request_running(request)
    return verify_flow(load_state(), fresh_start=False)


def _request_unpause(state: dict, request: dict) -> int:
    if state["phase"] != "idle":
        record_step("unpause-not-idle")
        print(f"request unpause: phase is {state['phase']}", file=sys.stderr)
        return 1
    try:
        git_out("fetch", "--quiet", "origin", "main")
        remote = git_out("rev-parse", "origin/main")
    except Fail as failure:
        record_step(failure.reason)
        return failure.code
    if request.get("main_commit") and remote != request["main_commit"]:
        _refuse(request, "unpause-main-moved")
        print(
            f"request unpause: origin/main moved to {remote[:12]}; "
            "re-register with the new commit",
            file=sys.stderr,
        )
        return 1
    mutate_state(lambda s: s.update({"hold": None}))
    snapshot = state.get("original_units") or {}
    if snapshot.get(DEPLOY_TIMER, {}).get("enabled", True):
        run_cmd(["systemctl", "enable", "--now", DEPLOY_TIMER], timeout=60)
    request["status"] = "done"
    request["result"] = {"unpaused_at": now_iso(), "main_commit": remote}
    request["finished_at"] = now_iso()
    with state_lock():
        write_json_atomic(request_path(), request)
    record_step("unpause-done")
    print("request unpause: hold cleared, automatic deploys resumed")
    return 0


def _request_vector_repair(state: dict, request: dict) -> int:
    repair_target = request.get("target")
    if state["phase"] != "verifying" or state.get("startup_complete") is not True:
        _refuse(request, "vector-repair-not-applicable")
        print(
            "request vector-repair: only for a healthy target with failed logs",
            file=sys.stderr,
        )
        return 1
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty.stdout.strip():
        _refuse(request, "vector-repair-worktree-dirty")
        print(
            "request vector-repair: tracked worktree modifications present",
            file=sys.stderr,
        )
        return 1
    head = git_out("rev-parse", "HEAD")
    if head != state["target_commit"]:
        _refuse(request, "vector-repair-head-moved")
        print(
            "request vector-repair: HEAD no longer the recorded target B",
            file=sys.stderr,
        )
        return 1
    changed = _changed_files(state["target_commit"], repair_target)
    if changed != ["ops/vector/vector.toml"]:
        _refuse(request, "vector-repair-scope")
        print(
            f"request vector-repair: B..C diff is {changed}, must be TOML-only",
            file=sys.stderr,
        )
        return 1
    drift_exit = sync_check_exit()
    if drift_exit == 1 and not _drift_only_vector():
        _refuse(request, "vector-repair-drift")
        print(
            "request vector-repair: unmanaged drift beyond vector.toml", file=sys.stderr
        )
        return 1

    def retarget(s: dict) -> None:
        s["target_commit"] = repair_target
        s["vector_repair_from"] = state["target_commit"]

    _mark_request_running(request)
    git_out("reset", "--hard", repair_target)
    result = run_cmd(
        [
            sys.executable,
            str(repo() / "ops" / "sync-config.py"),
            "apply",
            "--install-only",
            "--repo",
            str(repo()),
            "--source",
            "worktree",
        ],
        timeout=600,
    )
    if result.returncode != 0:
        mutate_state(retarget)
        record_step("vector-repair-install-failed")
        return 2
    mutate_state(retarget)
    probe = write_probe(str(uuid.uuid4()), repair_target)
    gpu_probe_flow(state["op_id"], probe)
    emit_deploy_probe(probe)
    code, report = run_log_check(probe)
    return finish_verification(load_state(), code, report)


def execute_reset(request: dict) -> int:
    """Rebuild a corrupt/missing state file from operator evidence (6.3)."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    if state_path().exists():
        os.rename(state_path(), state_dir() / f"release-state.corrupt.{stamp}")
    else:
        (state_dir() / f"release-state.corrupt.{stamp}.missing").write_text(
            json.dumps({"note": "state was missing at reset", "at": now_iso()}) + "\n"
        )
    baseline = request["baseline_commit"]
    op_id = str(uuid.uuid4())
    # verified_commit carries the operator-declared recovery anchor from the
    # start (the schema requires it, and a rollback during the reset must
    # have a target); "verified" in the acceptance sense is gated by
    # phase/log_accepted, which only a completed verify sets.
    fresh = {
        "schema_version": SCHEMA_VERSION,
        "op_id": op_id,
        "phase": "draining",
        "scope": "full",
        "direction": "reset",
        "from_commit": baseline,
        "target_commit": baseline,
        "verified_commit": baseline,
        "hold": {
            "reason": request.get("reason") or "state reset",
            "operator": request.get("operator", "unknown"),
            "at": now_iso(),
            "baseline_commit": baseline,
        },
        "original_units": snapshot_units(),
        "recovery_copy": request.get("recovery_copy"),
        "startup_complete": False,
        "log_accepted": False,
        "failure_step": None,
        "phase_since": now_iso(),
        "updated_at": now_iso(),
    }
    with state_lock():
        write_json_atomic(state_path(), fresh)
    run_cmd(["systemctl", "disable", "--now", DEPLOY_TIMER], timeout=60)
    request["status"] = "running"
    with state_lock():
        write_json_atomic(request_path(), request)
    emit_event("maintenance_started", op_id=op_id, target_commit=baseline)
    return tick_draining(fresh)


def _execute_reset_request(state: dict, request: dict) -> int:
    return execute_reset(request)


# ---------------------------------------------------------------------------
# ExecCondition / probe / dispatch entry points (spec 5.4, 7.2)


def cmd_gate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="release.py gate")
    parser.add_argument("--unit", required=True)
    args = parser.parse_args(argv)
    state = load_state()
    allowed, reason = exec_allow(state)
    if state is None:
        emit_event(
            "release_gate_error", unit=args.unit, reason="state-corrupt-or-missing"
        )
        print(json.dumps({"gate": "error", "unit": args.unit, "reason": reason}))
        return 255
    print(
        json.dumps(
            {
                "gate": "allow" if allowed else "deny",
                "unit": args.unit,
                "reason": reason,
            }
        )
    )
    return 0 if allowed else 1


def cmd_emit_probe(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="release.py emit-probe")
    parser.add_argument("--unit", required=True)
    args = parser.parse_args(argv)
    probe = read_json(probe_path())
    if isinstance(probe, dict) and probe.get("probe_id"):
        emit_event(
            "deployment_probe",
            probe_id=probe["probe_id"],
            unit=args.unit,
            target_commit=probe.get("target_commit"),
            host=probe.get("host"),
        )
    return 0


def cmd_gpu_reap_dispatch(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="release.py gpu-reap-dispatch")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.cmd
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("gpu-reap-dispatch: no command after --", file=sys.stderr)
        return 2

    request = read_json(gpu_request_path())
    consumed_probe = None
    if isinstance(request, dict) and not request.get("consumed"):
        state = load_state()
        issued = parse_iso(request.get("issued_at") or "")
        age = (parse_iso(now_iso()) - issued).total_seconds() if issued else -1
        deploy_invocation = _deploy_invocation_id()
        valid = (
            state is not None
            and (
                state["phase"] == "verifying"
                or (state["phase"] == "applying" and state["scope"] == "vector-only")
            )
            and request.get("op_id") == state["op_id"]
            and request.get("invocation_id") == deploy_invocation
            and 0 <= age <= 120
        )
        if valid:
            request["consumed"] = True
            with state_lock():
                current = read_json(gpu_request_path())
                if isinstance(current, dict) and not current.get("consumed"):
                    write_json_atomic(gpu_request_path(), request)
                    consumed_probe = request
    if consumed_probe is not None:
        emit_event(
            "gpu_reap_probe_warning",
            note="log-only probe run; real reap skipped by deploy request",
            probe_id=consumed_probe.get("probe_id"),
            op_id=consumed_probe.get("op_id"),
            unit="pareton-gpu-reap.service",
        )
        emit_event(
            "deployment_probe",
            probe_id=consumed_probe.get("probe_id"),
            unit="pareton-gpu-reap.service",
        )
        return 0
    os.execvp(command[0], command)
    return 2  # unreachable


def _deploy_invocation_id() -> str:
    result = run_cmd(
        ["systemctl", "show", DEPLOY_UNIT, "--property=InvocationID"], timeout=30
    )
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key == "InvocationID":
            return value.strip()
    return ""


def cmd_status(argv: list[str]) -> int:
    state = load_state()
    request = read_json(request_path())
    acceptance = read_json(acceptance_path())
    payload = {
        "state": state,
        "state_raw_present": state_path().exists(),
        "request": request,
        "notification_acceptance": acceptance,
        "probe_file": read_json(probe_path()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if state is None:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Notification acceptance drill record (spec 7.4)


def cmd_record_acceptance(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="release.py record-notification-acceptance")
    parser.add_argument("--invocation", required=True)
    parser.add_argument("--message-id", required=True)
    parser.add_argument("--confirmed-by", required=True)
    args = parser.parse_args(argv)

    try:
        head = git_out("rev-parse", "HEAD")
    except Fail:
        head = "unknown"
    _units, problem = installed_include_units()
    if problem:
        print(f"record-notification-acceptance: {problem}", file=sys.stderr)
        return 2

    token, token_problem = axiom_query_token()
    if token is None:
        print(f"record-notification-acceptance: {token_problem}", file=sys.stderr)
        return 2
    import tomllib

    dataset = tomllib.loads(p("/etc/vector/vector.toml").read_text())["sinks"]["axiom"][
        "dataset"
    ]
    base_url = os.environ.get("PARETON_AXIOM_API_URL", "https://api.axiom.co")
    apl = (
        f"['{dataset}'] | where invocation_id == '{args.invocation}' "
        f"| where ['_SYSTEMD_UNIT'] == 'pareton-deploy-failed.service'"
    )
    body = {
        "apl": apl,
        "startTime": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 6 * 3600)
        ),
        "endTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    status, data, error = http_post_json(
        f"{base_url}/v1/datasets/_apl?format=tabular", body, 30
    )
    if status != 200 or data is None:
        print(
            f"record-notification-acceptance: query failed ({error or status})",
            file=sys.stderr,
        )
        return 2
    events = [row for table in data.get("tables", []) for row in table.get("rows", [])]
    if not events:
        print(
            "record-notification-acceptance: no deploy-failed Axiom evidence for "
            "this invocation; refusing to record (spec 7.4)",
            file=sys.stderr,
        )
        return 1
    vector_version = ""
    result = run_cmd(["vector", "--version"], timeout=30)
    if result.returncode == 0:
        vector_version = result.stdout.strip()
    record = {
        "host": platform.node(),
        "commit": head,
        "vector_version": vector_version,
        "drilled_at": now_iso(),
        "failure_invocation": args.invocation,
        "discord_message_id": args.message_id,
        "confirmed_by": args.confirmed_by,
        "axiom_event_count": len(events),
    }
    with state_lock():
        write_json_atomic(acceptance_path(), record)
    print("record-notification-acceptance: drill evidence recorded")
    return 0


# ---------------------------------------------------------------------------
# CLI


def cmd_check_logs(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="release.py check-logs")
    parser.add_argument("--probe-id", default=None)
    parser.add_argument("--target", default=None)
    args = parser.parse_args(argv)
    probe = read_json(probe_path())
    if args.probe_id:
        probe = {
            "probe_id": args.probe_id,
            "target_commit": args.target,
            "host": platform.node(),
            "issued_at": now_iso(),
        }
    if not isinstance(probe, dict) or not probe.get("probe_id"):
        print("check-logs: no probe file and no --probe-id", file=sys.stderr)
        return 2
    if args.target:
        probe["target_commit"] = args.target
    code, report = run_log_check(probe)
    print(json.dumps(report, default=str))
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=(
            "tick",
            "gate",
            "emit-probe",
            "gpu-reap-dispatch",
            "status",
            "request",
            "record-notification-acceptance",
            "check-logs",
        ),
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    handlers = {
        "tick": tick,
        "gate": cmd_gate,
        "emit-probe": cmd_emit_probe,
        "gpu-reap-dispatch": cmd_gpu_reap_dispatch,
        "status": cmd_status,
        "request": cmd_request,
        "record-notification-acceptance": cmd_record_acceptance,
        "check-logs": cmd_check_logs,
    }
    try:
        return handlers[args.command](args.rest)
    except Fail as failure:
        record_step(failure.reason)
        print(json.dumps({"error": failure.reason, **failure.extra}), file=sys.stderr)
        return failure.code


if __name__ == "__main__":
    sys.exit(main())
