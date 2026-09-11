"""Offline tests for ops/release.py (stage-2 spec sections 4.5, 6.3, 7).

Everything runs against a temporary PARETON_RELEASE_BASE prefix with
git/systemctl/pip/venv subprocesses faked through release.run_cmd and the
Axiom HTTP layer faked through release.http_post_json. Real systemd and
Vector behavior is covered by the isolated acceptance matrix instead.
"""

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS = REPO_ROOT / "ops"


def load_ops_module(name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), OPS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load_ops_module("release")


class FakeRunner:
    """Programmable stand-in for release.run_cmd."""

    def __init__(self, base: Path):
        self.base = base
        self.calls: list[list[str]] = []
        self.git_refs = {"origin/main": "c2", "HEAD": "c1"}
        self.git_diff = ""
        self.db = {"error": None, "rounds": [], "submissions": []}
        self.active_units: set[str] = set()
        self.sync_exit = 0
        self.pip_exit = 0
        self.deploy_invocation = "inv-1"

    def __call__(self, argv, env=None, timeout=300, cwd=None):
        self.calls.append(list(argv))
        cmd = argv[0]
        out, rc = "", 0

        if cmd == "git":
            # argv: ["git", "-C", <repo>, verb, *args]
            rest = argv[3:] if argv[1] == "-C" else argv[1:]
            verb = rest[0] if rest else ""
            args = rest[1:]
            if verb == "fetch":
                rc = 0
            elif verb == "rev-parse":
                ref = args[0] if args else ""
                out = self.git_refs.get(ref, "")
                rc = 0 if out else 128
            elif verb == "status":
                out, rc = "", 0
            elif verb == "diff":
                out = self.git_diff
            elif verb in ("reset", "cat-file", "ls-tree"):
                rc = 0
            else:
                rc = 0
        elif cmd == "systemctl":
            if "is-active" in argv:
                unit = argv[-1]
                out = "active" if unit in self.active_units else "inactive"
            elif "is-enabled" in argv:
                out = "enabled"
            elif "show" in argv and any(
                "InvocationID" in part for part in argv
            ):
                out = f"InvocationID={self.deploy_invocation}"
            rc = 0
        elif cmd.endswith("python"):
            if "-c" in argv:
                out = json.dumps(self.db)
        elif cmd.endswith("pip"):
            rc = self.pip_exit
        elif cmd.endswith("sync-config.py"):
            rc = self.sync_exit
        else:
            rc = 0

        class Result:
            pass

        result = Result()
        result.returncode = rc
        result.stdout = out
        result.stderr = ""
        return result


def make_mini_venv(base: Path) -> None:
    venv = base / "opt/pareton/.venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "bin/python").write_text("#!/bin/sh\n")
    (venv / "lib/python3.12/site-packages").mkdir(parents=True)


@pytest.fixture()
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("PARETON_RELEASE_BASE", str(tmp_path))
    monkeypatch.setenv(
        "PARETON_OPS_DIR", str(tmp_path / "usr/local/lib/pareton-ops")
    )
    (tmp_path / "opt/pareton").mkdir(parents=True)
    (tmp_path / "var/lib/pareton-deploy").mkdir(parents=True)
    (tmp_path / "run/pareton-deploy").mkdir(parents=True)
    (tmp_path / "etc/vector").mkdir(parents=True)
    runner = FakeRunner(tmp_path)
    monkeypatch.setattr(release, "run_cmd", runner)
    monkeypatch.setattr(release, "repo", lambda: tmp_path / "opt/pareton")
    return tmp_path


def write_state(base: Path, **overrides) -> dict:
    state = {
        "schema_version": 2,
        "op_id": "op-1",
        "phase": "idle",
        "scope": "full",
        "direction": "forward",
        "from_commit": "c1",
        "target_commit": "c1",
        "verified_commit": "c1",
        "hold": None,
        "original_units": {},
        "recovery_copy": None,
        "failure_step": None,
        "startup_complete": False,
        "log_accepted": False,
        "updated_at": release.now_iso(),
    }
    state.update(overrides)
    release.write_json_atomic(base / "var/lib/pareton-deploy/release-state.json", state)
    return state


def read_state(base: Path) -> dict:
    return json.loads(
        (base / "var/lib/pareton-deploy/release-state.json").read_text()
    )


def write_vector_toml(base: Path, units=None):
    units = units or [
        "pareton-worker", "pareton-round-worker", "pareton-watcher",
        "pareton-api", "pareton-weights", "pareton-gpu-reap",
        "pareton-deploy", "pareton-deploy-failed",
    ]
    lines = ["[sources.journald]", "type = \"journald\"",
             "include_units = [" + ", ".join(f'"{u}"' for u in units) + "]",
             "[sinks.axiom]", "type = \"axiom\"",
             "dataset = \"pareton-prod\""]
    (base / "etc/vector/vector.toml").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Schema and gate matrix (spec 4.5)


def test_validate_state_accepts_complete_schema(base):
    state = write_state(base)
    assert release.validate_state(state) is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 3},
        {"schema_version": None},
        {"phase": "unknown"},
        {"scope": "everything"},
        {"op_id": ""},
        {"verified_commit": None},
        {"direction": "sideways"},
        {"from_commit": 42},
    ],
)
def test_validate_state_rejects_corruption(base, overrides):
    state = write_state(base, **overrides)
    assert release.validate_state(state) is None


MATRIX = [
    # phase, scope, startup_complete, gate_open, exec_allow
    ("idle", "full", None, True, True),
    ("draining", "full", None, False, True),
    ("quiescing", "full", None, False, False),
    ("applying", "full", None, False, False),
    ("verifying", "full", True, True, True),
    ("verifying", "full", False, False, True),
    ("verified", "full", None, True, True),
    ("applying", "vector-only", None, True, True),
]


@pytest.mark.parametrize("phase,scope,startup,gate,exec_", MATRIX)
def test_gate_matrix(base, phase, scope, startup, gate, exec_):
    overrides = {"phase": phase, "scope": scope}
    if startup is not None:
        overrides["startup_complete"] = startup
    state = write_state(base, **overrides)
    assert release.gate_open(state) == (gate, release.gate_open(state)[1])
    assert release.exec_allow(state) == (exec_, release.exec_allow(state)[1])


@pytest.mark.parametrize("phase,scope,startup,gate,exec_", MATRIX)
def test_worker_matrix_matches_release(base, phase, scope, startup, gate, exec_):
    from worker import coordination

    overrides = {"phase": phase, "scope": scope}
    if startup is not None:
        overrides["startup_complete"] = startup
    state = write_state(base, **overrides)
    (base / "var/lib/pareton-deploy/release-state.json").write_text(
        json.dumps(state)
    )
    monkey_state = base / "var/lib/pareton-deploy/release-state.json"
    import fcntl

    orig = coordination.state_path
    coordination.state_path = lambda: monkey_state
    try:
        assert coordination.gate_open() == (gate, coordination.gate_open()[1])
    finally:
        coordination.state_path = orig


def test_corrupt_state_fails_closed(base):
    (base / "var/lib/pareton-deploy/release-state.json").write_text("{broken")
    assert release.gate_open(None) == (False, "state-corrupt")
    assert release.exec_allow(None) == (False, "state-corrupt")
    assert release.load_state() is None


# ---------------------------------------------------------------------------
# check-logs (spec 7.3)


def axiom_response(units):
    columns = [{"name": "_SYSTEMD_UNIT"}, {"name": "probe_id"}]
    rows = [[u, "probe-x"] for u in units]
    return {
        "status": {"isPartial": False},
        "tables": [{"columns": columns, "rows": rows}],
    }


@pytest.fixture()
def axiom(monkeypatch, base):
    write_vector_toml(base)
    monkeypatch.setenv("PARETON_LOG_WAIT_BUDGET_S", "1")
    (base / "opt/pareton/.env").write_text("PARETON_AXIOM_TOKEN=t\n")
    state = {"response": axiom_response([]), "status": 200, "error": None}
    monkeypatch.setattr(
        release, "http_post_json", lambda *a, **k: (state["status"], state["response"], state["error"])
    )
    monkeypatch.setattr(release.time, "sleep", lambda s: None)
    return state


def test_check_logs_complete(axiom, base):
    all_units = [
        "pareton-worker.service", "pareton-round-worker.service",
        "pareton-watcher.service", "pareton-api.service",
        "pareton-weights.service", "pareton-gpu-reap.service",
        "pareton-deploy.service",
    ]
    axiom["response"] = axiom_response(all_units)
    (base / "run/pareton-deploy/notification-acceptance.json")
    code, report = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1",
         "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 0
    assert report["missing"] == []


def test_check_logs_missing_source_is_exit_1(axiom, base):
    axiom["response"] = axiom_response(
        [u for u in ["pareton-api.service", "pareton-deploy.service"]]
    )
    code, report = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1",
         "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 1
    assert "pareton-weights.service" in report["missing"]
    assert report["missing"]


def test_check_logs_query_failure_is_exit_2(axiom, base):
    axiom["status"], axiom["response"], axiom["error"] = 401, None, "http-401"
    code, report = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1",
         "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 2


def test_check_logs_partial_result_keeps_polling(axiom, base):
    responses = [
        (200, {"status": {"isPartial": True}, "tables": []}, None),
        (200, axiom_response(
            [f"pareton-{u}.service" for u in
             ("worker", "round-worker", "watcher", "api", "weights",
              "gpu-reap", "deploy")]), None),
    ]
    calls = {"n": 0}

    def fake(url, payload, timeout):
        result = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return result

    release.http_post_json = fake
    code, _ = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1",
         "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 0
    assert calls["n"] >= 2


def test_check_logs_rejects_unknown_source(axiom, base):
    write_vector_toml(base, units=[
        "pareton-worker", "pareton-deploy", "pareton-deploy-failed",
        "pareton-mystery",
    ])
    code, report = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1",
         "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 2
    assert report["error"] == "no-probe-method"


# ---------------------------------------------------------------------------
# Notification acceptance validity (spec 7.4)


class FakeGitBlobs:
    def __init__(self, monkeypatch, blobs: dict[str, dict[str, bytes]]):
        self.blobs = blobs

        def fake_git(*args, timeout=300):
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            verb = args[0]
            if verb == "cat-file":
                # git("cat-file", "-p", f"{ref}:{rel}")
                ref, _, rel = args[2].partition(":")
                content = self.blobs.get(ref, {}).get(rel)
                if content is None:
                    r = Result()
                    r.returncode = 1
                    return r
                r = Result()
                r.stdout = content.decode() if isinstance(content, bytes) else content
                return r
            if verb == "ls-tree":
                r = Result()
                r.stdout = "\n".join(
                    f"ops/systemd/{u}" for u in (
                        "pareton-deploy.service",
                        "pareton-deploy-failed.service",
                    )
                )
                return r
            return Result()

        monkeypatch.setattr(release, "git", fake_git)
        monkeypatch.setattr(release, "git_out", lambda *a: "c9")


def acceptance_record(base: Path, commit="cA"):
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/notification-acceptance.json",
        {
            "host": release.platform.node(),
            "commit": commit,
            "vector_version": "vector 0.57.0",
            "drilled_at": "2026-09-12T00:00:00Z",
            "failure_invocation": "i",
            "discord_message_id": "m",
            "confirmed_by": "o",
        },
    )


def make_blob(**files) -> dict[str, dict[str, bytes]]:
    return {"cA": {k: v.encode() for k, v in files.items()},
            "cB": {k: v.encode() for k, v in files.items()}}


BASELINE_FILES = {
    "ops/notify-deploy-failure.py": "n",
    "ops/deploy.sh": "d",
    "ops/release.py": "r",
    "ops/ops_common.py": "o",
    "ops/systemd/pareton-deploy.service": "u1",
    "ops/systemd/pareton-deploy-failed.service": "u2",
    "ops/vector/vector.service": "v",
}


def toml_blob(units=None):
    units = units or ["pareton-worker", "pareton-deploy-failed"]
    body = (
        "[sources.journald]\ninclude_units = ["
        + ", ".join(f'"{u}"' for u in units) + "]\n"
        "[sinks.axiom]\ndataset = \"pareton-prod\"\n"
    )
    return body


def test_acceptance_ok_when_unchanged(base, monkeypatch):
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob().encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob().encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, detail = release.acceptance_status("cB")
    assert status == "ok"


def test_acceptance_missing_record_required(base, monkeypatch):
    FakeGitBlobs(monkeypatch, make_blob(**BASELINE_FILES))
    status, detail = release.acceptance_status("cB")
    assert status == "required"
    assert detail["reason"] == "record-missing"


def test_acceptance_chain_file_change_requires_drill(base, monkeypatch):
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob().encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob().encode()
    blobs["cB"]["ops/deploy.sh"] = "changed"
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, detail = release.acceptance_status("cB")
    assert status == "required"
    assert "ops/deploy.sh" in detail["reason"]


def test_acceptance_include_units_exemption(base, monkeypatch):
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob(
        ["pareton-worker", "pareton-deploy-failed"]).encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob(
        ["pareton-worker", "pareton-round-worker", "pareton-deploy-failed"]).encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, _ = release.acceptance_status("cB")
    assert status == "ok"


def test_acceptance_exemption_unit_removed_requires_drill(base, monkeypatch):
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob(
        ["pareton-worker", "pareton-deploy-failed"]).encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob(["pareton-worker"]).encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, detail = release.acceptance_status("cB")
    assert status == "required"
    assert detail["reason"] == "exempt-unit-removed"


def test_acceptance_other_toml_field_requires_drill(base, monkeypatch):
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob().encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob().replace(
        "pareton-prod", "other-dataset"
    ).encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, detail = release.acceptance_status("cB")
    assert status == "required"
    assert detail["reason"] == "vector-toml-changed"


def test_acceptance_host_change_requires_drill(base, monkeypatch):
    FakeGitBlobs(monkeypatch, make_blob(**BASELINE_FILES))
    record = {
        "host": "other-host", "commit": "cA", "vector_version": "v",
        "drilled_at": "t", "failure_invocation": "i",
        "discord_message_id": "m", "confirmed_by": "o",
    }
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/notification-acceptance.json", record
    )
    status, detail = release.acceptance_status("cA")
    assert status == "required"
    assert detail["reason"] == "host-changed"


# ---------------------------------------------------------------------------
# Tick state machine (spec 4.2)


def test_tick_peer_deploy_silent_zero(base, monkeypatch):
    import fcntl

    write_state(base)
    fd = os.open(str(base / "run/pareton-deploy.lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert release.tick([]) == 0
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_tick_idle_no_change(base, monkeypatch):
    write_state(base, verified_commit="c2")
    runner = release.run_cmd
    runner.git_refs = {"origin/main": "c2", "HEAD": "c2"}
    assert release.tick([]) == 0
    assert read_state(base)["phase"] == "idle"


def test_tick_corrupt_state_fails(base):
    (base / "var/lib/pareton-deploy/release-state.json").write_text("junk")
    assert release.tick([]) == 2


def test_tick_held_readonly_returns_zero(base):
    write_state(base, hold={"reason": "r", "operator": "o",
                            "at": release.now_iso(), "baseline_commit": "c1"})
    assert release.tick([]) == 0
    assert read_state(base)["hold"] is not None


def test_tick_new_commit_reaches_reexec(base, monkeypatch):
    write_state(base)
    make_mini_venv(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.db = {"error": None, "rounds": [], "submissions": []}
    execve = {}
    monkeypatch.setattr(
        release.os,
        "execve",
        lambda path, args, env: execve.update(path=path, args=args),
    )
    assert release.tick([]) == 0
    state = read_state(base)
    assert state["target_commit"] == "c2"
    assert state["phase"] == "applying"
    # The handover goes to the freshly installed entrypoint with the same
    # operation id (spec 6.2-3); lock fds travel via the environment.
    assert execve["args"][1].endswith("release.py")
    assert execve["args"][2:] == ["tick", "--continue-op", state["op_id"]]


def test_tick_drain_busy_under_threshold_returns_zero(base, monkeypatch):
    import fcntl

    write_state(base, phase="draining", phase_since=release.now_iso())
    fd = os.open(str(base / "run/pareton-activity.lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        assert release.tick([]) == 0
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_tick_drain_busy_over_threshold_fails(base):
    import fcntl

    old = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 4000)
    )
    write_state(base, phase="draining", phase_since=old)
    fd = os.open(str(base / "run/pareton-activity.lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        assert release.tick([]) == 1
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_tick_drain_db_error_fails(base):
    write_state(base, phase="draining", phase_since=release.now_iso())
    release.run_cmd.db = {"error": "OperationalError"}
    assert release.tick([]) == 2


def test_tick_drain_round_residual_fails(base):
    write_state(base, phase="draining", phase_since=release.now_iso())
    release.run_cmd.db = {
        "error": None,
        "rounds": [{"id": "r1", "ordinal": 3, "campaign_id": "c",
                    "heartbeat_at": "t"}],
        "submissions": [],
    }
    assert release.tick([]) == 2


def test_tick_drain_submission_residual_fails(base):
    write_state(base, phase="draining", phase_since=release.now_iso())
    release.run_cmd.db = {
        "error": None,
        "rounds": [],
        "submissions": [{"id": 9, "submission_id": "s", "attempts": 2,
                         "phase": "bench", "heartbeat_at": "t"}],
    }
    assert release.tick([]) == 2


def test_tick_applying_interrupted_requires_resume(base):
    write_state(base, phase="applying", target_commit="c2")
    assert release.tick([]) == 2


# ---------------------------------------------------------------------------
# gpu-reap-dispatch (spec 7.2)


def write_gpu_request(base, **overrides):
    request = {
        "invocation_id": "inv-1",
        "op_id": "op-1",
        "probe_id": "probe-g",
        "issued_at": release.now_iso(),
        "consumed": False,
    }
    request.update(overrides)
    release.write_json_atomic(
        base / "run/pareton-deploy/gpu-reap-request.json", request
    )
    return request


def test_gpu_dispatch_consumes_valid_request(base, monkeypatch, capsys):
    write_state(base, phase="verifying", op_id="op-1")
    write_gpu_request(base)
    release.run_cmd.deploy_invocation = "inv-1"
    assert release.cmd_gpu_reap_dispatch(["--", "true"]) == 0
    out = capsys.readouterr().out
    assert "gpu_reap_probe_warning" in out
    assert "deployment_probe" in out
    consumed = json.loads(
        (base / "run/pareton-deploy/gpu-reap-request.json").read_text()
    )
    assert consumed["consumed"] is True


def test_gpu_dispatch_expired_request_runs_real_command(base, monkeypatch):
    write_state(base, phase="verifying", op_id="op-1")
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 500))
    write_gpu_request(base, issued_at=old)
    release.run_cmd.deploy_invocation = "inv-1"
    execvp = {"cmd": None}
    monkeypatch.setattr(
        release.os, "execvp", lambda name, argv: execvp.update(cmd=argv)
    )
    release.cmd_gpu_reap_dispatch(["--", "python", "-m", "gpu", "reap"])
    assert execvp["cmd"] == ["python", "-m", "gpu", "reap"]
    request = json.loads(
        (base / "run/pareton-deploy/gpu-reap-request.json").read_text()
    )
    assert request["consumed"] is False


def test_gpu_dispatch_wrong_invocation_runs_real_command(base, monkeypatch):
    write_state(base, phase="verifying", op_id="op-1")
    write_gpu_request(base)
    release.run_cmd.deploy_invocation = "inv-OTHER"
    execvp = {"cmd": None}
    monkeypatch.setattr(
        release.os, "execvp", lambda name, argv: execvp.update(cmd=argv)
    )
    release.cmd_gpu_reap_dispatch(["--", "true"])
    assert execvp["cmd"] == ["true"]


def test_gpu_dispatch_consumed_request_runs_real_command(base, monkeypatch):
    write_state(base, phase="verifying", op_id="op-1")
    write_gpu_request(base, consumed=True)
    release.run_cmd.deploy_invocation = "inv-1"
    execvp = {"cmd": None}
    monkeypatch.setattr(
        release.os, "execvp", lambda name, argv: execvp.update(cmd=argv)
    )
    release.cmd_gpu_reap_dispatch(["--", "true"])
    assert execvp["cmd"] == ["true"]


# ---------------------------------------------------------------------------
# Requests (spec 6.3)


def test_request_hold_registers_immediately(base, capsys):
    write_state(base)
    assert release.main(["request", "hold", "--reason", "r",
                         "--operator", "o"]) == 0
    assert read_state(base)["hold"]["reason"] == "r"


def test_request_reset_requires_evidence(base):
    assert release.main(["request", "reset", "--operator", "o"]) == 2


def test_request_vector_repair_requires_target(base):
    assert release.main(["request", "vector-repair", "--operator", "o"]) == 2
