"""Offline tests for ops/release.py (stage-2 spec sections 4.5, 6.3, 7).

Everything runs against a temporary PARETON_RELEASE_BASE prefix with
git/systemctl/pip/venv subprocesses faked through release.run_cmd and the
Axiom HTTP layer faked through release.http_post_json. Real systemd and
Vector behavior is covered by the isolated acceptance matrix instead.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
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

# The ops interpreter prerequisite is Python >= 3.11 for stdlib tomllib
# (spec 2.3); on older interpreters the TOML paths degrade (whole-file
# comparison, fail-closed check-logs) and these parsed-TOML tests skip.
needs_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="ops interpreter prerequisite (spec 2.3)"
)


class FakeRunner:
    """Programmable stand-in for release.run_cmd."""

    def __init__(self, base: Path):
        self.base = base
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.git_refs = {"origin/main": "c2", "HEAD": "c1"}
        self.git_diff = ""
        self.db = {"error": None, "rounds": [], "submissions": []}
        self.active_units: set[str] = set()
        self.sync_exit = 0
        self.sync_stdout = ""
        self.sync_apply_exit = 0
        self.vector_version = "vector 0.57.0"
        self.pip_exit = 0
        self.deploy_invocation = "inv-1"

    def __call__(self, argv, env=None, timeout=300, cwd=None, pass_fds=()):
        self.calls.append(list(argv))
        self.kwargs.append({"env": env or {}, "pass_fds": pass_fds})
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
            elif "show" in argv and any("InvocationID" in part for part in argv):
                out = f"InvocationID={self.deploy_invocation}"
            rc = 0
        elif any(str(a).endswith("sync-config.py") for a in argv) and "apply" in argv:
            rc = self.sync_apply_exit
            out = ""
        elif any(str(a).endswith("sync-config.py") for a in argv):
            rc = self.sync_exit
            out = self.sync_stdout
        elif cmd.endswith("python"):
            if "-c" in argv:
                out = json.dumps(self.db)
        elif cmd == "vector" and "--version" in argv:
            out = self.vector_version
        elif cmd.endswith("pip"):
            rc = self.pip_exit
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
    monkeypatch.setenv("PARETON_OPS_DIR", str(tmp_path / "usr/local/lib/pareton-ops"))
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
    return json.loads((base / "var/lib/pareton-deploy/release-state.json").read_text())


def write_vector_toml(base: Path, units=None):
    units = units or [
        "pareton-worker",
        "pareton-round-worker",
        "pareton-watcher",
        "pareton-api",
        "pareton-weights",
        "pareton-gpu-reap",
        "pareton-deploy",
        "pareton-deploy-failed",
    ]
    lines = [
        "[sources.journald]",
        'type = "journald"',
        "include_units = [" + ", ".join(f'"{u}"' for u in units) + "]",
        "[sinks.axiom]",
        'type = "axiom"',
        'dataset = "pareton-prod"',
    ]
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
    (base / "var/lib/pareton-deploy/release-state.json").write_text(json.dumps(state))
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
    """Official tabular shape: fields[] names align with column-major
    columns[] arrays (Axiom docs, endpoints/queryApl; PR-review P1-1)."""
    return {
        "status": {"isPartial": False},
        "tables": [
            {
                "fields": [
                    {"name": "_SYSTEMD_UNIT", "type": "string"},
                    {"name": "probe_id", "type": "string"},
                ],
                "columns": [list(units), ["probe-x"] * len(units)],
            }
        ],
    }


@pytest.fixture()
def axiom(monkeypatch, base):
    write_vector_toml(base)
    monkeypatch.setenv("PARETON_LOG_WAIT_BUDGET_S", "1")
    (base / "opt/pareton/.env").write_text("PARETON_AXIOM_TOKEN=t\n")
    state = {"response": axiom_response([]), "status": 200, "error": None}
    monkeypatch.setattr(
        release,
        "http_post_json",
        lambda *a, **k: (state["status"], state["response"], state["error"]),
    )
    monkeypatch.setattr(release.time, "sleep", lambda s: None)
    return state


@needs_tomllib
def test_check_logs_complete(axiom, base):
    all_units = [
        "pareton-worker.service",
        "pareton-round-worker.service",
        "pareton-watcher.service",
        "pareton-api.service",
        "pareton-weights.service",
        "pareton-gpu-reap.service",
        "pareton-deploy.service",
    ]
    axiom["response"] = axiom_response(all_units)
    (base / "run/pareton-deploy/notification-acceptance.json")
    code, report = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1", "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 0
    assert report["missing"] == []


@needs_tomllib
def test_check_logs_missing_source_is_exit_1(axiom, base):
    axiom["response"] = axiom_response(
        [u for u in ["pareton-api.service", "pareton-deploy.service"]]
    )
    code, report = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1", "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 1
    assert "pareton-weights.service" in report["missing"]
    assert report["missing"]


@needs_tomllib
def test_check_logs_query_failure_is_exit_2(axiom, base):
    axiom["status"], axiom["response"], axiom["error"] = 401, None, "http-401"
    code, report = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1", "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 2


@needs_tomllib
def test_check_logs_partial_result_keeps_polling(axiom, base):
    responses = [
        (200, {"status": {"isPartial": True}, "tables": []}, None),
        (
            200,
            axiom_response(
                [
                    f"pareton-{u}.service"
                    for u in (
                        "worker",
                        "round-worker",
                        "watcher",
                        "api",
                        "weights",
                        "gpu-reap",
                        "deploy",
                    )
                ]
            ),
            None,
        ),
    ]
    calls = {"n": 0}

    def fake(url, payload, timeout, **kwargs):
        result = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return result

    release.http_post_json = fake
    code, _ = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1", "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 0
    assert calls["n"] >= 2


@needs_tomllib
def test_check_logs_rejects_unknown_source(axiom, base):
    write_vector_toml(
        base,
        units=[
            "pareton-worker",
            "pareton-deploy",
            "pareton-deploy-failed",
            "pareton-mystery",
        ],
    )
    code, report = release.run_log_check(
        {"probe_id": "probe-x", "target_commit": "c1", "issued_at": release.now_iso()},
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
                    f"ops/systemd/{u}"
                    for u in (
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
            "vector_version": "vector 0.57.0",
            "drilled_at": "2026-09-12T00:00:00Z",
            "failure_invocation": "i",
            "discord_message_id": "m",
            "confirmed_by": "o",
        },
    )


def make_blob(**files) -> dict[str, dict[str, bytes]]:
    return {
        "cA": {k: v.encode() for k, v in files.items()},
        "cB": {k: v.encode() for k, v in files.items()},
    }


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
        + ", ".join(f'"{u}"' for u in units)
        + "]\n"
        '[sinks.axiom]\ndataset = "pareton-prod"\n'
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


@needs_tomllib
def test_acceptance_include_units_exemption(base, monkeypatch):
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob(
        ["pareton-worker", "pareton-deploy-failed"]
    ).encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob(
        ["pareton-worker", "pareton-round-worker", "pareton-deploy-failed"]
    ).encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, _ = release.acceptance_status("cB")
    assert status == "ok"


@needs_tomllib
def test_acceptance_exemption_unit_removed_requires_drill(base, monkeypatch):
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob(
        ["pareton-worker", "pareton-deploy-failed"]
    ).encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob(["pareton-worker"]).encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, detail = release.acceptance_status("cB")
    assert status == "required"
    assert detail["reason"] == "exempt-unit-removed"


def test_acceptance_other_toml_field_requires_drill(base, monkeypatch):
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob().encode()
    blobs["cB"]["ops/vector/vector.toml"] = (
        toml_blob().replace("pareton-prod", "other-dataset").encode()
    )
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, detail = release.acceptance_status("cB")
    assert status == "required"
    assert detail["reason"] == "vector-toml-changed"


def test_acceptance_host_change_requires_drill(base, monkeypatch):
    FakeGitBlobs(monkeypatch, make_blob(**BASELINE_FILES))
    record = {
        "host": "other-host",
        "commit": "cA",
        "vector_version": "v",
        "drilled_at": "t",
        "failure_invocation": "i",
        "discord_message_id": "m",
        "confirmed_by": "o",
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
    write_state(
        base,
        hold={
            "reason": "r",
            "operator": "o",
            "at": release.now_iso(),
            "baseline_commit": "c1",
        },
    )
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

    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 4000))
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
        "rounds": [{"id": "r1", "ordinal": 3, "campaign_id": "c", "heartbeat_at": "t"}],
        "submissions": [],
    }
    assert release.tick([]) == 2


def test_tick_drain_submission_residual_fails(base):
    write_state(base, phase="draining", phase_since=release.now_iso())
    release.run_cmd.db = {
        "error": None,
        "rounds": [],
        "submissions": [
            {
                "id": 9,
                "submission_id": "s",
                "attempts": 2,
                "phase": "bench",
                "heartbeat_at": "t",
            }
        ],
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
    assert release.main(["request", "hold", "--reason", "r", "--operator", "o"]) == 0
    assert read_state(base)["hold"]["reason"] == "r"


def test_request_reset_requires_evidence(base):
    assert release.main(["request", "reset", "--operator", "o"]) == 2


def test_request_vector_repair_requires_target(base):
    assert release.main(["request", "vector-repair", "--operator", "o"]) == 2


# ---------------------------------------------------------------------------
# CR regression tests (2026-09-12 code review)


def test_unit_stopped_semantics(base, monkeypatch):
    # "deactivating" is still running: treating it as stopped let applies
    # start while a stop was mid-flight (CR P1-3).
    states = {
        "pareton-api.service": "deactivating",
        "pareton-watcher.service": "activating",
        "pareton-weights.service": "inactive",
        "pareton-worker.service": "failed",
    }

    def fake_run(argv, env=None, timeout=300, cwd=None):
        class Result:
            pass

        result = Result()
        result.returncode = 0
        result.stdout = states.get(argv[-1], "inactive")
        result.stderr = ""
        return result

    monkeypatch.setattr(release, "run_cmd", fake_run)
    assert release.unit_is_active("pareton-api.service") is True
    assert release.unit_is_active("pareton-watcher.service") is True
    assert release.unit_is_stopped("pareton-weights.service") is True
    assert release.unit_is_stopped("pareton-worker.service") is True
    assert release.unit_is_stopped("pareton-api.service") is False


def test_requirements_detection_is_full_tree():
    # Root-level requirements.txt must count (CR P1-1).
    assert release._requirements_changed(["requirements.txt", "ops/runbook.md"])
    assert release._requirements_changed(["api/requirements.txt"])
    assert not release._requirements_changed(["ops/vector/vector.toml"])


def test_pip_runs_when_requirements_changed(base, monkeypatch):
    write_state(base)
    make_mini_venv(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = "requirements.txt\n"
    release.run_cmd.db = {"error": None, "rounds": [], "submissions": []}
    execve = {}
    monkeypatch.setattr(
        release.os,
        "execve",
        lambda path, args, env: execve.update(path=path, args=args),
    )
    assert release.tick([]) == 0
    pip_calls = [c for c in release.run_cmd.calls if c and c[0].endswith("/pip")]
    assert pip_calls, "requirements change must install deps before re-exec"


def test_business_commit_plus_drift_is_not_vector_fast_path(base, monkeypatch):
    # A business commit with a concurrent vector drift must enter draining,
    # never the fast path whose git reset would land new code with the gate
    # open (CR P1-2).
    write_state(base)
    make_mini_venv(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = "worker/main.py\n"
    release.run_cmd.sync_exit = 1
    release.run_cmd.sync_stdout = json.dumps(
        {"findings": [{"category": "different", "target": "/etc/vector/vector.toml"}]}
    )
    release.run_cmd.db = {"error": None, "rounds": [], "submissions": []}
    execve = {}
    monkeypatch.setattr(
        release.os,
        "execve",
        lambda path, args, env: execve.update(path=path, args=args),
    )
    assert release.tick([]) == 0
    state = read_state(base)
    assert state["phase"] == "applying"
    assert state["scope"] == "full"


def test_vector_fast_path_happy_path(base, monkeypatch):
    write_state(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = "ops/vector/vector.toml\n"
    release.run_cmd.sync_exit = 1
    release.run_cmd.sync_stdout = json.dumps(
        {"findings": [{"category": "different", "target": "/etc/vector/vector.toml"}]}
    )
    monkeypatch.setattr(release, "gpu_probe_flow", lambda op_id, probe: None)
    monkeypatch.setattr(
        release, "run_log_check", lambda probe, **kw: (0, {"missing": []})
    )
    assert release.tick([]) == 0
    state = read_state(base)
    assert state["phase"] == "idle"
    assert state["scope"] == "full"
    assert state["verified_commit"] == "c2"


def test_vector_fast_path_rejects_dirty_worktree(base, monkeypatch):
    write_state(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = "ops/vector/vector.toml\n"
    release.run_cmd.sync_exit = 1
    release.run_cmd.sync_stdout = json.dumps(
        {"findings": [{"category": "different", "target": "/etc/vector/vector.toml"}]}
    )

    def fake_git(*args, timeout=300):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        if args[0] == "status":
            result = Result()
            result.stdout = " M worker/main.py\n"
            return result
        return release.run_cmd(["git", *args])

    monkeypatch.setattr(release, "git", fake_git)
    monkeypatch.setattr(release, "git_out", lambda *a: "c2")
    assert release.tick([]) == 2
    state = read_state(base)
    assert state["phase"] == "idle"
    assert state["failure_step"] == "worktree-dirty"


def test_running_rollback_not_stranded_by_hold(base):
    # A rollback waiting on busy workers must continue on later ticks, not
    # fall into read-only held mode forever (CR P1-4).
    import fcntl

    write_state(
        base,
        phase="draining",
        direction="rollback",
        hold={
            "reason": "rollback",
            "operator": "o",
            "at": release.now_iso(),
            "baseline_commit": "c1",
        },
        recovery_copy="/tmp/copy",
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "rollback",
            "status": "running",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    fd = os.open(str(base / "run/pareton-activity.lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        assert release.tick([]) == 0  # active-work wait, not held
        request = json.loads(
            (base / "var/lib/pareton-deploy/release-request.json").read_text()
        )
        assert request["status"] == "running"
        assert read_state(base)["phase"] == "draining"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_refused_request_marked_failed(base):
    write_state(
        base,
        hold={
            "reason": "r",
            "operator": "o",
            "at": release.now_iso(),
            "baseline_commit": "c1",
        },
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "unpause",
            "status": "pending",
            "operator": "o",
            "main_commit": "cold",
            "registered_at": release.now_iso(),
        },
    )
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    assert release.tick([]) == 1  # main moved: refused
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "failed"
    assert request["result"]["step"] == "unpause-main-moved"


def test_verify_from_idle_enters_verifying(base, monkeypatch):
    write_state(base, phase="idle")
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "verify",
            "status": "pending",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    monkeypatch.setattr(release, "verify_flow", lambda state, **kw: 0)
    monkeypatch.setattr(release, "gpu_probe_flow", lambda *a: None)
    assert release.tick([]) == 0
    # verify_flow received a state already in verifying (the GPU dispatch
    # only consumes one-shot requests in that phase; CR P2-2)
    seen = {}

    def spy(state, **kw):
        seen["phase"] = state["phase"]
        return 0

    monkeypatch.setattr(release, "verify_flow", spy)
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "verify",
            "status": "pending",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    write_state(base, phase="idle")
    assert release.tick([]) == 0
    assert seen["phase"] == "verifying"


def test_cancel_restores_phase_before_starts(base, monkeypatch):
    write_state(
        base, phase="quiescing", original_units={"pareton-api": {"active": True}}
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "cancel",
            "status": "running",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    order = []
    real_start = release.start_unit

    def spy_start(unit):
        order.append(("start", unit, read_state(base)["phase"]))
        return real_start(unit)

    monkeypatch.setattr(release, "start_unit", spy_start)
    assert release.tick([]) == 0
    assert order, "cancel must start the snapshot-active residents"
    for _kind, _unit, phase in order:
        assert phase == "idle", "starts must happen after the phase restore"
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "done"


def test_log_failure_restores_maint_timers(base, monkeypatch):
    write_state(
        base,
        phase="verifying",
        startup_complete=True,
        target_commit="c2",
        original_units={"pareton-gpu-reap.timer": {"active": True}},
    )
    started = []
    monkeypatch.setattr(release, "start_unit", lambda u: started.append(u))
    code = release.finish_verification(
        read_state(base), 1, {"missing": ["pareton-api.service"]}
    )
    assert code == 1
    assert "pareton-gpu-reap.timer" in started  # CR P2-1


def test_no_auto_reverify_after_log_failure(base):
    write_state(
        base,
        phase="verifying",
        startup_complete=True,
        failure_step="log-ingestion",
    )
    assert release.tick([]) == 0
    run_state = (base / "var/lib/pareton-deploy/last-run.env").read_text()
    assert "last_step=log-unaccepted" in run_state
    assert not (base / "run/pareton-deploy/probe.json").exists()


def test_gpu_wait_timeout_reports_and_restores(base, monkeypatch):
    write_state(
        base,
        phase="verifying",
        startup_complete=True,
        original_units={"pareton-gpu-reap.timer": {"active": True}},
    )
    monkeypatch.setenv("PARETON_GPU_REAP_WAIT_S", "0")
    release.run_cmd.active_units = {"pareton-gpu-reap.service"}
    started = []
    monkeypatch.setattr(release, "start_unit", lambda u: started.append(u))
    with pytest.raises(release.Fail) as info:
        release.gpu_probe_flow("op-1", {"probe_id": "p"})
    assert info.value.reason == "gpu-reap-wait-timeout"
    assert "pareton-gpu-reap.timer" in started  # CR P2-3


def test_record_step_started_at_is_tick_time(base, monkeypatch):
    monkeypatch.setenv("PARETON_TEST_NOW", "2026-09-12T12:00:00Z")
    write_state(base, updated_at="2026-09-11T00:00:00Z")
    release.run_cmd.git_refs = {}  # rev-parse fails -> early tick failure
    assert release.tick([]) == 2
    run_state = (base / "var/lib/pareton-deploy/last-run.env").read_text()
    assert "started_at=2026-09-12T12:00:00Z" in run_state  # CR P1-5
    assert "2026-09-11" not in run_state


def test_venv_swap_failure_keeps_original(base, monkeypatch):
    make_mini_venv(base)
    copy_dir = base / "var/lib/pareton-deploy/recovery/1"
    shutil.copytree(base / "opt/pareton/.venv", copy_dir, symlinks=True)
    real_rename = os.rename
    renames = {"n": 0}

    def flaky_rename(src, dst):
        renames["n"] += 1
        if renames["n"] == 2:
            raise OSError("simulated swap failure")
        return real_rename(src, dst)

    monkeypatch.setattr(release.os, "rename", flaky_rename)
    with pytest.raises(release.Fail):
        release.restore_recovery_venv(copy_dir)
    # The original venv survived the failed swap (CR P3).
    assert (base / "opt/pareton/.venv/pyvenv.cfg").is_file()
    assert (base / "opt/pareton/.venv/bin/python").exists()


def test_venv_restore_with_missing_target(base):
    # A swap that died between the two renames leaves no .venv: the restore
    # must place the copy instead of crashing on the missing target.
    make_mini_venv(base)
    copy_dir = base / "var/lib/pareton-deploy/recovery/1"
    shutil.copytree(base / "opt/pareton/.venv", copy_dir, symlinks=True)
    (base / "opt/pareton/.venv/bin/marker").write_text("old\n")
    leftover = base / "opt/pareton/.venv.displaced.999"
    shutil.copytree(base / "opt/pareton/.venv", leftover, symlinks=True)
    shutil.rmtree(base / "opt/pareton/.venv")
    release.restore_recovery_venv(copy_dir)
    assert (base / "opt/pareton/.venv/pyvenv.cfg").is_file()
    assert not (base / "opt/pareton/.venv/bin/marker").exists()


def test_rollback_after_success_targets_recovery_commit(base, monkeypatch):
    # Post-success, verified_commit == the current commit; the rollback must
    # return to the commit the recovery copy captures, not pair old deps
    # with new code (B12; found while building S8).
    write_state(
        base,
        phase="idle",
        verified_commit="B",
        from_commit="A",
        target_commit="B",
        recovery_copy="/tmp/copy",
        recovery_commit="A",
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "rollback",
            "status": "pending",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    release.run_cmd.db = {"error": None, "rounds": [], "submissions": []}
    make_mini_venv(base)
    copy_dir = base / "var/lib/pareton-deploy/recovery/1"
    shutil.copytree(base / "opt/pareton/.venv", copy_dir, symlinks=True)
    write_state(
        base,
        phase="idle",
        verified_commit="B",
        from_commit="A",
        target_commit="B",
        recovery_copy=str(copy_dir),
        recovery_commit="A",
    )
    execve = {}
    monkeypatch.setattr(
        release.os, "execve", lambda path, args, env: execve.update(args=args)
    )
    assert release.tick([]) == 0
    state = read_state(base)
    assert state["target_commit"] == "A"
    assert state["direction"] == "rollback"
    # The venv came from the recovery copy, not pip.
    pip_calls = [c for c in release.run_cmd.calls if c and c[0].endswith("/pip")]
    assert not pip_calls


def test_rollback_hold_anchors_to_target(base, monkeypatch):
    # hold.baseline_commit must say where the environment is heading, not
    # the pre-rollback verified commit, or `status` reads as a no-op (obs 1).
    write_state(
        base,
        phase="idle",
        verified_commit="B",
        from_commit="A",
        target_commit="B",
        recovery_copy=None,
    )
    copy_dir = base / "var/lib/pareton-deploy/recovery/1"
    make_mini_venv(base)
    shutil.copytree(base / "opt/pareton/.venv", copy_dir, symlinks=True)
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/recovery/1/recovery-meta.json",
        {"from_commit": "A", "target_commit": "B"},
    )
    write_state(
        base,
        phase="idle",
        verified_commit="B",
        recovery_copy=str(copy_dir),
        # No recovery_commit: the pre-field state shape (obs 2) — the copy's
        # own meta must supply the target.
    )
    assert release.rollback_target(read_state(base)) == "A"
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "rollback",
            "status": "pending",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    release.run_cmd.db = {"error": None, "rounds": [], "submissions": []}
    execve = {}
    monkeypatch.setattr(
        release.os, "execve", lambda path, args, env: execve.update(args=args)
    )
    assert release.tick([]) == 0
    state = read_state(base)
    assert state["hold"]["baseline_commit"] == "A"
    assert state["target_commit"] == "A"


# ---------------------------------------------------------------------------
# PR-review (Bugbot) regressions


@needs_tomllib
def test_axiom_query_carries_bearer_token(axiom, base, monkeypatch):
    captured = {}

    def fake(url, payload, timeout, token=""):
        captured["token"] = token
        return (
            200,
            axiom_response(
                [
                    f"pareton-{u}.service"
                    for u in (
                        "worker",
                        "round-worker",
                        "watcher",
                        "api",
                        "weights",
                        "gpu-reap",
                        "deploy",
                    )
                ]
            ),
            None,
        )

    monkeypatch.setattr(release, "http_post_json", fake)
    code, _ = release.run_log_check(
        {"probe_id": "p", "target_commit": "c", "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 0
    assert captured["token"] == "t"  # from the fixture .env


def test_cancel_in_draining_completes_without_waiting(base):
    # Draining sent no stop signals; cancel must finish in one tick instead
    # of waiting for units that are merely serving (Bugbot High 2).
    write_state(
        base, phase="draining", original_units={"pareton-api": {"active": True}}
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "cancel",
            "status": "pending",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    release.run_cmd.active_units = {"pareton-api.service"}  # still serving
    assert release.tick([]) == 0
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "done"
    assert read_state(base)["phase"] == "idle"


def test_worker_start_failure_parks_in_partial_startup(base, monkeypatch):
    # A worker that fails to start must leave startup_complete False so the
    # next tick reports partial startup; the old code set it True after the
    # residents and then resumed past the dead workers (Bugbot High 3).
    import fcntl

    write_state(base)
    make_mini_venv(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = ""
    release.run_cmd.db = {"error": None, "rounds": [], "submissions": []}
    execve = {}
    monkeypatch.setattr(
        release.os, "execve", lambda path, args, env: execve.update(args=args)
    )
    monkeypatch.setattr(release, "api_healthy", lambda: True)
    monkeypatch.setattr(release, "API_HEALTH_TIMEOUT_S", 0)
    # All units stay inactive through quiescing/apply; residents come up
    # for verify while the workers never do.
    assert release.tick([]) == 0  # apply done, re-exec stubbed
    state = read_state(base)
    assert state["phase"] == "applying"
    release.run_cmd.active_units = {
        "pareton-api",
        "pareton-watcher",
        "pareton-weights",
    }
    # Drive the verify stage exactly as the re-exec would, with a real
    # inherited deploy-lock fd (tick_continue releases and closes it).
    fd = os.open(str(base / "run/pareton-deploy.lock"), os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    monkeypatch.setenv("PARETON_INHERIT_DEPLOY_LOCK_FD", str(fd))
    assert release.tick(["--continue-op", state["op_id"]]) == 2
    state = read_state(base)
    assert state["startup_complete"] is False
    assert state["failure_step"] == "start-failed"
    # Next tick reports the partial startup instead of skipping it.
    assert release.tick([]) == 2
    run_state = (base / "var/lib/pareton-deploy/last-run.env").read_text()
    assert "verifying-partial-startup" in run_state


def test_start_failure_restores_timers_and_finishes_request(base, monkeypatch):
    # The start-failure park must not strand maintenance timers or a
    # running request: GPU TTL reaping resumes, and the operator can
    # register the resume the error message calls for (Bugbot High).
    import fcntl

    state = write_state(
        base,
        phase="applying",
        direction="rollback",
        target_commit="c2",
        op_id="op-ver",
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "rollback",
            "status": "running",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    monkeypatch.setattr(release, "api_healthy", lambda: True)
    monkeypatch.setattr(release, "API_HEALTH_TIMEOUT_S", 0)
    # Residents healthy, workers never come up.
    release.run_cmd.active_units = {
        "pareton-api",
        "pareton-watcher",
        "pareton-weights",
    }
    started = []
    real_start = release.start_unit
    monkeypatch.setattr(
        release, "start_unit", lambda u: (started.append(u), real_start(u))
    )
    fd = os.open(str(base / "run/pareton-deploy.lock"), os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    monkeypatch.setenv("PARETON_INHERIT_DEPLOY_LOCK_FD", str(fd))
    assert release.tick(["--continue-op", "op-ver"]) == 2
    # Maintenance timers restored.
    assert "pareton-gpu-reap.timer" in started
    assert "pareton-builder-cleanup.timer" in started
    # The in-flight request reached a terminal state...
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "failed"
    assert request["result"]["step"] == "start-failed"
    # ...so the operator can register the recovery request right away.
    assert release.cmd_request(["resume", "--operator", "o"]) == 0
    registered = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert registered["type"] == "resume"
    assert registered["status"] == "pending"


def test_unpause_not_idle_refusal_is_terminal(base):
    write_state(
        base,
        phase="verifying",
        startup_complete=True,
        hold={
            "reason": "r",
            "operator": "o",
            "at": release.now_iso(),
            "baseline_commit": "c1",
        },
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "unpause",
            "status": "pending",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    assert release.tick([]) == 1
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "failed"
    assert request["result"]["step"] == "unpause-not-idle"


def test_vector_fast_path_log_failure_is_unaccepted_steady_state(base, monkeypatch):
    # The worktree already moved; idling would make the next tick a FULL
    # drain for a TOML-only change. The failure must park in verifying with
    # log_accepted=False (Bugbot Medium 6).
    write_state(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = "ops/vector/vector.toml\n"
    release.run_cmd.sync_exit = 1
    release.run_cmd.sync_stdout = json.dumps(
        {"findings": [{"category": "different", "target": "/etc/vector/vector.toml"}]}
    )
    monkeypatch.setattr(release, "gpu_probe_flow", lambda op_id, probe: None)
    monkeypatch.setattr(
        release,
        "run_log_check",
        lambda probe, **kw: (1, {"missing": ["pareton-api.service"]}),
    )
    assert release.tick([]) == 1
    state = read_state(base)
    assert state["phase"] == "verifying"
    assert state["scope"] == "vector-only"
    assert state["log_accepted"] is False
    assert state["failure_step"] == "log-ingestion"
    assert state["verified_commit"] == "c1"  # not advanced
    # Steady state: no auto re-verify.
    assert release.tick([]) == 0
    run_state = (base / "var/lib/pareton-deploy/last-run.env").read_text()
    assert "last_step=log-unaccepted" in run_state


def test_degraded_mode_fails_closed_and_compares_whole_files(base, monkeypatch):
    # Interpreter without tomllib (CI's 3.10 leg, or a <3.11 production
    # system before a managed interpreter is arranged): check-logs fails
    # closed with a distinct category, and the TOML drill rule degrades to
    # whole-file comparison instead of silently waiving (spec 7.4).
    write_vector_toml(base)
    monkeypatch.setattr(release, "tomllib", None)
    code, report = release.run_log_check(
        {"probe_id": "p", "target_commit": "c", "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 2
    assert report["error"] == "tomllib-unavailable"

    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob().encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob().encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, _ = release.acceptance_status("cB")
    assert status == "ok"  # byte-identical TOML is still fine
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob(
        ["pareton-worker", "pareton-round-worker", "pareton-deploy-failed"]
    ).encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    status, detail = release.acceptance_status("cB")
    # In degraded mode even the include_units exemption does not engage.
    assert status == "required"
    assert detail["reason"] == "vector-toml-changed"


# ---------------------------------------------------------------------------
# Independent PR-review regressions (six P1 + four P2)


@needs_tomllib
def test_axiom_official_shape_parses_and_empty_is_valid(axiom, base):
    # Official tabular: fields[] names + column-major columns[]. Empty
    # result = zero-height columns, not a missing "rows" key (P1-1).
    axiom["response"] = {
        "status": {"isPartial": False},
        "tables": [
            {
                "fields": [{"name": "probe_id"}, {"name": "_SYSTEMD_UNIT"}],
                "columns": [
                    ["p"] * 7,
                    [
                        f"pareton-{u}.service"
                        for u in (
                            "worker",
                            "round-worker",
                            "watcher",
                            "api",
                            "weights",
                            "gpu-reap",
                            "deploy",
                        )
                    ],
                ],
            }
        ],
    }
    code, report = release.run_log_check(
        {"probe_id": "p", "target_commit": "c", "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 0
    # An empty table is a valid "no events yet" answer, not a parse error.
    axiom["response"] = {
        "status": {"isPartial": False},
        "tables": [{"fields": [{"name": "_SYSTEMD_UNIT"}], "columns": [[]]}],
    }
    code, report = release.run_log_check(
        {"probe_id": "p", "target_commit": "c", "issued_at": release.now_iso()},
        skip_acceptance=True,
    )
    assert code == 1  # missing sources, not a query/parse failure


@needs_tomllib
def test_drill_evidence_requires_sent_matching_notification(axiom, base, monkeypatch):
    # A suppressed or mismatched notification is not drill evidence (P1-1).
    write_vector_toml(base)
    monkeypatch.setenv("PARETON_AXIOM_QUERY_TOKEN", "t")
    (base / "opt/pareton/.env").write_text("PARETON_AXIOM_TOKEN=t\n")
    monkeypatch.setattr(
        release,
        "http_post_json",
        lambda *a, **k: (
            200,
            {
                "status": {"isPartial": False},
                "tables": [
                    {
                        "fields": [
                            {"name": "invocation_id"},
                            {"name": "outcome"},
                            {"name": "message_id"},
                            {"name": "host"},
                        ],
                        "columns": [
                            ["inv-1"],
                            ["suppressed"],
                            ["m-1"],
                            [release.platform.node()],
                        ],
                    }
                ],
            },
            None,
        ),
    )
    monkeypatch.setattr(release, "git_out", lambda *a: "cA")
    assert (
        release.cmd_record_acceptance(
            ["--invocation", "inv-1", "--message-id", "m-1", "--confirmed-by", "o"]
        )
        == 1
    )


def test_activating_auto_restart_is_not_healthy(base, monkeypatch):
    # A crash-looping worker shows "activating"; startup acceptance must
    # require exactly "active" (P1-2).
    states = {"pareton-worker.service": "activating"}

    def fake_run(argv, env=None, timeout=300, cwd=None, pass_fds=()):
        class Result:
            pass

        result = Result()
        result.returncode = 0
        result.stdout = states.get(argv[-1], "inactive")
        result.stderr = ""
        return result

    monkeypatch.setattr(release, "run_cmd", fake_run)
    assert release.unit_strictly_active("pareton-worker.service") is False
    assert release.unit_is_active("pareton-worker.service") is True  # stop-wait view


def test_cancel_resets_target_to_installed_version(base, monkeypatch):
    # A cancelled A->B release leaves A installed; the abandoned target
    # must not survive for a later verify to certify (P1-3).
    write_state(base, phase="quiescing", verified_commit="A", target_commit="B")
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "cancel",
            "status": "pending",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    assert release.tick([]) == 0
    state = read_state(base)
    assert state["phase"] == "idle"
    assert state["target_commit"] == "A"
    assert state["from_commit"] == "A"
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "done"


def test_applying_install_failure_fails_running_resume(base, monkeypatch):
    # A resume whose pip fails again must free the request slot — a stuck
    # running request blocks the next explicit recovery (P1-4).
    import fcntl

    state = write_state(
        base,
        phase="applying",
        direction="forward",
        target_commit="c2",
        verified_commit="c1",
        from_commit="c1",
        op_id="op-a",
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "resume",
            "status": "running",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    make_mini_venv(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = "requirements.txt\n"
    release.run_cmd.pip_exit = 1  # the re-install fails
    release.run_cmd.db = {"error": None, "rounds": [], "submissions": []}
    execve = {}
    monkeypatch.setattr(
        release.os, "execve", lambda path, args, env: execve.update(args=args)
    )
    # _request_resume requires apply/verify; drive the continuation via the
    # phase machine exactly as a resumed tick would.
    lock_fd = os.open(str(base / "run/pareton-deploy.lock"), os.O_RDWR | os.O_CREAT)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    monkeypatch.setattr(release, "_DEPLOY_LOCK_FD", lock_fd)
    try:
        assert release.tick_applying(read_state(base)) == 2
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "failed"
    assert request["result"]["step"] == "deps-failed"
    # The slot is free: the next recovery registers immediately.
    assert release.cmd_request(["rollback", "--operator", "o"]) == 0


def test_sync_write_modes_refuse_during_release(base, tmp_path, monkeypatch):
    # Manual apply must not bypass the deploy mutex, an interrupted apply,
    # or a held state (P1-5).
    sync = load_ops_module("sync-config")
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(OPS, repo / "ops")
    for sub in ("systemd", "gpu", "vector"):
        (repo / "ops" / sub).mkdir(exist_ok=True)
    lock = tmp_path / "deploy.lock"
    state_file = tmp_path / "release-state.json"
    monkeypatch.setenv("PARETON_SYNC_BASE", str(tmp_path / "base"))
    monkeypatch.setenv("PARETON_DEPLOY_LOCK", str(lock))
    monkeypatch.setenv("PARETON_RELEASE_STATE", str(state_file))
    monkeypatch.setenv("PARETON_ACTIVITY_LOCK", str(tmp_path / "activity.lock"))

    import io
    from contextlib import redirect_stderr, redirect_stdout

    def run_apply():
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = sync.main(["apply", "--repo", str(repo), "--source", "worktree"])
        return code

    # ANY stage-2 state — even idle with no work in flight — makes writes
    # coordinator-owned: probing locks and releasing them is a TOCTOU
    # window during validation/install (review R3-1).
    for state in (
        {"phase": "applying"},
        {"phase": "quiescing"},
        {"phase": "draining"},
        {"phase": "idle", "hold": {"reason": "r"}},
        {"phase": "idle"},
    ):
        release.write_json_atomic(state_file, state)
        assert run_apply() == 3
    # Fresh bootstrap (no state yet): only the deploy-mutex race remains.
    state_file.unlink()
    import fcntl

    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert run_apply() == 3  # deploy-in-progress
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_maintenance_services_carry_the_gate():
    # P1-6: both maintenance services gate on release state before any
    # venv-dependent step.
    gpu = (OPS / "gpu" / "pareton-gpu-reap.service").read_text()
    assert "ExecCondition=/usr/local/lib/pareton-ops/release.py gate" in gpu
    assert gpu.index("ExecCondition=") < gpu.index("ExecStart=")
    cleanup = (OPS / "systemd" / "pareton-builder-cleanup.service").read_text()
    assert "ExecCondition=/usr/local/lib/pareton-ops/release.py gate" in cleanup
    assert cleanup.index("ExecCondition=") < cleanup.index("ExecStartPre=")


def test_cancel_waits_only_deactivating_units(base):
    # Early-quiescing cancel: never-signalled residents keep running; only
    # in-flight stops are waited on (P2-1).
    write_state(
        base, phase="quiescing", original_units={"pareton-api": {"active": True}}
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "cancel",
            "status": "pending",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    # Residents are active (serving) but NOT deactivating.
    release.run_cmd.active_units = {"pareton-api", "pareton-watcher", "pareton-weights"}
    assert release.tick([]) == 0
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "done"
    assert read_state(base)["phase"] == "idle"


def test_query_token_prefers_env_then_envfile_query_token(base, monkeypatch):
    # P2-3: manual registrations read the dedicated query credential from
    # .env, not only from the unit-injected environment.
    monkeypatch.delenv("PARETON_AXIOM_QUERY_TOKEN", raising=False)
    (base / "opt/pareton/.env").write_text(
        "PARETON_AXIOM_TOKEN=ingest\nPARETON_AXIOM_QUERY_TOKEN=query\n"
    )
    assert release.axiom_query_token() == ("query", None)
    monkeypatch.setenv("PARETON_AXIOM_QUERY_TOKEN", "env")
    assert release.axiom_query_token() == ("env", None)
    monkeypatch.delenv("PARETON_AXIOM_QUERY_TOKEN")
    (base / "opt/pareton/.env").write_text("PARETON_AXIOM_TOKEN=ingest\n")
    assert release.axiom_query_token() == ("ingest", None)


def test_vector_version_change_requires_new_drill(base, monkeypatch):
    # P2-4: the stored version is actually compared; unknown current
    # version also requires a re-drill.
    blobs = make_blob(**BASELINE_FILES)
    blobs["cA"]["ops/vector/vector.toml"] = toml_blob().encode()
    blobs["cB"]["ops/vector/vector.toml"] = toml_blob().encode()
    FakeGitBlobs(monkeypatch, blobs)
    acceptance_record(base)
    assert release.acceptance_status("cB")[0] == "ok"
    release.run_cmd.vector_version = "vector 0.58.0"
    status, detail = release.acceptance_status("cB")
    assert status == "required"
    assert detail["reason"] == "vector-version-changed"
    release.run_cmd.vector_version = ""  # unobtainable
    status, detail = release.acceptance_status("cB")
    assert detail["reason"] == "vector-version-unknown"


# ---------------------------------------------------------------------------
# Review round-2 regressions (R2-1..R2-6)


def test_vector_fast_path_hands_the_lock_to_sync(base, monkeypatch):
    # R2-3: the fast path's sync subprocess must inherit the deploy lock;
    # without it the coordination guard refuses with deploy-in-progress.
    write_state(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = "ops/vector/vector.toml\n"
    release.run_cmd.sync_exit = 1
    release.run_cmd.sync_stdout = json.dumps(
        {"findings": [{"category": "different", "target": "/etc/vector/vector.toml"}]}
    )
    monkeypatch.setattr(release, "gpu_probe_flow", lambda op_id, probe: None)
    monkeypatch.setattr(
        release, "run_log_check", lambda probe, **kw: (0, {"missing": []})
    )
    assert release.tick([]) == 0
    apply_kwargs = [
        release.run_cmd.kwargs[i]
        for i, argv in enumerate(release.run_cmd.calls)
        if any(str(a).endswith("sync-config.py") for a in argv) and "apply" in argv
    ]
    assert apply_kwargs, "fast path must call sync-config apply"
    for kw in apply_kwargs:
        assert kw["env"].get("PARETON_INHERIT_DEPLOY_LOCK_FD")
        assert kw["pass_fds"]


def test_sync_guard_accepts_inherited_lock_in_real_subprocess(tmp_path, monkeypatch):
    # The fd-passing mechanics for real: a subprocess holding nothing
    # refuses; the same subprocess launched with the parent's lock fd
    # passes the guard (review R2-3 requirement: exercise the actual
    # subprocess handoff, not a FakeRunner).
    sync = load_ops_module("sync-config")
    lock = tmp_path / "deploy.lock"
    state = tmp_path / "release-state.json"
    state_file = state  # the guard probe below uses this name
    monkeypatch.setenv("PARETON_DEPLOY_LOCK", str(lock))
    monkeypatch.setenv("PARETON_RELEASE_STATE", str(state))
    monkeypatch.setenv("PARETON_ACTIVITY_LOCK", str(tmp_path / "activity.lock"))
    release.write_json_atomic(state, {"phase": "draining"})  # hostile to manual

    import fcntl
    import subprocess

    guard_script = tmp_path / "guard_probe.py"
    guard_script.write_text(
        "import importlib.util\n"
        "import sys\n"
        f"spec = importlib.util.spec_from_file_location('syncmod', {str(OPS)!r} + '/sync-config.py')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "try:\n"
        "    m.guard_release_coordination()\n"
        "    print('ok')\n"
        "except m.Fail as f:\n"
        "    print('refused', f.reason)\n"
        "    sys.exit(3)\n"
    )
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        # Fresh bootstrap (no state) + the parent's lock, no handoff:
        # refused as a concurrent deploy.
        state_file.unlink()
        result = subprocess.run(
            [sys.executable, str(guard_script)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 3, result.stdout + result.stderr
        assert "deploy-in-progress" in result.stdout

        # With stage-2 state and no handoff: writes are coordinator-owned
        # regardless of the lock (review R3-1).
        release.write_json_atomic(state_file, {"phase": "draining"})
        result = subprocess.run(
            [sys.executable, str(guard_script)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 3, result.stdout + result.stderr
        assert "coordinator-owned" in result.stdout

        # With the inherited fd: the guard accepts (same OFD re-flock).
        result = subprocess.run(
            [sys.executable, str(guard_script)],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PARETON_INHERIT_DEPLOY_LOCK_FD": str(fd)},
            pass_fds=(fd,),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ok" in result.stdout
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_pip_timeout_frees_the_request(base, monkeypatch):
    # R2-1: TimeoutExpired is not a Fail return; the tick-level net must
    # finish the in-flight request so re-registration works.
    write_state(
        base,
        phase="draining",
        direction="forward",
        target_commit="c2",
        verified_commit="c1",
        from_commit="c1",
        op_id="op-t",
        phase_since=release.now_iso(),
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "resume",
            "status": "running",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    make_mini_venv(base)
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    release.run_cmd.git_diff = "requirements.txt\n"
    release.run_cmd.db = {"error": None, "rounds": [], "submissions": []}

    real_run_cmd = release.run_cmd

    def exploding_run_cmd(argv, **kwargs):
        if argv and str(argv[0]).endswith("/pip"):
            raise subprocess.TimeoutExpired(argv, 3600)
        return real_run_cmd(argv, **kwargs)

    monkeypatch.setattr(release, "run_cmd", exploding_run_cmd)
    assert release.tick([]) == 2  # drained, applied, pip timed out
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "failed"
    assert request["result"]["step"] == "install-timeout"
    assert release.cmd_request(["rollback", "--operator", "o"]) == 0


def test_interrupted_applying_frees_request_slot(base):
    # R2-1: a killed process cannot finish its request; the next tick must
    # free the slot while keeping phase and materials for resume/rollback.
    write_state(
        base,
        phase="applying",
        direction="forward",
        target_commit="c2",
        verified_commit="c1",
        from_commit="c1",
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "resume",
            "status": "running",
            "operator": "o",
            "registered_at": release.now_iso(),
        },
    )
    assert release.tick([]) == 2  # applying-interrupted
    request = json.loads(
        (base / "var/lib/pareton-deploy/release-request.json").read_text()
    )
    assert request["status"] == "failed"
    assert read_state(base)["phase"] == "applying"  # materials kept
    assert release.cmd_request(["rollback", "--operator", "o"]) == 0


def test_unpause_always_resumes_automatic_deploys(base):
    # R2-6: the bootstrap snapshot records a deliberately disabled deploy
    # timer; unpause must re-enable it regardless of that snapshot.
    write_state(
        base,
        phase="idle",
        hold={
            "reason": "bootstrap",
            "operator": "o",
            "at": release.now_iso(),
            "baseline_commit": "c1",
        },
        original_units={DEPLOY_TIMER_TEST: {"active": False, "enabled": False}},
    )
    release.write_json_atomic(
        base / "var/lib/pareton-deploy/release-request.json",
        {
            "type": "unpause",
            "status": "pending",
            "operator": "o",
            "main_commit": "c2",
            "registered_at": release.now_iso(),
        },
    )
    release.run_cmd.git_refs = {"origin/main": "c2", "HEAD": "c1"}
    assert release.tick([]) == 0
    enabled = [
        argv
        for argv, kw in zip(release.run_cmd.calls, release.run_cmd.kwargs)
        if "enable" in argv and "--now" in argv
    ]
    assert any("pareton-deploy.timer" in argv for argv in enabled)
    assert read_state(base)["hold"] is None


DEPLOY_TIMER_TEST = "pareton-deploy.timer"
