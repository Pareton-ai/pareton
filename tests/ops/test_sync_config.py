"""Offline tests for ops/sync-config.py (spec sections 5.1-5.3, A2-A6, A13-A15).

Everything runs against a temporary PARETON_SYNC_BASE prefix with fake
systemctl/systemd-analyze/vector binaries on PATH. systemd- and
Vector-specific behaviors that need a real Linux host are covered by the
isolated-environment acceptance matrix instead (spec section 9).
"""

import importlib.util
import json
import os
import shutil
import stat
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


sync = load_ops_module("sync-config")


FAKE_SYSTEMCTL = """#!/bin/sh
echo "$@" >> "$SYSTEMCTL_LOG"
joined="$1${2:+ $2}"
oldIFS=$IFS
IFS=,
for rule in $FAKE_SYSTEMCTL_FAIL; do
  [ "$joined" = "$rule" ] && { IFS=$oldIFS; exit 1; }
done
IFS=$oldIFS
if [ "$1" = is-active ]; then
  echo "${FAKE_IS_ACTIVE:-active}"
fi
exit 0
"""

FAKE_SYSTEMD_ANALYZE = """#!/bin/sh
echo "verify $@" >> "$SYSTEMCTL_LOG"
exit ${FAKE_VERIFY_RC:-0}
"""

FAKE_VECTOR = """#!/bin/sh
echo "vector $@" >> "$SYSTEMCTL_LOG"
for arg in "$@"; do
  case "$arg" in
    /*) if grep -q INVALID_MARKER "$arg" 2>/dev/null; then exit 1; fi ;;
  esac
done
exit ${FAKE_VECTOR_RC:-0}
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    base = tmp_path / "base"
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(OPS, repo / "ops")
    for sub in ("systemd", "gpu", "vector"):
        (repo / "ops" / sub).mkdir(exist_ok=True)

    monkeypatch.setenv("PARETON_SYNC_BASE", str(base))
    monkeypatch.setenv("PARETON_SYNC_EXPECTED_UID", str(os.getuid()))
    monkeypatch.setenv("PARETON_NOTIFY_BASE", str(base))
    monkeypatch.setenv("PARETON_NOTIFY_EXPECTED_UID", str(os.getuid()))
    # Stage-2 release coordination: write modes take the deploy mutex and
    # read the release state — remap both into the sandbox so the gate
    # exercises for real without touching /run or /var.
    monkeypatch.setenv("PARETON_DEPLOY_LOCK", str(tmp_path / "deploy.lock"))
    monkeypatch.setenv("PARETON_RELEASE_STATE", str(tmp_path / "release-state.json"))
    monkeypatch.setenv("PARETON_ACTIVITY_LOCK", str(tmp_path / "activity.lock"))

    env_file = base / "opt/pareton/.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "PARETON_DISCORD_DEPLOY_WEBHOOK=https://discord.com/api/webhooks/test\n"
        "PARETON_AXIOM_TOKEN=axiom-token\n"
    )
    env_file.chmod(0o600)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "systemctl.log"
    for name, body in {
        "systemctl": FAKE_SYSTEMCTL,
        "systemd-analyze": FAKE_SYSTEMD_ANALYZE,
        "vector": FAKE_VECTOR,
    }.items():
        tool = bin_dir / name
        tool.write_text(body)
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("SYSTEMCTL_LOG", str(log))

    class Env:
        def __init__(self):
            self.base = base
            self.repo = repo
            self.log = log

        def target(self, absolute):
            return base / absolute.lstrip("/")

        def calls(self):
            return [line for line in self.log.read_text().splitlines() if line]

    return Env()


def run_mode(env, *mode_args):
    """Run sync-config.main capturing its JSON line and exit code."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = sync.main(["--repo", str(env.repo), "--source", "worktree", *mode_args])
    line = out.getvalue().strip().splitlines()
    payload = json.loads(line[-1]) if line else {}
    return code, payload


def install_clean_state(env):
    """First apply from nothing, as bootstrap would."""
    code, _ = run_mode(env, "deploy-hook")
    assert code == 0
    # Re-chown expectations are handled by PARETON_SYNC_EXPECTED_UID.
    return code


def write_unit(env, name, text, mode=0o644):
    path = env.target(f"/etc/systemd/system/{name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def unit_text(name):
    return (OPS / "systemd" / name).read_text()


# ---------------------------------------------------------------- check


def test_check_reports_missing_until_installed(env):
    code, payload = run_mode(env, "check")
    assert code == 1
    categories = {f["category"] for f in payload["findings"]}
    assert "missing" in categories
    assert "unexpected" not in categories


def test_deploy_hook_bootstraps_and_second_run_is_noop(env):
    code, payload = run_mode(env, "deploy-hook")
    assert code == 0
    assert payload["action"] == "applied"
    # daemon-reload and vector restart happened exactly once each.
    calls = env.calls()
    assert calls.count("daemon-reload") == 1
    assert "restart vector" in calls
    assert any(c.startswith("vector validate") for c in calls)

    env.log.write_text("")
    code, payload = run_mode(env, "deploy-hook")
    assert code == 0
    assert payload["action"] == "none"
    assert env.calls() == []  # A3: no rewrite, no extra reload/restart


def test_managed_drift_is_reconverged(env):
    install_clean_state(env)
    drifted = unit_text("pareton-api.service").replace("pareton", "paretonx")
    write_unit(env, "pareton-api.service", drifted)
    code, _payload = run_mode(env, "deploy-hook")
    assert code == 0
    assert env.target("/etc/systemd/system/pareton-api.service").read_text() == (
        unit_text("pareton-api.service")
    )
    assert "daemon-reload" in env.calls()


def test_perms_drift_reported_and_fixed(env):
    install_clean_state(env)
    target = env.target("/etc/systemd/system/pareton-api.service")
    target.chmod(0o600)
    code, payload = run_mode(env, "check")
    assert code == 1
    assert any(f["category"] == "perms" for f in payload["findings"])
    code, _ = run_mode(env, "deploy-hook")
    assert code == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


# ---------------------------------------------------------------- blocking


def test_unmanaged_drop_in_blocks_and_protects_other_targets(env):
    install_clean_state(env)
    api = env.target("/etc/systemd/system/pareton-api.service")
    api.write_text("garbage")
    write_unit(env, "pareton-worker.service.d/unknown.conf", "[Service]\nNice=1\n")
    code, payload = run_mode(env, "deploy-hook")
    assert code == 3
    assert payload["error"] == "blocked"
    # A4: not even the managed drift gets rewritten in the same run.
    assert api.read_text() == "garbage"


def test_masked_unit_blocks(env):
    install_clean_state(env)
    api = env.target("/etc/systemd/system/pareton-api.service")
    api.unlink()
    api.symlink_to("/dev/null")
    code, payload = run_mode(env, "check")
    assert code == 3
    assert any(f["category"] == "masked" for f in payload["findings"])


def test_runtime_override_blocks(env):
    install_clean_state(env)
    override = env.target("/run/systemd/system/pareton-worker.service.d/x.conf")
    override.parent.mkdir(parents=True)
    override.write_text("[Service]\n")
    code, payload = run_mode(env, "check")
    assert code == 3
    assert any(f["category"] == "override" for f in payload["findings"])


def test_missing_onfailure_converges_like_any_drift(env):
    # Production state: a live deploy unit WITHOUT the OnFailure line must not
    # block the apply that installs it (Cursor review: the standalone check
    # deadlocked bootstrap). Content equality is the guarantee (spec 5.1).
    install_clean_state(env)
    deploy = env.target("/etc/systemd/system/pareton-deploy.service")
    stripped = deploy.read_text().replace(
        "OnFailure=pareton-deploy-failed.service\n", ""
    )
    deploy.write_text(stripped)
    code, payload = run_mode(env, "deploy-hook")
    assert code == 0
    assert payload["action"] == "applied"
    assert "OnFailure=pareton-deploy-failed.service" in deploy.read_text()


# ---------------------------------------------------------------- validation


def test_invalid_vector_config_keeps_original(env, monkeypatch):
    install_clean_state(env)
    toml = env.target("/etc/vector/vector.toml")
    original = toml.read_text()
    # The fake vector binary rejects candidates carrying INVALID_MARKER.
    (env.repo / "ops/vector/vector.toml").write_text(original + "\n# INVALID_MARKER\n")
    code, _payload = run_mode(env, "deploy-hook")
    assert code != 0
    assert toml.read_text() == original  # A5: candidate rejected, live kept


def test_vector_binary_failure_aborts(env, monkeypatch):
    install_clean_state(env)
    monkeypatch.setenv("FAKE_VECTOR_RC", "1")
    toml = env.target("/etc/vector/vector.toml")
    original = toml.read_text()
    (env.repo / "ops/vector/vector.toml").write_text(original + "\n# changed\n")
    code, _payload = run_mode(env, "deploy-hook")
    assert code == 2
    assert toml.read_text() == original


def test_missing_axiom_token_blocks_vector_apply(env):
    install_clean_state(env)
    (env.base / "opt/pareton/.env").write_text(
        "PARETON_DISCORD_DEPLOY_WEBHOOK=https://discord.com/api/webhooks/test\n"
    )
    toml = env.target("/etc/vector/vector.toml")
    original = toml.read_text()
    (env.repo / "ops/vector/vector.toml").write_text(original + "\n# changed\n")
    code, payload = run_mode(env, "deploy-hook")
    assert code == 3
    assert payload["error"] == "axiom-token-unavailable"
    assert toml.read_text() == original


# ---------------------------------------------------------------- rollback


def test_daemon_reload_failure_rolls_back_and_keeps_debt(env, monkeypatch):
    install_clean_state(env)
    api = env.target("/etc/systemd/system/pareton-api.service")
    original = api.read_text()
    monkeypatch.setenv("FAKE_SYSTEMCTL_FAIL", "daemon-reload")
    (env.repo / "ops/systemd/pareton-api.service").write_text(
        original + "# local edit\n"
    )
    code, payload = run_mode(env, "deploy-hook")
    assert code != 0
    assert api.read_text() == original  # A6: restored
    assert payload.get("rollback") or payload.get("installed") is not None

    monkeypatch.delenv("FAKE_SYSTEMCTL_FAIL")
    code, _ = run_mode(env, "deploy-hook")
    assert code == 0
    assert (
        env.target("/etc/systemd/system/pareton-api.service")
        .read_text()
        .endswith("# local edit\n")
    )


def test_vector_restart_failure_rolls_back(env, monkeypatch):
    install_clean_state(env)
    toml = env.target("/etc/vector/vector.toml")
    original = toml.read_text()
    monkeypatch.setenv("FAKE_SYSTEMCTL_FAIL", "restart vector")
    (env.repo / "ops/vector/vector.toml").write_text(original + "\n# changed\n")
    code, _ = run_mode(env, "deploy-hook")
    assert code != 0
    assert toml.read_text() == original
    # Cursor review: a failed restart must still run the recovery sequence and
    # record the debt for the next tick.
    calls = env.calls()
    assert "stop vector" in calls
    assert "reset-failed vector" in calls
    assert "start vector" in calls
    pending = json.loads(
        (env.base / "var/lib/pareton-deploy/sync-pending.json").read_text()
    )
    assert pending["vector_restart"] is True

    monkeypatch.delenv("FAKE_SYSTEMCTL_FAIL")
    code, _ = run_mode(env, "deploy-hook")
    assert code == 0
    pending = json.loads(
        (env.base / "var/lib/pareton-deploy/sync-pending.json").read_text()
    )
    assert pending["vector_restart"] is False


# ---------------------------------------------------------------- owed actions


def test_worker_change_sets_pending_flag_api_change_owes_restart(env):
    install_clean_state(env)
    queue = env.repo / "ops/systemd/pareton-worker.service.d/queue.conf"
    queue.write_text(queue.read_text() + "# tuned\n")
    api = env.repo / "ops/systemd/pareton-api.service"
    api.write_text(api.read_text() + "# tuned\n")
    code, _payload = run_mode(env, "deploy-hook")
    assert code == 0
    assert (env.repo / ".deploy-pending").exists()
    assert not (env.repo / ".deploy-rounds-pending").exists()

    code, _ = run_mode(env, "check")
    assert code == 0  # files converged; only runtime restart is owed
    out_code, _payload = run_mode(env, "deploy-hook")
    assert out_code == 0

    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        code = sync.main(
            ["--repo", str(env.repo), "--source", "worktree", "owed-restarts"]
        )
    assert code == 0
    assert "pareton-api.service" in out.getvalue().split()

    code, _ = run_mode(env, "clear-restarts", "pareton-api.service")
    assert code == 0
    out = io.StringIO()
    with redirect_stdout(out):
        sync.main(["--repo", str(env.repo), "--source", "worktree", "owed-restarts"])
    assert out.getvalue().strip() == ""


def test_vector_restart_keeps_toml_mode(env):
    install_clean_state(env)
    toml = env.target("/etc/vector/vector.toml")
    assert (
        stat.S_IMODE(toml.stat().st_mode) == 0o600
    )  # A13: no inline token, tight mode


# ---------------------------------------------------------------- mapping


def test_target_collision_is_an_error(env):
    (env.repo / "ops/gpu/pareton-api.service").write_text(
        (OPS / "gpu/pareton-gpu-reap.service").read_text()
    )
    code, payload = run_mode(env, "check")
    assert code == 2
    assert payload["error"].startswith("target-collision")


def test_missing_repo_source_is_incomplete(env):
    shutil.rmtree(env.repo / "ops")
    code, payload = run_mode(env, "check")
    assert code == 2
    assert payload["error"] == "cannot-list-repo-source"
