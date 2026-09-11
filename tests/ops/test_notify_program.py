"""Offline tests for ops/notify-deploy-failure.py (spec sections 7.1-7.4, A16-A18).

HTTP transport is monkeypatched; the clock is driven through
PARETON_TEST_NOW. The webhook value must never reach stdout/stderr even on
failure paths.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_notifier():
    spec = importlib.util.spec_from_file_location(
        "notify_deploy_failure", REPO_ROOT / "ops" / "notify-deploy-failure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def notifier(tmp_path, monkeypatch):
    module = load_notifier()
    base = tmp_path / "base"
    monkeypatch.setenv("PARETON_NOTIFY_BASE", str(base))
    monkeypatch.setenv("PARETON_NOTIFY_EXPECTED_UID", str(os.getuid()))
    monkeypatch.setenv("PARETON_TEST_NOW", "2026-09-10T12:00:00Z")
    sent: list[dict] = []

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Key=Value lines in a NON-requested order, exactly like real systemd
    # (owner-verified on prod): positional parsing silently mislabels values.
    (bin_dir / "systemctl").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = show ]; then\n'
        '  printf "Result=failed\\nExecMainStatus=1\\nInvocationID=inv-1\\n"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    (bin_dir / "systemctl").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    def fake_post(url, payload, timeout):
        if os.environ.get("FAKE_HTTP_FAIL") == "1":
            return False, "http-429"
        sent.append({"url": url, "payload": payload})
        return True, "111222333"

    monkeypatch.setattr(module, "http_post_json", fake_post)
    module._test_sent = sent

    def set_now(iso):
        monkeypatch.setenv("PARETON_TEST_NOW", iso)

    def write_run_env(
        step="restart-app",
        invocation="inv-1",
        started="2026-09-10T11:59:00Z",
        target="abc123",
        from_="000000",
    ):
        run = base / "var/lib/pareton-deploy/last-run.env"
        run.parent.mkdir(parents=True, exist_ok=True)
        run.write_text(
            f"invocation_id={invocation}\nstarted_at={started}\n"
            f"from_commit={from_}\ntarget_commit={target}\nlast_step={step}\n"
        )

    def write_env_file(
        mode=0o600, webhook="https://discord.com/api/webhooks/SECRETVALUE", extra=""
    ):
        env_file = base / "opt/pareton/.env"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(f"PARETON_DISCORD_DEPLOY_WEBHOOK={webhook}\n{extra}\n")
        env_file.chmod(mode)

    module._set_now = set_now
    module._write_run_env = write_run_env
    module._write_env_file = write_env_file
    return module


def run_mode(module, *args):
    import io
    from contextlib import redirect_stderr, redirect_stdout

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = module.main(list(args))
    return code, out.getvalue(), err.getvalue()


def state(module):
    path = module.p("/var/lib/pareton-deploy/alert-state.json")
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------- validate-local


def test_validate_local_ok(notifier):
    notifier._write_env_file()
    code, out, _ = run_mode(notifier, "validate-local")
    payload = json.loads(out.strip().splitlines()[-1])
    assert code == 0
    assert payload["status"] == "ok"


def test_validate_local_reports_missing_and_empty(notifier, tmp_path):
    code, out, _ = run_mode(notifier, "validate-local")
    payload = json.loads(out.strip().splitlines()[-1])
    assert code == 1
    assert payload["status"] == "broken"

    notifier._write_env_file(webhook="")
    code, out, _ = run_mode(notifier, "validate-local")
    payload = json.loads(out.strip().splitlines()[-1])
    assert code == 1
    assert "webhook-variable-empty" in payload["problems"]


def test_validate_local_reports_bad_mode(notifier):
    notifier._write_env_file(mode=0o644)
    code, out, _ = run_mode(notifier, "validate-local")
    payload = json.loads(out.strip().splitlines()[-1])
    assert code == 1
    assert any(p.startswith("env-mode-") for p in payload["problems"])


def test_validate_local_unreadable_is_unverifiable(notifier):
    notifier._write_env_file()
    env_file = notifier.p("/opt/pareton/.env")
    env_file.chmod(0o000)
    code, out, _ = run_mode(notifier, "validate-local")
    payload = json.loads(out.strip().splitlines()[-1])
    assert code == 2
    assert payload["status"] == "unverifiable"


# --------------------------------------------------------------- notify-failure


def test_first_failure_sends_with_required_fields(notifier):
    notifier._write_env_file()
    notifier._write_run_env(step="install-config")
    code, _out, _err = run_mode(notifier, "notify-failure")
    assert code == 0
    assert len(notifier._test_sent) == 1
    message = notifier._test_sent[0]["payload"]["content"]
    for field in (
        "host:",
        "unit: pareton-deploy.service",
        "step: install-config",
        "count: 1",
    ):
        assert field in message
    # The invocation must match despite systemd's own property order, so the
    # run's facts resolve and the fault key is specific (owner-verified bug).
    assert "000000 -> abc123" in message
    fault = state(notifier)["fault"]
    assert fault["count"] == 1
    assert fault["key"]["step"] == "install-config"
    assert fault["key"]["target_commit"] == "abc123"


def test_facts_unmatched_reports_unknown_without_breaking(notifier):
    notifier._write_env_file()
    # A last-run.env from a DIFFERENT run (e.g. the next tick already ran).
    notifier._write_run_env(step="fetch", invocation="someone-else")
    code, _out, _err = run_mode(notifier, "notify-failure")
    assert code == 0
    message = notifier._test_sent[0]["payload"]["content"]
    assert "step: unknown" in message
    assert state(notifier)["fault"]["key"]["step"] == "unknown"


def test_same_fault_suppressed_then_reminded_after_window(notifier, monkeypatch):
    notifier._write_env_file()
    notifier._write_run_env(step="fetch")
    assert run_mode(notifier, "notify-failure")[0] == 0

    notifier._set_now("2026-09-10T12:10:00Z")  # +10 min: suppressed
    code, _out, _ = run_mode(notifier, "notify-failure")
    assert code == 0
    assert len(notifier._test_sent) == 1
    assert state(notifier)["fault"]["count"] == 2

    notifier._set_now("2026-09-10T12:31:00Z")  # past the 30 min window
    code, _out, _ = run_mode(notifier, "notify-failure")
    assert code == 0
    assert len(notifier._test_sent) == 2  # reminder went out
    assert state(notifier)["fault"]["count"] == 3


def test_fault_key_change_sends_immediately(notifier):
    notifier._write_env_file()
    notifier._write_run_env(step="fetch")
    run_mode(notifier, "notify-failure")
    notifier._set_now("2026-09-10T12:01:00Z")
    notifier._write_run_env(step="restart-app")
    code, _, _ = run_mode(notifier, "notify-failure")
    assert code == 0
    assert len(notifier._test_sent) == 2


def test_send_failure_does_not_open_suppression_window(notifier, monkeypatch):
    notifier._write_env_file()
    notifier._write_run_env(step="fetch")
    monkeypatch.setenv("FAKE_HTTP_FAIL", "1")
    code, _out, _err = run_mode(notifier, "notify-failure")
    assert code == 1
    assert state(notifier)["fault"]["last_notified"] is None

    monkeypatch.delenv("FAKE_HTTP_FAIL")
    notifier._set_now("2026-09-10T12:01:00Z")
    code, _, _ = run_mode(notifier, "notify-failure")
    assert code == 0
    assert len(notifier._test_sent) == 1  # retried send went out, not suppressed


def test_webhook_value_never_reaches_output(notifier, monkeypatch):
    notifier._write_env_file()
    notifier._write_run_env(step="fetch")
    monkeypatch.setenv("FAKE_HTTP_FAIL", "1")
    _code, out, err = run_mode(notifier, "notify-failure")
    assert "SECRETVALUE" not in out and "SECRETVALUE" not in err

    _code, out, err = run_mode(notifier, "validate-local")
    assert "SECRETVALUE" not in out and "SECRETVALUE" not in err

    monkeypatch.delenv("FAKE_HTTP_FAIL")
    notifier._write_run_env(step="restart-vector")
    _code, out, err = run_mode(notifier, "test-send")
    assert "SECRETVALUE" not in out and "SECRETVALUE" not in err


# --------------------------------------------------------------- success / recovery


def test_record_success_clears_fault(notifier):
    notifier._write_env_file()
    notifier._write_run_env(step="fetch")
    run_mode(notifier, "notify-failure")
    assert state(notifier)["fault"] is not None
    code, _, _ = run_mode(
        notifier,
        "record-success",
        "--invocation",
        "inv-2",
        "--from",
        "000",
        "--to",
        "abc",
    )
    assert code == 0
    assert state(notifier)["fault"] is None
    assert state(notifier)["last_success"]["invocation"] == "inv-2"


def test_stale_failure_callback_does_not_resurrect(notifier):
    notifier._write_env_file()
    notifier._write_run_env(step="fetch")
    run_mode(notifier, "notify-failure")
    # Deploy for inv-2 started later and fully succeeded at 12:05.
    code, _, _ = run_mode(notifier, "record-success", "--invocation", "inv-2")
    assert code == 0

    notifier._set_now("2026-09-10T12:06:00Z")
    # Late OnFailure callback whose run started at 11:59 (before the success).
    code, out, _ = run_mode(notifier, "notify-failure")
    assert code == 0
    assert "recovered" in out
    assert state(notifier)["fault"] is None
    assert len(notifier._test_sent) == 1  # only the original alert; stale sent nothing


def test_recovered_fault_notifies_again_on_recurrence(notifier):
    notifier._write_env_file()
    notifier._write_run_env(step="fetch")
    run_mode(notifier, "notify-failure")
    run_mode(notifier, "record-success", "--invocation", "inv-2")
    notifier._set_now("2026-09-10T12:06:00Z")
    notifier._write_run_env(
        step="fetch", invocation="inv-3", started="2026-09-10T12:05:30Z"
    )
    code, _, _ = run_mode(notifier, "notify-failure")
    assert code == 0
    assert len(notifier._test_sent) == 2


def test_test_send_bypasses_and_preserves_state(notifier):
    notifier._write_env_file()
    notifier._write_run_env(step="fetch")
    run_mode(notifier, "notify-failure")  # active fault, notified at 12:00
    notifier._set_now("2026-09-10T12:01:00Z")
    code, _, _ = run_mode(notifier, "test-send", "--note", "channel-check")
    assert code == 0
    body = notifier._test_sent[-1]["payload"]["content"]
    assert body.startswith("[test]") and "channel-check" in body
    # The dedup window was not consumed by the probe.
    assert state(notifier)["fault"]["last_notified"] == "2026-09-10T12:00:00Z"


def test_notify_failure_without_webhook_fails_cleanly(notifier):
    notifier._write_run_env()
    code, _out, err = run_mode(notifier, "notify-failure")
    assert code == 1
    assert "webhook unavailable" in err
    assert state(notifier) is None  # nothing written
