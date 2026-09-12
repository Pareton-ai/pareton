#!/usr/bin/env python3
"""Pareton deploy-failure notifier (spec: docs/第一阶段配置与部署告警-spec.md, section 7).

Started by pareton-deploy-failed.service through OnFailure from
pareton-deploy.service. Standard library only: it must not depend on the
application venv, the database, Vector, or Axiom.

Modes:
  notify-failure   deduplicated Discord alert for the failed deploy
  record-success   deploy success path clears the active fault
  validate-local   read-only credential/prerequisite check (no network, no
                   state access); also used by sync-config.py check
  test-send        explicit channel probe; bypasses and preserves dedup state

The webhook URL is read from PARETON_DISCORD_DEPLOY_WEBHOOK in
/opt/pareton/.env (owner-provided). The value never appears in output:
failures are reported as categories only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops_common import (
    locked,
    now_iso,
    parse_env_file,
    parse_iso,
    read_json,
    write_json_atomic,
)


def emit_structured(event: str, **fields) -> None:
    """Single-line JSON on stdout for journald/Vector (stage-2 spec 7.2).

    No secrets: identifiers and outcome categories only.
    """
    payload = {"event": event, **fields}
    print(json.dumps(payload, default=str, separators=(",", ":")), flush=True)


BASE_ENV = "PARETON_NOTIFY_BASE"
UID_ENV = "PARETON_NOTIFY_EXPECTED_UID"
WEBHOOK_VAR = "PARETON_DISCORD_DEPLOY_WEBHOOK"
SUPPRESS_SECONDS = 30 * 60
REQUEST_TIMEOUT = 20
MAX_ATTEMPTS = 2
TOTAL_BUDGET_SECONDS = 60
DEPLOY_UNIT = "pareton-deploy.service"


def p(absolute: str) -> Path:
    base = os.environ.get(BASE_ENV, "")
    return Path(base + absolute) if base else Path(absolute)


def expected_uid() -> int:
    return int(os.environ.get(UID_ENV, "0"))


def env_file_path() -> Path:
    return p("/opt/pareton/.env")


def state_path() -> Path:
    return p("/var/lib/pareton-deploy/alert-state.json")


def lock_path() -> Path:
    return p("/var/lib/pareton-deploy/alert-state.lock")


def run_env_path() -> Path:
    return p("/var/lib/pareton-deploy/last-run.env")


def load_webhook() -> tuple[str | None, str | None]:
    """Return (url, problem). The URL itself must never be logged."""
    values, problems = parse_env_file(env_file_path())
    if problems:
        return None, ",".join(problems)
    url = values.get(WEBHOOK_VAR, "")
    if not url:
        return None, "variable-empty"
    if not url.startswith("https://"):
        return None, "variable-not-https"
    return url, None


# --------------------------------------------------------------------------
# Transport (monkeypatched in tests)


def http_post_json(url: str, payload: dict, timeout: int) -> tuple[bool, str | None]:
    """POST the webhook; return (ok, message_id-or-error-category)."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{url}?wait=true", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode() or "{}")
            return True, str(data.get("id") or "")
    except urllib.error.HTTPError as exc:
        return False, f"http-{exc.code}"
    except urllib.error.URLError as exc:
        return False, type(exc.reason).__name__ if exc.reason else "URLError"
    except (ValueError, OSError) as exc:
        return False, type(exc).__name__


def send_with_retry(url: str, content: str) -> tuple[bool, str | None]:
    deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ok, detail = http_post_json(url, {"content": content}, REQUEST_TIMEOUT)
        if ok:
            return True, detail
        if attempt == MAX_ATTEMPTS or time.monotonic() + REQUEST_TIMEOUT > deadline:
            return False, detail
    return False, "retry-budget-exhausted"


# --------------------------------------------------------------------------
# Failure facts


def systemctl_show_deploy() -> dict:
    """Query the failed deploy unit's facts as Key=Value lines.

    Deliberately WITHOUT --value and WITHOUT any ordering assumption:
    systemd emits properties in its own internal order, not the requested
    one (owner-verified on the production box — positional parsing labeled
    the wrong value as the invocation ID, so every alert reported unknown
    facts). Map by key name instead.
    """
    facts: dict = {}
    try:
        result = subprocess.run(
            [
                "systemctl",
                "show",
                DEPLOY_UNIT,
                "--property=InvocationID,ExecMainStatus,Result",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return facts
    if result.returncode != 0:
        return facts
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in ("InvocationID", "ExecMainStatus", "Result"):
            facts[key] = value.strip()
    facts["invocation_id"] = facts.get("InvocationID", "")
    facts["exec_main_status"] = facts.get("ExecMainStatus", "")
    facts["result"] = facts.get("Result", "")
    return facts


def failure_facts() -> dict:
    facts = systemctl_show_deploy()
    run_values, _ = parse_env_file(run_env_path())
    matched = bool(
        run_values.get("invocation_id")
        and facts.get("invocation_id")
        and run_values["invocation_id"] == facts["invocation_id"]
    )
    status = facts.get("exec_main_status", "unknown")
    if status.lstrip("-").isdigit():
        exit_class = "nonzero" if status not in ("0", "") else "failed-result"
    else:
        exit_class = "exec-error"
    return {
        "host": platform.node(),
        "invocation_id": facts.get("invocation_id") or "unknown",
        "started_at": (run_values.get("started_at") if matched else None) or now_iso(),
        "from_commit": (run_values.get("from_commit") if matched else None)
        or "unknown",
        "target_commit": (run_values.get("target_commit") if matched else None)
        or "unknown",
        "step": (run_values.get("last_step") if matched else None) or "unknown",
        "exec_status": status or "unknown",
        "exit_class": exit_class,
        "facts_matched": matched,
    }


def fault_key(facts: dict) -> dict:
    return {
        "host": facts["host"],
        "target_commit": facts["target_commit"],
        "step": facts["step"],
        "exit_class": facts["exit_class"],
    }


def build_message(facts: dict, fault: dict) -> str:
    count = fault.get("count", 1)
    first = fault.get("first_seen", facts["started_at"])
    lines = [
        "🚨 pareton deploy failed",
        f"host: {facts['host']}",
        f"unit: {DEPLOY_UNIT} (exit {facts['exec_status']}, result {facts['exit_class']})",
        f"step: {facts['step']}",
        f"commits: {facts['from_commit'][:12]} -> {facts['target_commit'][:12]}",
        f"first seen: {first}, latest: {fault.get('last_seen', facts['started_at'])}, count: {count}",
        "logs: journalctl -u pareton-deploy.service | tail -50",
    ]
    return "\n".join(lines)


def default_state() -> dict:
    return {
        "fault": None,
        "last_success": None,
        "last_processed_invocation": None,
        "send_failures": 0,
    }


# --------------------------------------------------------------------------
# Modes


def cmd_notify_failure(args: argparse.Namespace) -> int:
    facts = failure_facts()
    key = fault_key(facts)
    url, problem = load_webhook()
    if url is None:
        print(f"notify: webhook unavailable ({problem})", file=sys.stderr)
        return 1

    with locked(lock_path()):
        state = read_json(state_path()) or default_state()
        fault = state.get("fault") or None

        last_success = state.get("last_success") or {}
        success_time = parse_iso(last_success.get("completed_at") or "")
        failure_time = parse_iso(facts["started_at"])
        if success_time and failure_time and success_time > failure_time:
            # A full deploy already succeeded after this failure started; a
            # late OnFailure callback must not resurrect the fault (7.3).
            write_json_atomic(state_path(), state)
            emit_structured(
                "deploy_failure_notified",
                invocation_id=facts["invocation_id"],
                message_id="",
                outcome="skipped-recovered",
                step=facts["step"],
                host=facts["host"],
            )
            print("notify: failure already recovered by a later success; skipped")
            return 0

        same = bool(fault and fault.get("key") == key)
        now = now_iso()
        if same:
            fault["count"] = int(fault.get("count", 1)) + 1
            fault["last_seen"] = now
        else:
            fault = {
                "key": key,
                "first_seen": now,
                "last_seen": now,
                "count": 1,
                "last_notified": None,
                "last_message_id": None,
            }

        last_notified = parse_iso(fault.get("last_notified") or "")
        now_dt = parse_iso(now)
        if (
            same
            and last_notified
            and now_dt
            and (now_dt - last_notified).total_seconds() < SUPPRESS_SECONDS
        ):
            state["fault"] = fault
            write_json_atomic(state_path(), state)
            emit_structured(
                "deploy_failure_notified",
                invocation_id=facts["invocation_id"],
                message_id="",
                outcome="suppressed",
                step=facts["step"],
                host=facts["host"],
                count=fault["count"],
            )
            print(f"notify: suppressed (same fault, count {fault['count']})")
            return 0

        ok, detail = send_with_retry(url, build_message(facts, fault))
        if ok:
            fault["last_notified"] = now
            fault["last_message_id"] = detail or ""
            state["fault"] = fault
            state["last_processed_invocation"] = facts["invocation_id"]
            write_json_atomic(state_path(), state)
            emit_structured(
                "deploy_failure_notified",
                invocation_id=facts["invocation_id"],
                message_id=detail or "",
                outcome="sent",
                step=facts["step"],
                host=facts["host"],
            )
            print(f"notify: sent (message id {detail or 'n/a'})")
            return 0
        state["fault"] = fault  # Count grows, suppression window stays closed.
        state["send_failures"] = int(state.get("send_failures", 0)) + 1
        write_json_atomic(state_path(), state)
        emit_structured(
            "deploy_failure_notified",
            invocation_id=facts["invocation_id"],
            message_id="",
            outcome=f"send-failed:{detail}",
            step=facts["step"],
            host=facts["host"],
        )
        print(f"notify: send failed ({detail})", file=sys.stderr)
        return 1


def cmd_record_success(args: argparse.Namespace) -> int:
    with locked(lock_path()):
        state = read_json(state_path()) or default_state()
        state["fault"] = None
        state["last_success"] = {
            "invocation": args.invocation,
            "completed_at": now_iso(),
            "from": args.from_commit,
            "to": args.to_commit,
        }
        state["last_processed_invocation"] = args.invocation
        write_json_atomic(state_path(), state)
        print("notify: success recorded, active fault cleared")
        return 0


def cmd_validate_local(args: argparse.Namespace) -> int:
    payload = {"mode": "validate-local"}
    path = env_file_path()
    if not path.exists():
        payload.update(status="broken", problems=["env-file-missing"])
        print(json.dumps(payload))
        return 1
    url, load_problem = load_webhook()
    if load_problem == "permission-denied":
        # Non-root callers cannot read a correctly locked-down env file;
        # that is "cannot verify", not "broken" (spec 2.3 review note).
        payload.update(status="unverifiable", problems=["permission-denied"])
        print(json.dumps(payload))
        return 2
    info = path.stat()
    problems = []
    if info.st_mode & 0o777 != 0o600:
        problems.append(f"env-mode-{oct(info.st_mode & 0o777)}")
    if info.st_uid != expected_uid():
        problems.append(f"env-owner-{info.st_uid}")
    if url is None:
        problems.append(f"webhook-{load_problem}")
    if problems:
        payload.update(status="broken", problems=problems)
        print(json.dumps(payload))
        return 1
    payload.update(status="ok", problems=[])
    print(json.dumps(payload))
    return 0


def cmd_test_send(args: argparse.Namespace) -> int:
    url, problem = load_webhook()
    if url is None:
        print(f"test-send: webhook unavailable ({problem})", file=sys.stderr)
        return 1
    note = f" {args.note}" if args.note else ""
    ok, detail = send_with_retry(
        url, f"[test] pareton deploy alert channel probe{note} ({now_iso()})"
    )
    if not ok:
        print(f"test-send: failed ({detail})", file=sys.stderr)
        return 1
    print(f"test-send: delivered (message id {detail or 'n/a'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("notify-failure")
    record = sub.add_parser("record-success")
    record.add_argument("--invocation", required=True)
    record.add_argument("--from", dest="from_commit", default=None)
    record.add_argument("--to", dest="to_commit", default=None)
    validate = sub.add_parser("validate-local")
    validate.add_argument("--json", dest="as_json", action="store_true")
    test = sub.add_parser("test-send")
    test.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    if args.mode == "notify-failure":
        return cmd_notify_failure(args)
    if args.mode == "record-success":
        return cmd_record_success(args)
    if args.mode == "validate-local":
        return cmd_validate_local(args)
    return cmd_test_send(args)


if __name__ == "__main__":
    sys.exit(main())
