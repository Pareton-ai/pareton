#!/usr/bin/env python3
"""Pareton stage-1 config synchronizer (spec: docs/第一阶段配置与部署告警-spec.md).

Compares the managed ops files (systemd units and drop-ins, Vector config,
deploy script, ops programs) between a selected Git commit and the live
filesystem, and installs verified candidates atomically. Standard library
only: it must run from the system Python without the application venv.

Modes (see sections 5.1-5.3 of the spec):
  check           read-only comparison; exit 0 clean, 1 drift, 2 incomplete,
                  3 blocked (unknown files, masks, credential prerequisites);
                  apply failures exit 4 (validation or effectuation failed)
  apply           validate candidates, install atomically, reload/restart
  deploy-hook     the per-tick decision table: auto-apply managed drift,
                  refuse blocked states, retry owed follow-up actions
  owed-restarts   print app units that still owe a restart after apply
  clear-restarts  drop restart debts after the deploy script performed them

Test/isolation usage: PARETON_SYNC_BASE remaps every absolute target path
under a prefix and --source worktree reads the repo working tree instead of
a Git commit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops_common import now_iso, parse_env_file, read_json, write_json_atomic

UNIT_MODE = 0o644
EXEC_MODE = 0o755
# Spec section 4.4: vector.service runs as root, so the TOML is root:root 0600
# once the inline token is replaced by the env reference. Adjust together with
# the vector unit if the service user ever changes.
VECTOR_TOML_MODE = 0o600

WORKER_PENDING = {
    "pareton-worker.service": ".deploy-pending",
    "pareton-round-worker.service": ".deploy-rounds-pending",
}
APP_UNITS = (
    "pareton-api.service",
    "pareton-watcher.service",
    "pareton-weights.service",
)
DEPLOY_UNIT = "pareton-deploy.service"
DEPLOY_FAILED_UNIT = "pareton-deploy-failed.service"

# Categories that must block any install attempt (spec section 5.3, row 4).
BLOCKED_CATEGORIES = {
    "unexpected",
    "masked",
    "override",
    "notify_prereq",
    "notify_unverifiable",
}
# Categories that plain file convergence can fix (spec section 5.3, row 3).
DRIFT_CATEGORIES = {"missing", "different", "perms"}


def p(absolute: str) -> Path:
    """Remap an absolute target path under the test base when set."""
    base = os.environ.get("PARETON_SYNC_BASE", "")
    return Path(base + absolute) if base else Path(absolute)


def expected_uid() -> int:
    return int(os.environ.get("PARETON_SYNC_EXPECTED_UID", "0"))


class Fail(Exception):
    """Controlled failure with an exit code and a JSON-safe reason."""

    def __init__(self, code: int, reason: str, **extra):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.extra = extra


# --------------------------------------------------------------------------
# Repo source access


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


def list_repo_ops_files(repo: Path, ref: str, use_git: bool) -> list[str] | None:
    """Relative ops/ paths provided by the selected source (None = error)."""
    wanted = ("ops/systemd/", "ops/gpu/", "ops/vector/")
    explicit = (
        "ops/deploy.sh",
        "ops/sync-config.py",
        "ops/notify-deploy-failure.py",
        "ops/ops_common.py",
    )
    files: list[str] = []
    if use_git:
        result = git(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            ref,
            "ops/systemd",
            "ops/gpu",
            "ops/vector",
        )
        if result.returncode != 0:
            return None
        files = [line for line in result.stdout.splitlines() if line]
        for rel in explicit:
            if git(repo, "cat-file", "-e", f"{ref}:{rel}").returncode == 0:
                files.append(rel)
    else:
        if not repo.is_dir():
            return None
        for sub in wanted:
            area = repo / sub.rstrip("/")
            if area.is_dir():
                files.extend(
                    str(f.relative_to(repo)) for f in area.rglob("*") if f.is_file()
                )
        for rel in explicit:
            if (repo / rel).is_file():
                files.append(rel)
    if not files:
        # An ops tree with nothing managed is a broken source, not an empty
        # configuration (spec 2.3: never treat missing config as empty).
        return None
    return files


def source_bytes(repo: Path, ref: str, use_git: bool, rel: str) -> bytes | None:
    if use_git:
        result = git(repo, "show", f"{ref}:{rel}")
        return result.stdout.encode() if result.returncode == 0 else None
    try:
        return (repo / rel).read_bytes()
    except OSError:
        return None


# --------------------------------------------------------------------------
# Mapping


class Entry:
    def __init__(self, rel: str, target: str, mode: int):
        self.rel = rel
        self.target = target
        self.mode = mode


def build_mapping(
    repo: Path, ref: str, use_git: bool
) -> tuple[list[Entry] | None, str | None]:
    files = list_repo_ops_files(repo, ref, use_git)
    if files is None:
        return None, "cannot-list-repo-source"
    entries: list[Entry] = []
    for rel in sorted(files):
        name = Path(rel).name
        if rel.startswith("ops/systemd/") and rel.endswith((".service", ".timer")):
            entries.append(Entry(rel, f"/etc/systemd/system/{name}", UNIT_MODE))
        elif re.fullmatch(r"ops/systemd/[^\s/]+\.service\.d/[^/\s]+\.conf", rel):
            unit_dir = Path(rel).parent.name
            entries.append(
                Entry(rel, f"/etc/systemd/system/{unit_dir}/{name}", UNIT_MODE)
            )
        elif rel.startswith("ops/gpu/") and rel.endswith((".service", ".timer")):
            entries.append(Entry(rel, f"/etc/systemd/system/{name}", UNIT_MODE))
        elif rel == "ops/vector/vector.service":
            entries.append(Entry(rel, "/etc/systemd/system/vector.service", UNIT_MODE))
        elif rel == "ops/vector/vector.toml":
            entries.append(Entry(rel, "/etc/vector/vector.toml", VECTOR_TOML_MODE))
        elif rel == "ops/deploy.sh":
            entries.append(Entry(rel, "/usr/local/bin/pareton-deploy", EXEC_MODE))
        elif rel in (
            "ops/sync-config.py",
            "ops/notify-deploy-failure.py",
            "ops/ops_common.py",
        ):
            entries.append(Entry(rel, f"/usr/local/lib/pareton-ops/{name}", EXEC_MODE))
    targets: dict[str, str] = {}
    for entry in entries:
        if entry.target in targets:
            return None, f"target-collision:{entry.target}"
        targets[entry.target] = entry.rel
    return entries, None


# --------------------------------------------------------------------------
# Check


def stat_problem(path: Path, mode: int) -> str | None:
    try:
        info = path.lstat()
    except OSError:
        return "unreadable"
    if info.st_mode & 511 != mode:
        return f"mode-{oct(info.st_mode & 0o777)}"
    if info.st_uid != expected_uid():
        return f"owner-{info.st_uid}"
    return None


def notify_program_path(
    args: argparse.Namespace, entries: list[Entry]
) -> tuple[Path | None, bool]:
    """Installed notifier when present, else the repo candidate (bootstrap).

    The candidate is a fresh checkout file (mode 0644), so callers must run it
    through the interpreter rather than executing it directly.
    """
    if args.notify_program:
        return Path(args.notify_program), False
    installed = p("/usr/local/lib/pareton-ops/notify-deploy-failure.py")
    if installed.is_file():
        return installed, False
    candidate = Path(args.repo) / "ops" / "notify-deploy-failure.py"
    if candidate.is_file():
        return candidate, True
    return None, False


def check_notify_prereq(args: argparse.Namespace, entries: list[Entry]) -> list[dict]:
    program, is_candidate = notify_program_path(args, entries)
    if program is None:
        return [
            {
                "category": "notify_prereq",
                "target": "/usr/local/lib/pareton-ops/notify-deploy-failure.py",
                "detail": "notifier-missing",
            }
        ]
    result = subprocess.run(
        [sys.executable, str(program), "validate-local", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ},
    )
    try:
        payload = json.loads(result.stdout)
        status = payload.get("status")
        problems = payload.get("problems") or []
    except ValueError:
        return [
            {
                "category": "error",
                "target": "notify-validate-local",
                "detail": "bad-output",
            }
        ]
    if status == "ok":
        return []
    category = "notify_prereq" if status == "broken" else "notify_unverifiable"
    detail = ",".join(problems) if problems else str(status)
    if is_candidate:
        detail = f"candidate:{detail}"
    return [{"category": category, "target": str(program), "detail": detail}]


def check_deploy_onfailure(entries: list[Entry]) -> list[dict]:
    """Explicit section 5.1 check: installed deploy unit must chain OnFailure."""
    target = p(f"/etc/systemd/system/{DEPLOY_UNIT}")
    try:
        text = target.read_text()
    except OSError:
        return []  # Missing file is already reported through the mapping.
    if re.search(rf"^OnFailure=.*{re.escape(DEPLOY_FAILED_UNIT)}", text, re.MULTILINE):
        return []
    return [
        {
            "category": "notify_prereq",
            "target": str(target),
            "detail": f"OnFailure-{DEPLOY_FAILED_UNIT}-missing",
        }
    ]


def run_check(args: argparse.Namespace) -> tuple[list[dict], list[Entry]]:
    entries, mapping_error = build_mapping(args.repo, args.ref, args.source == "git")
    if entries is None:
        raise Fail(2, mapping_error or "mapping-error")
    findings: list[dict] = []
    for entry in entries:
        content = source_bytes(args.repo, args.ref, args.source == "git", entry.rel)
        if content is None:
            findings.append(
                {
                    "category": "error",
                    "target": entry.target,
                    "detail": f"source-missing:{entry.rel}",
                }
            )
            continue
        target = p(entry.target)
        if not target.exists():
            findings.append(
                {"category": "missing", "target": entry.target, "source": entry.rel}
            )
            continue
        if target.read_bytes() != content:
            findings.append(
                {"category": "different", "target": entry.target, "source": entry.rel}
            )
        problem = stat_problem(target, entry.mode)
        if problem and problem != "unreadable":
            findings.append(
                {"category": "perms", "target": entry.target, "detail": problem}
            )
        elif problem == "unreadable":
            findings.append(
                {"category": "error", "target": entry.target, "detail": problem}
            )

    findings.extend(scan_unexpected(entries))
    findings.extend(check_deploy_onfailure(entries))
    findings.extend(check_notify_prereq(args, entries))
    return findings, entries


def scan_unexpected(entries: list[Entry]) -> list[dict]:
    findings: list[dict] = []
    managed = {entry.target for entry in entries}
    etc = p("/etc/systemd/system")
    try:
        present = sorted(
            str(f.relative_to(etc))
            for f in etc.iterdir()
            if (f.name.startswith("pareton-") or f.name == "vector.service")
            and not f.is_dir()
        )
        drop_dirs = sorted(
            f for f in etc.iterdir() if f.is_dir() and f.name.endswith(".d")
        )
    except FileNotFoundError:
        return []  # Nothing installed yet (fresh bootstrap); mapping covers the rest.
    except OSError:
        return [{"category": "error", "target": str(etc), "detail": "cannot-enumerate"}]
    for rel in present:
        absolute = f"/etc/systemd/system/{rel}"
        path = etc / rel
        if path.is_symlink() and os.readlink(path) == "/dev/null":
            findings.append(
                {"category": "masked", "target": absolute, "detail": "masked"}
            )
        elif absolute not in managed:
            findings.append(
                {"category": "unexpected", "target": absolute, "detail": "not-managed"}
            )
    # Drop-in directories: unmanaged *.conf files and /run overrides both count.
    for dropdir in drop_dirs:
        for conf in sorted(dropdir.glob("*.conf")):
            absolute = f"/etc/systemd/system/{dropdir.name}/{conf.name}"
            if absolute not in managed:
                findings.append(
                    {
                        "category": "unexpected",
                        "target": absolute,
                        "detail": "unmanaged-drop-in",
                    }
                )
        runtime = p(f"/run/systemd/system/{dropdir.name}")
        if runtime.is_dir() and any(runtime.iterdir()):
            findings.append(
                {
                    "category": "override",
                    "target": str(runtime),
                    "detail": "runtime-drop-in",
                }
            )
    return findings


def findings_summary(findings: list[dict]) -> str:
    if not findings:
        return "clean"
    parts = []
    for category in (
        "missing",
        "different",
        "perms",
        "unexpected",
        "masked",
        "override",
        "notify_prereq",
        "notify_unverifiable",
        "error",
    ):
        count = sum(1 for f in findings if f["category"] == category)
        if count:
            parts.append(f"{category}={count}")
    return " ".join(parts)


def exit_code_for(findings: list[dict]) -> int:
    if any(f["category"] == "error" for f in findings):
        return 2
    if any(f["category"] in BLOCKED_CATEGORIES for f in findings):
        return 3
    if findings:
        return 1
    return 0


# --------------------------------------------------------------------------
# Pending follow-up actions


def pending_path() -> Path:
    return p("/var/lib/pareton-deploy/sync-pending.json")


def load_pending() -> dict:
    data = read_json(pending_path()) or {}
    data.setdefault("owed_restart_units", [])
    data.setdefault("daemon_reload", False)
    data.setdefault("vector_restart", False)
    return data


def save_pending(data: dict) -> None:
    data["updated_at"] = now_iso()
    write_json_atomic(pending_path(), data)


# --------------------------------------------------------------------------
# Validation helpers


def run_cmd(argv: list[str], env: dict | None = None, timeout: int = 120):
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
        timeout=timeout,
    )


def validate_candidates(
    args: argparse.Namespace, plan: list[tuple[Entry, bytes]]
) -> None:
    if args.skip_validation:
        return
    with tempfile.TemporaryDirectory(prefix="pareton-sync-validate-") as tmp:
        stage = Path(tmp)
        units_staged = False
        for entry, content in plan:
            if entry.target.endswith(".toml"):
                (stage / "vector.toml").write_bytes(content)
            else:
                staged = stage / entry.target.lstrip("/")
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(content)
                units_staged = True
        if units_staged:
            if shutil.which("systemd-analyze") is None:
                raise Fail(2, "systemd-analyze-unavailable")
            result = run_cmd(["systemd-analyze", "verify", "--root", str(stage)])
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip().splitlines()
                raise Fail(2, "unit-validation-failed", validation=detail[:10])
        if (stage / "vector.toml").exists():
            if shutil.which("vector") is None:
                raise Fail(2, "vector-binary-unavailable")
            env_file = p(args.env_file)
            values, problems = parse_env_file(env_file)
            token = values.get("PARETON_AXIOM_TOKEN", "")
            if problems or not token:
                # Spec 4.4: never guess or fake credentials; fail before install.
                raise Fail(3, "axiom-token-unavailable", problems=problems or ["empty"])
            result = run_cmd(
                ["vector", "validate", str(stage / "vector.toml")],
                env={"PARETON_AXIOM_TOKEN": token},
            )
            if result.returncode != 0:
                raise Fail(2, "vector-validation-failed")


# --------------------------------------------------------------------------
# Apply


def install_entry(entry: Entry, content: bytes, backup_dir: Path | None) -> dict:
    target = p(entry.target)
    record: dict = {"target": entry.target, "source": entry.rel, "action": "created"}
    if target.exists():
        record["action"] = "replaced"
        if backup_dir is not None:
            slot = backup_dir / entry.target.lstrip("/").replace("/", "_")
            slot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, slot)
            record["backup"] = str(slot)
    from ops_common import atomic_write

    atomic_write(target, content, entry.mode)
    return record


def units_changed(plan: list[tuple[Entry, bytes]]) -> list[str]:
    changed = []
    for entry, _ in plan:
        name = Path(entry.target).name
        if name.endswith((".service", ".timer")):
            changed.append(name)
        elif Path(entry.target).parent.name.endswith(".service.d"):
            changed.append(Path(entry.target).parent.name[: -len(".d")])
    return sorted(set(changed))


def vector_changed(plan: list[tuple[Entry, bytes]]) -> bool:
    return any(
        entry.target
        in ("/etc/vector/vector.toml", "/etc/systemd/system/vector.service")
        for entry, _ in plan
    )


def daemon_reload() -> None:
    result = run_cmd(["systemctl", "daemon-reload"])
    if result.returncode != 0:
        raise Fail(4, "daemon-reload-failed")


def restart_vector() -> None:
    result = run_cmd(["systemctl", "restart", "vector"])
    if result.returncode != 0:
        raise Fail(4, "vector-restart-failed")
    active = run_cmd(["systemctl", "is-active", "vector"])
    if active.stdout.strip() != "active":
        raise Fail(4, "vector-not-active", detail=active.stdout.strip())


def restart_changed_timers(changed: list[str]) -> None:
    for unit in changed:
        if not unit.endswith(".timer"):
            continue
        active = run_cmd(["systemctl", "is-active", unit])
        if active.stdout.strip() == "active":
            result = run_cmd(["systemctl", "restart", unit])
            if result.returncode != 0:
                raise Fail(4, f"timer-restart-failed:{unit}")


def rollback(
    records: list[dict], did_reload: bool, did_vector_restart: bool
) -> list[str]:
    """Best-effort restore of this run's replaced files (spec section 5.3)."""
    log: list[str] = []
    for record in reversed(records):
        target = p(record["target"])
        backup = record.get("backup")
        try:
            if backup and Path(backup).exists():
                shutil.copy2(backup, target)
                log.append(f"restored:{record['target']}")
            elif record["action"] == "created" and target.exists():
                target.unlink()
                log.append(f"removed:{record['target']}")
        except OSError as exc:
            log.append(f"rollback-error:{record['target']}:{type(exc).__name__}")
    if did_reload:
        run_cmd(["systemctl", "daemon-reload"])
    if did_vector_restart:
        run_cmd(["systemctl", "restart", "vector"])
    return log


def run_apply(args: argparse.Namespace) -> dict:
    findings, entries = run_check(args)
    errors = [f for f in findings if f["category"] == "error"]
    blocked = [f for f in findings if f["category"] in BLOCKED_CATEGORIES]
    if errors:
        raise Fail(2, "check-incomplete", findings=findings)
    if blocked:
        raise Fail(3, "blocked-by-unmanaged-state", findings=blocked)

    plan: list[tuple[Entry, bytes]] = []
    for entry in entries:
        content = source_bytes(args.repo, args.ref, args.source == "git", entry.rel)
        target = p(entry.target)
        if content is None:
            raise Fail(2, f"source-missing:{entry.rel}")
        needs = (
            not target.exists()
            or target.read_bytes() != content
            or stat_problem(target, entry.mode) is not None
        )
        if needs:
            plan.append((entry, content))
    if not plan:
        return {"action": "none", "installed": [], "changed_units": []}

    validate_candidates(args, plan)

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    backup_dir = p(f"/var/lib/pareton-deploy/sync-backup/{stamp}")
    records: list[dict] = []
    did_reload = False
    did_vector = False
    try:
        for entry, content in plan:
            records.append(install_entry(entry, content, backup_dir))

        changed = units_changed(plan)
        if changed:
            daemon_reload()
            did_reload = True
            restart_changed_timers(changed)
            pending = load_pending()
            pending["daemon_reload"] = False

        if vector_changed(plan):
            restart_vector()
            did_vector = True
            pending = load_pending()
            pending["vector_restart"] = False
            save_pending(pending)
    except Fail as failure:
        log = rollback(records, did_reload, did_vector)
        pending = load_pending()
        if not did_reload and units_changed(plan):
            pending["daemon_reload"] = True
        save_pending(pending)
        raise Fail(
            failure.code, failure.reason, rollback=log, installed=records
        ) from failure

    pending = load_pending()
    replaced_units = {
        Path(r["target"]).name
        for r in records
        if r.get("action") == "replaced"
        and r["target"].endswith((".service", ".timer"))
    }
    replaced_units |= {
        Path(r["target"]).parent.name[: -len(".d")]
        for r in records
        if r.get("action") == "replaced"
        and Path(r["target"]).parent.name.endswith(".service.d")
    }
    owed = [u for u in APP_UNITS if u in replaced_units]
    # Only a *replaced* worker unit owes a restart. A freshly created unit was
    # never running; starting it is the operator's call (spec section 4.2).
    for unit, flag in WORKER_PENDING.items():
        if unit in replaced_units:
            (Path(args.repo) / flag).touch()
    if owed:
        pending["owed_restart_units"] = sorted(
            set(pending["owed_restart_units"]) | set(owed)
        )
    save_pending(pending)

    post_findings, _ = run_check(args)
    remaining = [f for f in post_findings if f["category"] in DRIFT_CATEGORIES]
    if remaining:
        raise Fail(
            4, "post-install-check-unclean", findings=remaining, installed=records
        )

    return {
        "action": "applied",
        "installed": records,
        "changed_units": units_changed(plan),
        "vector_restarted": did_vector,
        "daemon_reloaded": did_reload,
        "owed_restart_units": owed,
    }


# --------------------------------------------------------------------------
# deploy-hook (spec section 5.3 decision table)


def retry_owed(args: argparse.Namespace) -> dict:
    pending = load_pending()
    performed = []
    if pending["daemon_reload"]:
        daemon_reload()
        pending["daemon_reload"] = False
        performed.append("daemon-reload")
    if pending["vector_restart"]:
        restart_vector()
        pending["vector_restart"] = False
        performed.append("vector-restart")
    save_pending(pending)
    return {"performed": performed, "owed_restart_units": pending["owed_restart_units"]}


def run_deploy_hook(args: argparse.Namespace) -> dict:
    findings, _ = run_check(args)
    code = exit_code_for(findings)
    if code == 2:
        raise Fail(2, "check-incomplete", findings=findings)
    if code == 3:
        raise Fail(3, "blocked", findings=findings)
    if findings:
        result = run_apply(args)
        result["action"] = result.get("action", "applied")
        return result
    result = retry_owed(args)
    return {"action": "none", "findings": [], **result}


# --------------------------------------------------------------------------
# CLI


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo", required=True, help="repository root (explicit, spec 5)"
    )
    parser.add_argument("--ref", default="HEAD", help="Git ref to read sources from")
    parser.add_argument("--source", choices=("git", "worktree"), default="git")
    parser.add_argument("--env-file", default="/opt/pareton/.env")
    parser.add_argument("--notify-program", default=None)
    parser.add_argument("--skip-validation", action="store_true")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("check")
    sub.add_parser("apply")
    sub.add_parser("deploy-hook")
    sub.add_parser("owed-restarts")
    clear = sub.add_parser("clear-restarts")
    clear.add_argument("units", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.repo = Path(args.repo)
    try:
        if args.mode == "check":
            findings, _ = run_check(args)
            emit(
                {
                    "mode": "check",
                    "summary": findings_summary(findings),
                    "findings": findings,
                }
            )
            return exit_code_for(findings)
        if args.mode == "apply":
            emit({"mode": "apply", **run_apply(args)})
            return 0
        if args.mode == "deploy-hook":
            emit({"mode": "deploy-hook", **run_deploy_hook(args)})
            return 0
        if args.mode == "owed-restarts":
            pending = load_pending()
            print(" ".join(pending["owed_restart_units"]))
            return 0
        if args.mode == "clear-restarts":
            pending = load_pending()
            pending["owed_restart_units"] = [
                u for u in pending["owed_restart_units"] if u not in args.units
            ]
            save_pending(pending)
            emit(
                {
                    "mode": "clear-restarts",
                    "owed_restart_units": pending["owed_restart_units"],
                }
            )
            return 0
    except Fail as failure:
        payload = {"mode": args.mode, "error": failure.reason, **failure.extra}
        emit(payload)
        print(f"sync-config: {failure.reason}", file=sys.stderr)
        return failure.code
    except Exception as exc:  # noqa: BLE001 - never leak env values via tracebacks
        emit({"mode": args.mode, "error": type(exc).__name__})
        print(f"sync-config: internal error ({type(exc).__name__})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
