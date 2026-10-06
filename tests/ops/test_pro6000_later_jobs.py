"""Later controller lifecycle tests; Python workloads/database are stubbed."""

import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def environment(tmp_path, stage, *, fail="", hold=""):
    (tmp_path / "step2.exit-code").write_text("0\n")
    rule = tmp_path / "sampling_rule.json"
    rule.write_text("{}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        """#!/bin/bash
case "$*" in
  *ops.pro6000_preflight*)
    if [[ "$TEST_FAIL" == preflight ]]; then echo 'missing tokenizers' >&2; exit 9; fi
    exit 0 ;;
  *bench.preview_longform*) phase=preview ;;
  *ops.pro6000_model_volume*) phase=shadow ;;
  *campaign.seed*) phase=seed ;;
  *' -c '*) phase=engine ;;
  *) phase=request; cat >/dev/null ;;
esac
printf '%s\\n' "$phase" >> "$TEST_CALLS"
if [[ "$phase" == "$TEST_HOLD" ]]; then
  touch "$TEST_ROOT/ready"
  while [[ ! -f "$TEST_ROOT/release" ]]; do /bin/sleep 0.05; done
fi
if [[ "$phase" == "$TEST_FAIL" ]]; then echo "$phase failed" >&2; exit 9; fi
if [[ "$phase" == engine ]]; then
  printf 'ghcr.io/pareton-ai/pareton-baseline@sha256:%064d\\n' 0
fi
"""
    )
    python.chmod(0o755)
    jq = bin_dir / "jq"
    jq.write_text('#!/bin/bash\necho report >> "$TEST_CALLS"\n')
    jq.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_ROOT": str(tmp_path),
        "TEST_CALLS": str(tmp_path / "calls"),
        "TEST_HOLD": hold,
        "TEST_FAIL": fail,
        "PRO6000_RUN_DIR": str(tmp_path),
        "PRO6000_SEED_DIR": str(tmp_path),
        "PRO6000_FIELDS": "unused",
        "PRO6000_QUALIFIED_RULE": str(rule),
        "PARETON_DATABASE_URL": "",
        "PARETON_TEST_DATABASE_URL": "",
    }
    script = (
        ROOT
        / "ops"
        / ("shadow-pro6000.sh" if stage == "step3" else "seed-pro6000-job.sh")
    )
    return env, script


@pytest.mark.parametrize(
    ("stage", "fail", "expected"),
    [
        ("step3", "preflight", []),
        ("seed", "preflight", []),
        ("step3", "preview", ["preview"]),
        ("step3", "request", ["preview", "request"]),
        ("step3", "shadow", ["preview", "request", "shadow"]),
        ("step3", "", ["preview", "request", "shadow", "report"]),
        ("seed", "engine", ["engine"]),
        ("seed", "seed", ["engine", "seed"]),
        ("seed", "", ["engine", "seed"]),
    ],
)
def test_later_jobs_stop_on_failure_and_record_status(tmp_path, stage, fail, expected):
    env, script = environment(tmp_path, stage, fail=fail)
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=5
    )
    status = 9 if fail else 0
    assert result.returncode == status, result.stderr
    assert (tmp_path / f"{stage}.exit-code").read_text().strip() == str(status)
    calls = tmp_path / "calls"
    assert (calls.read_text().splitlines() if calls.exists() else []) == expected


def test_shadow_requires_completed_qualification(tmp_path):
    env, script = environment(tmp_path, "step3")
    (tmp_path / "step2.exit-code").unlink()
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=5
    )
    assert result.returncode != 0
    assert not (tmp_path / "calls").exists()
    assert not (tmp_path / "step3.lock").exists()


def wait_for(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, f"timed out waiting for {path}"
        time.sleep(0.02)


@pytest.mark.parametrize(
    ("stage", "hold"),
    [("step3", "preview"), ("step3", "request"), ("step3", "shadow"), ("seed", "seed")],
)
def test_later_jobs_survive_hangup_and_log_viewer_interrupt(tmp_path, stage, hold):
    env, script = environment(tmp_path, stage, hold=hold)
    doc = (ROOT / "ops/README.md").read_text()
    section = doc.split("#### 3." if stage == "step3" else "#### 4.")[1]
    block = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
    # Use the documented launch itself; seed environment is already stubbed.
    launch = next(line for line in block.splitlines() if line.startswith("nohup "))
    subprocess.run(
        ["bash", "-c", "set -m\n" + launch],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        start_new_session=True,
        check=True,
        timeout=5,
    )
    viewer = None
    try:
        wait_for(tmp_path / "ready")
        pid = int((tmp_path / f"{stage}.pid").read_text())
        assert os.getpgid(pid) == pid
        os.killpg(pid, signal.SIGHUP)
        viewer = subprocess.Popen(
            ["tail", "-f", str(tmp_path / f"{stage}.log")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        os.killpg(viewer.pid, signal.SIGINT)
        assert viewer.wait(timeout=5) != 0
        os.kill(pid, 0)
        assert not (tmp_path / f"{stage}.exit-code").exists()
        duplicate = subprocess.run(
            ["bash", str(script)], env=env, capture_output=True, text=True, timeout=5
        )
        assert duplicate.returncode != 0
        assert int((tmp_path / f"{stage}.pid").read_text()) == pid
        assert not (tmp_path / f"{stage}.exit-code").exists()
    finally:
        (tmp_path / "release").touch()
        if viewer is not None and viewer.poll() is None:
            viewer.kill()
            viewer.wait(timeout=5)
        wait_for(tmp_path / f"{stage}.exit-code")
    assert (tmp_path / f"{stage}.exit-code").read_text().strip() == "0"
    expected = (
        ["preview", "request", "shadow", "report"]
        if stage == "step3"
        else ["engine", "seed"]
    )
    assert (tmp_path / "calls").read_text().splitlines() == expected
