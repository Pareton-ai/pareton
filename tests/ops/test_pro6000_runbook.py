"""Execute the documented shell control flow with no Docker, GPU, or network."""

import os
import re
import signal
import time
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def fake_environment(tmp_path, scenario):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "fake"
    fake.write_text(
        """#!/bin/bash
command=$(basename "$0")
printf '%s %s\\n' "$command" "$*" >> "$TEST_CALLS"
case "$command" in
  python)
    if [[ "$*" == *bench.qualify_longform* ]]; then
      if [[ "$TEST_CASE" == qualification_failure ]]; then
        echo 'qualification failed' >&2; exit 9
      fi
    else
      cat >/dev/null
      if [[ "$TEST_CASE" == detached ]]; then
        touch "$PRO6000_RUN_DIR/start-ready"
        while [[ ! -f "$PRO6000_RUN_DIR/release" ]]; do /bin/sleep 0.05; done
      fi
      if [[ "$TEST_CASE" == start_failure ]]; then
        echo 'original startup error' >&2; exit 7
      fi
    fi
    ;;
  docker)
    if [[ "$1" == inspect ]]; then
      if [[ "$TEST_CASE" == missing_container ]]; then
        echo 'No such container' >&2; exit 1
      elif [[ "$TEST_CASE" == exited_container ]]; then
        echo exited
      else
        echo running
      fi
    elif [[ "$1" == logs ]]; then
      echo 'container startup log'
    fi
    ;;
  curl)
    case "$TEST_CASE" in
      health_timeout|interrupt) exit 7 ;;
      *) echo '{"data": []}' ;;
    esac
    ;;
  sleep)
    if [[ "$TEST_CASE" == interrupt ]]; then
      kill -INT "$PPID"
    else
      exit 99
    fi
    ;;
  jq) echo test-model-volume ;;
esac
"""
    )
    fake.chmod(0o755)
    for command in ("python", "docker", "curl", "sleep", "jq"):
        (bin_dir / command).symlink_to(fake)
    calls_path = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_CASE": scenario,
        "TEST_CALLS": str(calls_path),
        "PRO6000_RUN_DIR": str(tmp_path),
        "PRO6000_FIELDS": "unused-fields.json",
        "PRO6000_ENGINE_REF": "test-image",
        "PRO6000_BASELINE_CONTAINER": "test-baseline",
        "PRO6000_QUAL_NET": "test-network",
        "PARETON_BENCH_HEALTH_TIMEOUT_S": "0" if scenario == "health_timeout" else "60",
    }
    return env, calls_path


@pytest.mark.parametrize(
    ("scenario", "status", "qualifies", "cleans"),
    [
        ("start_failure", 7, False, False),
        ("missing_container", 1, False, False),
        ("exited_container", 1, False, False),
        ("health_timeout", 1, False, False),
        ("interrupt", 130, False, False),
        ("qualification_failure", 9, True, False),
        ("success", 0, True, True),
    ],
)
def test_step2_stops_before_dependent_work(
    tmp_path, scenario, status, qualifies, cleans
):
    env, calls_path = fake_environment(tmp_path, scenario)
    script = ROOT / "ops/qualify-pro6000.sh"
    # Deliberately disable the parent shell's errexit, as in an interactive paste.
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == status, result.stderr
    assert (tmp_path / "step2.exit-code").read_text().strip() == str(status)
    calls = calls_path.read_text()
    assert ("bench.qualify_longform" in calls) == qualifies
    assert ("docker stop" in calls) == cleans
    assert ("docker rm" in calls) == cleans
    assert ("docker volume rm" in calls) == cleans
    assert ("docker network rm" in calls) == cleans
    if scenario == "start_failure":
        assert "original startup error" in result.stderr
        assert "docker inspect" not in calls
        assert "curl " not in calls
    if scenario in ("missing_container", "exited_container"):
        assert "curl " not in calls
    if scenario == "exited_container":
        assert "container startup log" in result.stderr
    if scenario == "qualification_failure":
        assert "qualification failed" in result.stderr
    if status:
        assert "Step 2 stopped" in result.stderr


def wait_for(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, f"timed out waiting for {path}"
        time.sleep(0.02)


def test_nohup_controller_survives_hangup_and_viewer_interrupt(tmp_path):
    env, calls_path = fake_environment(tmp_path, "detached")
    section = (ROOT / "ops/README.md").read_text().split("#### 2.")[1]
    launch = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
    # Job control models the interactive SSH shell's separate process groups.
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
        wait_for(tmp_path / "start-ready")
        pid = int((tmp_path / "step2.pid").read_text())
        assert os.getpgid(pid) == pid
        os.killpg(pid, signal.SIGHUP)
        viewer = subprocess.Popen(
            ["tail", "-f", str(tmp_path / "step2.log")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        os.killpg(viewer.pid, signal.SIGINT)
        assert viewer.wait(timeout=5) != 0
        os.kill(pid, 0)  # The actual background controller is still alive.
        assert not (tmp_path / "step2.exit-code").exists()

        # An accidental duplicate launch must preserve the active PID/status.
        duplicate = subprocess.run(
            ["bash", str(ROOT / "ops/qualify-pro6000.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert duplicate.returncode != 0
        assert int((tmp_path / "step2.pid").read_text()) == pid
        assert not (tmp_path / "step2.exit-code").exists()
    finally:
        (tmp_path / "release").touch()
        if viewer is not None and viewer.poll() is None:
            viewer.kill()
            viewer.wait(timeout=5)
        wait_for(tmp_path / "step2.exit-code")
    assert (tmp_path / "step2.exit-code").read_text().strip() == "0"
    calls = calls_path.read_text()
    assert calls.count("bench.qualify_longform") == 1
    assert "docker network rm" in calls
