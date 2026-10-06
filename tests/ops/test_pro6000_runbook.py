"""Execute the documented shell control flow with no Docker, GPU, or network."""

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


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
    section = (ROOT / "ops/README.md").read_text().split("#### 2.")[1]
    block = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
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
    # Deliberately disable the parent shell's errexit, as in an interactive paste.
    result = subprocess.run(
        ["bash", "-c", "set +e\n" + block],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == status, result.stderr
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
