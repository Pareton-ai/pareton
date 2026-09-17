"""Stage-2 deploy entry tests: ops/deploy.sh is a thin wrapper.

The stage-1 behaviors this file used to cover (pending-flag worker
deferral, in-script busy probes, run_owed_restarts) moved into
ops/release.py's tick state machine and are covered by
tests/ops/test_release.py. What remains here is the wrapper contract:
the deploy unit's ExecStart must end up in ``release.py tick`` with
PARETON_OPS_DIR honored (stage-2 spec 4.5).
"""

import os
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY = REPO_ROOT / "ops" / "deploy.sh"


def test_wrapper_execs_release_tick(tmp_path):
    ops = tmp_path / "ops"
    ops.mkdir()
    stub = ops / "release.py"
    stub.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env python3
            import json, sys, os
            print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}))
            """
        ).lstrip()
    )
    stub.chmod(0o755)
    env = {**os.environ, "PARETON_OPS_DIR": str(ops)}
    result = subprocess.run(
        ["bash", str(DEPLOY)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["argv"] == ["tick"]


def test_wrapper_passes_extra_args(tmp_path):
    ops = tmp_path / "ops"
    ops.mkdir()
    stub = ops / "release.py"
    stub.write_text("#!/usr/bin/env python3\nimport sys\nprint(sys.argv[1:])\n")
    stub.chmod(0o755)
    env = {**os.environ, "PARETON_OPS_DIR": str(ops)}
    result = subprocess.run(
        ["bash", str(DEPLOY), "--continue-op", "op-9"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert result.stdout.strip().endswith("['tick', '--continue-op', 'op-9']")


import json  # noqa: E402  (kept last to mirror runtime-only usage above)
