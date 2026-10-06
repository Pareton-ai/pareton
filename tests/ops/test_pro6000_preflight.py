"""Dependency checks are CPU-only and fail before any engine work."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "preflight", ROOT / "ops/pro6000_preflight.py"
)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
pytestmark = pytest.mark.unit


def load(name):
    return SimpleNamespace(
        __version__={"tokenizers": "0.22.2", "jinja2": "3.1.6"}.get(name, "0")
    )


def test_missing_tokenizer_and_late_shadow_import_are_reported_together():
    def imports(name):
        if name in ("tokenizers", "worker.round_job"):
            raise ModuleNotFoundError(name)
        return load(name)

    errors = preflight.check(imports)
    assert len(errors) == 2
    assert errors[0].startswith("tokenizers: ModuleNotFoundError")
    assert errors[1].startswith("worker.round_job: ModuleNotFoundError")


def test_tokenizer_and_template_versions_follow_repo_requirements():
    assert preflight.check(load) == []
    errors = preflight.check(lambda name: SimpleNamespace(__version__="0"))
    assert len(errors) == 2
    assert "tokenizers: require 0.22.2" in errors[0]
    assert "jinja2: require 3.1.6" in errors[1]


@pytest.mark.parametrize(
    "readonly, name, accepted",
    [(True, "expected", True), (False, "expected", False), (True, "wrong", False)],
)
def test_reuse_checks_recorded_volume(tmp_path, monkeypatch, readonly, name, accepted):
    import json
    import re
    import subprocess
    import bench.qualify_longform as qualifier

    record = {"name": "expected", "verified": True}
    (tmp_path / "model_volume.json").write_text(json.dumps(record))
    monkeypatch.setenv("PRO6000_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("PRO6000_BASELINE_CONTAINER", "baseline")
    monkeypatch.setenv("PRO6000_ENGINE_REF", "pinned")
    checked = []
    monkeypatch.setattr(
        qualifier, "verify_baseline_image", lambda **kw: checked.append(kw)
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps(
                [
                    {
                        "Mounts": [
                            {
                                "Type": "volume",
                                "Name": name,
                                "Destination": "/model",
                                "RW": not readonly,
                            }
                        ]
                    }
                ]
            )
        ),
    )
    code = re.search(
        r"<<'REUSE'\n(.*?)\nREUSE", (ROOT / "ops/qualify-pro6000.sh").read_text(), re.S
    ).group(1)
    if accepted:
        exec(compile(code, "<reuse>", "exec"), {})
    else:
        with pytest.raises(RuntimeError, match="recorded verified read-only"):
            exec(compile(code, "<reuse>", "exec"), {})
    assert checked == [
        {
            "engine_ref": "pinned",
            "base_url": "http://127.0.0.1:30000",
            "container": "baseline",
        }
    ]


@pytest.mark.parametrize("fail", [False, True])
def test_setup_stops_on_install_error_and_records_status(tmp_path, fail):
    import os
    import subprocess

    checkout = tmp_path / "checkout"
    (checkout / "ops").mkdir(parents=True)
    script = checkout / "ops/setup-pro6000.sh"
    script.write_text((ROOT / "ops/setup-pro6000.sh").read_text())
    output = tmp_path / "setup"
    output.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text("""#!/bin/bash
printf '%s\\n' "$*" >> "$TEST_CALLS"
if [[ "$*" == '-m venv .venv' ]]; then
  mkdir -p .venv/bin; touch .venv/bin/activate
elif [[ "$*" == *'pip install'* && "$TEST_FAIL" == yes ]]; then
  exit 9
fi
""")
    python.chmod(0o755)
    (bin_dir / "python3").symlink_to(python)
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PRO6000_SETUP_DIR": str(output),
        "TEST_CALLS": str(calls),
        "TEST_FAIL": "yes" if fail else "no",
    }
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=5
    )
    assert result.returncode == (9 if fail else 0), result.stderr
    assert (output / "setup.exit-code").read_text().strip() == str(result.returncode)
    commands = calls.read_text()
    assert "pip install -r requirements.txt" in commands
    assert ("pip check" in commands) != fail
    assert ("ops.pro6000_preflight" in commands) != fail
