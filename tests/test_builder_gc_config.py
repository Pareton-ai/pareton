"""Tests for the production Docker builder GC invariant."""

from __future__ import annotations

import json

import pytest

import config
from builder.gc_config import validate_daemon_gc_config, validate_daemon_gc_file


@pytest.mark.unit
def test_daemon_gc_must_be_disabled():
    validate_daemon_gc_config({"builder": {"gc": {"enabled": False}}})

    unsafe = [
        {},
        {"builder": {}},
        {"builder": {"gc": {}}},
        {"builder": {"gc": {"enabled": True}}},
    ]
    for data in unsafe:
        with pytest.raises(ValueError, match="builder.gc.enabled=false"):
            validate_daemon_gc_config(data)


@pytest.mark.unit
def test_committed_daemon_config_preserves_ccache_without_selecting_image_store():
    path = config.REPO_ROOT / "ops" / "docker" / "daemon.json"
    validate_daemon_gc_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "containerd-snapshotter" not in data.get("features", {})


@pytest.mark.unit
def test_missing_or_invalid_daemon_config_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        validate_daemon_gc_file(tmp_path / "missing.json")

    path = tmp_path / "daemon.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        validate_daemon_gc_file(path)
