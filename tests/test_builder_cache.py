"""Offline checks for snapshot validation, registry commands and failure handling."""

import json
import subprocess
from pathlib import Path

import pytest

from builder import cache

COMMIT = "a" * 40
NEXT_COMMIT = "b" * 40
BASE = "ghcr.io/pareton-ai/base@sha256:" + "1" * 64
SNAPSHOT = "ghcr.io/pareton-ai/cache@sha256:" + "2" * 64
TAG = "ghcr.io/pareton-ai/cache:sglang-test"


def args(action, *, engine="sglang", commit=COMMIT, image_ref=None):
    return [
        action,
        "--engine",
        engine,
        "--baseline-commit",
        commit,
        "--base-image",
        BASE,
        "--image-ref",
        image_ref or (TAG if action == "backup" else SNAPSHOT),
    ]


def metadata(**overrides):
    return {
        "version": 1,
        "engine": "sglang",
        "baseline_commit": COMMIT,
        "base_image": BASE,
        "platform": "linux/amd64",
        **overrides,
    }


@pytest.fixture
def docker(monkeypatch, tmp_path):
    monkeypatch.setattr(cache.config, "BUILDER_LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(cache.config, "BUILDER_NAME", "test-builder")
    monkeypatch.setattr(cache, "_docker_login_ghcr", lambda: None)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "imagetools" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {"config": {"Labels": {cache._LABEL: json.dumps(metadata())}}}
                ),
            )
        context = Path(command[-1])
        calls[-1] += ((context / "Dockerfile").read_text(),)
        if "--metadata-file" in command:
            Path(command[command.index("--metadata-file") + 1]).write_text(
                json.dumps({"containerimage.digest": "sha256:" + "2" * 64})
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cache.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_backup_publishes_cache_mount_as_layer(docker, capsys, engine):
    assert cache.main(args("backup", engine=engine)) == 0
    command, _kwargs, dockerfile = docker[0]
    assert command[command.index("--builder") + 1] == "test-builder"
    assert "--no-cache" not in command and "--push" in command
    assert any(value.startswith("PARETON_CACHE_NONCE=") for value in command)
    assert "--provenance=false" in command and "--network=none" in command
    assert "FROM scratch" in dockerfile
    assert f"id=pareton-ccache-{COMMIT}" in dockerfile
    assert "sharing=locked" in dockerfile
    assert "cp -a /root/.ccache/. /snapshot/ccache/" in dockerfile
    assert engine in dockerfile
    assert capsys.readouterr().out.strip() == SNAPSHOT
    assert not Path(command[-1]).exists()


def test_restore_can_seed_later_commit_without_exporting_helper(docker, capsys):
    assert cache.main(args("restore", commit=NEXT_COMMIT)) == 0
    command, _kwargs, dockerfile = docker[1]
    assert "--no-cache" not in command and "--network=none" in command
    assert any(value.startswith("PARETON_CACHE_NONCE=") for value in command)
    assert "type=cacheonly" in command
    assert "--push" not in command and "--load" not in command
    assert f"FROM {SNAPSHOT} AS snapshot" in dockerfile
    assert f"id=pareton-ccache-{NEXT_COMMIT}" in dockerfile
    assert "sharing=locked" in dockerfile
    assert "cp -a /seed/. /root/.ccache/" in dockerfile
    assert capsys.readouterr().out.strip() == f"pareton-ccache-{NEXT_COMMIT}"


@pytest.mark.parametrize(
    "overrides",
    [
        {"engine": "vllm"},
        {"platform": "linux/arm64"},
    ],
)
def test_incompatible_snapshot_rejected_before_mutating_cache(
    docker, monkeypatch, overrides
):
    monkeypatch.setattr(cache, "read_snapshot", lambda _: metadata(**overrides))
    assert cache.main(args("restore")) == 1
    assert not docker


def test_changed_toolchain_is_allowed_as_a_seed(docker, monkeypatch, capsys):
    monkeypatch.setattr(
        cache,
        "read_snapshot",
        lambda _: metadata(base_image="ghcr.io/pareton-ai/base@sha256:" + "3" * 64),
    )
    assert cache.main(args("restore")) == 0
    assert "Base image differs" in capsys.readouterr().err


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"config": {}},
        {"config": {"Labels": {cache._LABEL: json.dumps(metadata(version=2))}}},
    ],
)
def test_missing_or_unknown_snapshot_metadata_rejected(monkeypatch, payload):
    monkeypatch.setattr(
        cache.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, json.dumps(payload)),
    )
    with pytest.raises(ValueError, match="supported Pareton"):
        cache.read_snapshot(SNAPSHOT)


@pytest.mark.parametrize(
    "argv",
    [
        args("restore", image_ref=TAG),
        args("backup", image_ref=SNAPSHOT),
        args("backup", commit="main"),
        args("backup", commit=COMMIT + "\nRUN evil"),
        args("backup", image_ref="cache:tag\nRUN evil"),
    ],
)
def test_invalid_inputs_fail_before_docker(docker, argv):
    with pytest.raises(SystemExit) as exc:
        cache.main(argv)
    assert exc.value.code == 2
    assert not docker


@pytest.mark.parametrize("action", ["backup", "restore"])
def test_docker_failure_is_not_reported_as_success(docker, monkeypatch, capsys, action):
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1, a[0])

    monkeypatch.setattr(cache.subprocess, "run", fail)
    assert cache.main(args(action)) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "failed" in captured.err


def test_repeated_transfers_force_copy_without_resetting_mount(docker):
    assert cache.main(args("backup")) == 0
    assert cache.main(args("backup")) == 0
    nonces = [
        next(value for value in call[0] if value.startswith("PARETON_CACHE_NONCE="))
        for call in docker
    ]
    assert nonces[0] != nonces[1]
    assert all("--no-cache" not in call[0] for call in docker)
