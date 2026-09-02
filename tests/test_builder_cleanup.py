"""Unit tests for persistent Docker builder cleanup."""

from __future__ import annotations

import shutil
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from builder import cleanup

_BASE = "sha256:" + "a" * 64
_ENGINE = "sha256:" + "b" * 64
_CANDIDATE = "ghcr.io/pareton-ai/pareton-engine:" + "c" * 64
_USAGE_TYPE = type(shutil.disk_usage("/"))


def _campaign(status: str = "open") -> SimpleNamespace:
    return SimpleNamespace(
        campaign_id="campaign-1",
        status=status,
        base_image_digest=_BASE,
        bench={"baseline_engine_image_digest": _ENGINE},
    )


class DockerFake:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.ids = {
            "ghcr.io/pareton-ai/pareton-baseline@" + _BASE: "base-id",
            "ghcr.io/pareton-ai/pareton-engine@" + _ENGINE: "engine-id",
        }

    def __call__(self, cmd, **_kwargs):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[:3] == ["docker", "image", "ls"]:
            ref_filter = cmd[cmd.index("--filter") + 1]
            output = (
                "pareton-retain:expired\n"
                if "pareton-retain" in ref_filter
                else _CANDIDATE + "\n"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")
        if cmd[:3] == ["docker", "image", "inspect"]:
            image_id = self.ids.get(cmd[-1])
            return subprocess.CompletedProcess(
                cmd, 0 if image_id else 1, stdout=image_id or "", stderr=""
            )
        if cmd[:2] == ["docker", "tag"]:
            self.ids[cmd[-1]] = self.ids[cmd[-2]]
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def _usage(percent: int):
    return _USAGE_TYPE(100, percent, 100 - percent)


@pytest.mark.unit
def test_retention_tags_only_cover_active_campaigns():
    tags = cleanup._retention_tags([_campaign("draft"), _campaign("closed")])
    assert set(tags) == {
        "pareton-retain:campaign-1-build",
        "pareton-retain:campaign-1-engine",
    }


@pytest.mark.unit
def test_cleanup_retains_baselines_removes_candidates_and_preserves_ccache(
    monkeypatch,
):
    fake = DockerFake()
    monkeypatch.setattr(cleanup.subprocess, "run", fake)
    monkeypatch.setattr(cleanup.shutil, "disk_usage", lambda _path: _usage(80))

    result = cleanup.cleanup_once([_campaign()])

    assert result["pruned"] is True
    assert ["docker", "image", "rm", _CANDIDATE] in fake.calls
    assert any(cmd[:2] == ["docker", "tag"] for cmd in fake.calls)
    buildx = next(cmd for cmd in fake.calls if cmd[:3] == ["docker", "buildx", "prune"])
    assert "type!=exec.cachemount" in buildx
    assert not any(cmd[:3] == ["docker", "system", "prune"] for cmd in fake.calls)


@pytest.mark.unit
def test_cleanup_continues_to_prune_after_candidate_removal_timeout(monkeypatch):
    fake = DockerFake()

    def run(cmd, **kwargs):
        if list(cmd) == ["docker", "image", "rm", _CANDIDATE]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=120)
        return fake(cmd, **kwargs)

    monkeypatch.setattr(cleanup, "_run", run)
    monkeypatch.setattr(cleanup.shutil, "disk_usage", lambda _path: _usage(80))

    result = cleanup.cleanup_once([_campaign()])

    assert result["pruned"] is True
    assert any(cmd[:3] == ["docker", "image", "prune"] for cmd in fake.calls)


@pytest.mark.unit
def test_dry_run_and_database_failure_do_not_mutate(monkeypatch):
    fake = DockerFake()
    monkeypatch.setattr(cleanup.subprocess, "run", fake)
    monkeypatch.setattr(cleanup.shutil, "disk_usage", lambda _path: _usage(95))
    cleanup.cleanup_once([_campaign()], dry_run=True)
    assert not any(cmd[:2] == ["docker", "tag"] for cmd in fake.calls)
    assert not any(
        cmd[:3] in (["docker", "image", "rm"], ["docker", "image", "prune"])
        for cmd in fake.calls
    )

    monkeypatch.setattr(
        cleanup,
        "_load_campaigns",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    assert cleanup.main([]) == 1


@pytest.mark.unit
def test_main_loads_campaigns_after_acquiring_lock(monkeypatch):
    events = []

    @contextmanager
    def lock(**_kwargs):
        events.append("lock")
        yield True

    monkeypatch.setattr(cleanup, "builder_storage_lock", lock)
    monkeypatch.setattr(cleanup, "_load_campaigns", lambda: events.append("load") or [])
    monkeypatch.setattr(
        cleanup,
        "cleanup_once",
        lambda campaigns, **_kwargs: {
            "usage_after_percent": 0,
        },
    )

    assert cleanup.main([]) == 0
    assert events == ["lock", "load"]


@pytest.mark.unit
def test_evict_rejects_non_candidate_and_removes_exact_candidate(monkeypatch, tmp_path):
    fake = DockerFake()
    monkeypatch.setattr(cleanup.subprocess, "run", fake)
    monkeypatch.setattr(cleanup.config, "BUILDER_LOCK_PATH", tmp_path / "lock")
    with pytest.raises(ValueError, match="not a Pareton candidate"):
        cleanup.evict_candidate_image("ghcr.io/pareton-ai/pareton-engine:baseline")
    assert cleanup.evict_candidate_image(_CANDIDATE) is True
    assert ["docker", "image", "rm", _CANDIDATE] in fake.calls


@pytest.mark.unit
def test_evict_skips_when_builder_storage_lock_is_busy(monkeypatch):
    fake = DockerFake()

    @contextmanager
    def busy_lock(**_kwargs):
        yield False

    monkeypatch.setattr(cleanup, "builder_storage_lock", busy_lock)
    monkeypatch.setattr(cleanup.subprocess, "run", fake)

    assert cleanup.evict_candidate_image(_CANDIDATE) is False
    assert fake.calls == []


@pytest.mark.unit
def test_evict_normalizes_docker_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(cleanup.config, "BUILDER_LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(
        cleanup,
        "_run",
        lambda cmd: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd=cmd, timeout=120)
        ),
    )

    with pytest.raises(RuntimeError, match="docker image rm timed out"):
        cleanup.evict_candidate_image(_CANDIDATE)
