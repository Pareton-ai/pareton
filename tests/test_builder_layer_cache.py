"""Registry layer cache validation and deterministic Git source contexts."""

import os
import subprocess
from pathlib import Path

import pytest

from builder import hermetic
from builder.layer_cache import cache_arguments, stabilize_git_metadata

COMMIT = "a" * 40
BASE = "ghcr.io/pareton-ai/base@sha256:" + "1" * 64
CACHE = "ghcr.io/pareton-ai/layers@sha256:" + "2" * 64
TAG = "ghcr.io/pareton-ai/layers:test"
IMAGE = "ghcr.io/pareton-ai/engine:test"


def git(repo, *args):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Cache Test",
            "-c",
            "user.email=cache@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "origin"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    (repo / "hello.c").write_text("int main(void) { return 0; }\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "first")
    git(repo, "tag", "-a", "v1.0.0", "-m", "release")
    (repo / "README").write_text("Pinned build\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "second")
    return repo, git(repo, "rev-parse", "HEAD")


def test_fresh_clones_have_identical_metadata_and_preserve_version(
    repository, tmp_path
):
    repo, commit = repository
    first = tmp_path / "first"
    second = tmp_path / "second"
    git(tmp_path, "clone", "--no-local", str(repo), str(first))
    git(first, "checkout", "--detach", commit)
    expected_version = git(first, "describe", "--tags", "--dirty")
    # A later clone has extra refs, objects, index stat data and different config.
    (repo / "README").write_text("Later release\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "third")
    git(repo, "tag", "-a", "v2.0.0", "-m", "next release")
    git(tmp_path, "clone", "--no-local", str(repo), str(second))
    git(second, "checkout", "--detach", commit)
    git(second, "config", "remote.origin.url", "https://different.example/repo.git")
    snapshots = []
    for clone in (first, second):
        stabilize_git_metadata(clone)
        snapshots.append(
            {
                str(p.relative_to(clone)): p.read_bytes()
                for p in clone.rglob("*")
                if p.is_file()
            }
        )
        assert git(clone, "describe", "--tags", "--dirty") == expected_version
        assert git(clone, "status", "--porcelain") == ""
        assert git(clone, "rev-parse", "HEAD") == commit
        assert git(clone, "rev-list", "--count", "HEAD") == "2"
        assert git(clone, "tag") == "v1.0.0"
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"trusted": False},
        {"baseline_commit": "main"},
        {"base_image": "base:latest"},
        {"cache_from": TAG},
        {"cache_to": CACHE},
        {"cache_to": IMAGE},
        {"cache_to": TAG + ",mode=min"},
    ],
)
def test_invalid_cache_requests_rejected(overrides):
    args = dict(
        cache_from=CACHE,
        cache_to=TAG,
        image_ref=IMAGE,
        baseline_commit=COMMIT,
        base_image=BASE,
        trusted=True,
    )
    args.update(overrides)
    with pytest.raises(ValueError):
        cache_arguments(**args)


def test_opt_out_preserves_existing_build_inputs():
    assert (
        cache_arguments(
            None,
            None,
            image_ref=IMAGE,
            baseline_commit="main",
            base_image="base:latest",
            trusted=False,
        )
        == []
    )


def test_build_wires_registry_cache_and_returns_digest(
    repository, tmp_path, monkeypatch
):
    repo, commit = repository
    monkeypatch.setattr(hermetic.config, "BUILDER_LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(hermetic, "_base_image_torch_arch", lambda _: "9.0")
    monkeypatch.setattr(hermetic, "_docker_login_ghcr", lambda: None)
    monkeypatch.setattr(hermetic, "resolve_cache_ref", lambda _: CACHE)
    seen = []

    def build(command, **kwargs):
        seen.append(command)
        clone = Path(command[-1]) / "baseline"
        assert git(clone, "rev-parse", "HEAD") == commit
        assert not (clone / ".git" / "logs").exists()
        return 0

    monkeypatch.setattr(hermetic, "_run_logged", build)
    result = hermetic.build_engine_image(
        baseline_repo=str(repo),
        baseline_commit=commit,
        base_image=BASE,
        patch_bytes=b"",
        patch_hash="sha256:" + "3" * 64,
        allow_empty_patch=True,
        push=False,
        image_ref_override=IMAGE,
        log_dir=tmp_path / "logs",
        layer_cache_from=CACHE,
        layer_cache_to=TAG,
    )
    assert result.ok, result.evidence
    assert result.evidence["layer_cache_ref"] == CACHE
    command = seen[0]
    assert command[command.index("--cache-from") + 1] == f"type=registry,ref={CACHE}"
    assert (
        command[command.index("--cache-to") + 1] == f"type=registry,ref={TAG},mode=max"
    )


def test_patch_build_cannot_import_or_publish_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(hermetic.config, "BUILDER_LOCK_PATH", tmp_path / "lock")
    result = hermetic.build_engine_image(
        baseline_repo="unused",
        baseline_commit=COMMIT,
        base_image=BASE,
        patch_bytes=b"a patch",
        patch_hash="unused",
        allow_empty_patch=True,
        layer_cache_to=TAG,
    )
    assert not result.ok
    assert result.reason == "build_config_invalid"


def test_cli_forwards_options_and_reports_cache_digest(monkeypatch, capsys):
    from builder import __main__ as cli
    from gate.types import GateResult, SubmissionState

    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return GateResult.success(
            SubmissionState.BUILT, image_ref=IMAGE, layer_cache_ref=CACHE
        )

    monkeypatch.setattr(cli, "build_engine_image", build)
    assert (
        cli.main(
            [
                "--baseline-repo",
                "unused",
                "--baseline-commit",
                COMMIT,
                "--base-image",
                BASE,
                "--image-ref",
                IMAGE,
                "--empty-patch",
                "--layer-cache-from",
                CACHE,
                "--layer-cache-to",
                TAG,
            ]
        )
        == 0
    )
    assert captured["layer_cache_from"] == CACHE
    assert captured["layer_cache_to"] == TAG
    output = capsys.readouterr()
    assert output.out.strip() == IMAGE
    assert f"layer_cache_ref={CACHE}" in output.err
