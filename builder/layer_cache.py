"""Registry layer caching for trusted baselines with stable Git version metadata."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_PINNED = r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}"
_TAGGED = r"[a-z0-9][a-z0-9._:/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*"


def cache_arguments(
    cache_from: str | None,
    cache_to: str | None,
    *,
    image_ref: str,
    baseline_commit: str,
    base_image: str,
    trusted: bool,
) -> list[str]:
    if cache_from is None and cache_to is None:
        return []
    if not trusted:
        raise ValueError("layer cache options require a trusted empty-patch build")
    if not re.fullmatch(r"[0-9a-f]{40}", baseline_commit):
        raise ValueError("layer caching requires a full lowercase baseline commit SHA")
    if not re.fullmatch(_PINNED, base_image):
        raise ValueError("layer caching requires a digest-pinned base image")
    args = []
    if cache_from is not None:
        if not re.fullmatch(_PINNED, cache_from):
            raise ValueError(
                "--layer-cache-from must be a trusted digest-pinned registry cache"
            )
        args += ["--cache-from", f"type=registry,ref={cache_from}"]
    if cache_to is not None:
        if not re.fullmatch(_TAGGED, cache_to):
            raise ValueError("--layer-cache-to must be a registry image tag")
        if cache_to == image_ref:
            raise ValueError("layer cache and engine image must use different tags")
        args += ["--cache-to", f"type=registry,ref={cache_to},mode=max"]
    return args


def stabilize_git_metadata(repo: Path) -> None:
    """Keep reachable history/tags for versioning, without clone-specific bytes.

    Repack without reusing the clone's pack layout. read-tree creates an index
    with zeroed filesystem stat data; version tools can refresh it in-container.
    The working tree is untouched. Only opt-in trusted builds use this path.
    """

    def git(*args: str, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "git",
                "-c",
                "pack.window=10",
                "-c",
                "pack.depth=50",
                "-c",
                "pack.writeReverseIndex=false",
                "-c",
                "index.version=2",
                *args,
            ],
            cwd=repo,
            check=True,
            timeout=600,
            **kwargs,
        )

    commit = git("rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
    refs = git(
        "for-each-ref",
        "--merged=HEAD",
        "--sort=refname",
        "--format=%(refname) %(objectname)",
        "refs/tags",
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    with tempfile.TemporaryDirectory(prefix="pareton-git-") as directory:
        root = Path(directory)
        metadata = root / "git"
        (metadata / "objects" / "pack").mkdir(parents=True)
        (metadata / "refs").mkdir()
        (metadata / "HEAD").write_text(commit + "\n")
        (metadata / "config").write_text(
            "[core]\nrepositoryformatversion = 0\nbare = false\n"
        )
        revisions = [commit]
        for line in refs:
            name, oid = line.split()
            target = metadata / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(oid + "\n")
            revisions.append(oid)
        pack = root / "history.pack"
        with pack.open("wb") as output:
            git(
                "pack-objects",
                "--stdout",
                "--revs",
                "--threads=1",
                "--no-reuse-delta",
                "--no-reuse-object",
                "--compression=6",
                input="\n".join(revisions) + "\n",
                text=True,
                stdout=output,
                stderr=subprocess.PIPE,
            )
        with pack.open("rb") as source:
            git(
                f"--git-dir={metadata}",
                "index-pack",
                "--stdin",
                stdin=source,
                capture_output=True,
            )
        git(f"--git-dir={metadata}", "read-tree", commit, capture_output=True)
        shutil.rmtree(repo / ".git")
        shutil.move(str(metadata), repo / ".git")


def resolve_cache_ref(tag: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            tag,
            "--format",
            "{{json .Manifest}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    digest = json.loads(result.stdout)["digest"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("registry returned an invalid layer cache digest")
    return tag.rsplit(":", 1)[0] + "@" + digest
