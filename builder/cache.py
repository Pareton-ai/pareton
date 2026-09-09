"""Ops-only backup/restore of the builder's commit-scoped ccache via OCI images."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import config
from builder.digest import digest_pinned_ref
from builder.hermetic import _ccache_mount_id, _docker_login_ghcr
from builder.lock import serialized_build_storage
from campaign.engine import ENGINE_PRESETS

_LABEL = "ai.pareton.ccache"
_PINNED = r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}"
_TAGGED = r"[a-z0-9][a-z0-9._:/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*"


def snapshot_dockerfile(base_image: str, commit: str, metadata: dict) -> str:
    """Copy the mutable cache into an ordinary layer that a registry can store."""
    cache_id = _ccache_mount_id(commit)
    label = json.dumps(json.dumps(metadata, sort_keys=True))
    return f"""FROM {base_image} AS snapshot
ARG PARETON_CACHE_NONCE
RUN --mount=type=cache,id={cache_id},target=/root/.ccache,sharing=locked \\
    test -n "$(find /root/.ccache -path /root/.ccache/tmp -prune -o -type f ! -name stats ! -name ccache.conf ! -name CACHEDIR.TAG -print -quit)" \\
    && mkdir -p /snapshot/ccache \\
    && cp -a /root/.ccache/. /snapshot/ccache/ \\
    && rm -rf /snapshot/ccache/tmp /snapshot/ccache/ccache.conf
FROM scratch
COPY --from=snapshot /snapshot/ccache /ccache
LABEL {_LABEL}={label}
"""


def restore_dockerfile(base_image: str, commit: str, image_ref: str) -> str:
    """Merge snapshot entries into the same mount used by engine builds."""
    cache_id = _ccache_mount_id(commit)
    return f"""FROM {image_ref} AS snapshot
FROM {base_image}
ARG PARETON_CACHE_NONCE
RUN --mount=type=bind,from=snapshot,source=/ccache,target=/seed \\
    --mount=type=cache,id={cache_id},target=/root/.ccache,sharing=locked \\
    cp -a /seed/. /root/.ccache/
"""


def read_snapshot(image_ref: str) -> dict:
    # Read only image config, not the potentially multi-GB cache layers.
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            image_ref,
            "--format",
            "{{json .Image}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        image = json.loads(result.stdout)
        metadata = json.loads(image["config"]["Labels"][_LABEL])
        if (
            metadata["version"] != 1
            or metadata["engine"] not in ENGINE_PRESETS
            or not re.fullmatch(r"[0-9a-f]{40}", metadata["baseline_commit"])
            or not re.fullmatch(_PINNED, metadata["base_image"])
            or metadata["platform"] not in ("linux/amd64", "linux/arm64")
        ):
            raise ValueError("invalid snapshot metadata")
        return metadata
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("image is not a supported Pareton ccache snapshot") from exc


@serialized_build_storage
def transfer(args: argparse.Namespace) -> str:
    """Serialize with builds/cleanup; report success only after Docker completes."""
    _docker_login_ghcr()
    metadata = {
        "version": 1,
        "engine": args.engine,
        "baseline_commit": args.baseline_commit,
        "base_image": args.base_image,
        "platform": args.platform,
    }
    if args.action == "restore":
        source = read_snapshot(args.image_ref)
        for key in ("engine", "platform"):
            if source[key] != metadata[key]:
                raise ValueError(f"snapshot {key} does not match target {key}")
        print(
            f"Restoring {source['engine']} cache from {source['baseline_commit']} "
            f"into {args.baseline_commit}",
            file=sys.stderr,
        )
        if source["base_image"] != args.base_image:
            print(
                "Base image differs; compiler cache hits may decrease.", file=sys.stderr
            )
        dockerfile = restore_dockerfile(
            args.base_image, args.baseline_commit, args.image_ref
        )
    else:
        dockerfile = snapshot_dockerfile(
            args.base_image, args.baseline_commit, metadata
        )

    with tempfile.TemporaryDirectory(prefix="pareton-ccache-") as directory:
        context = Path(directory)
        (context / "Dockerfile").write_text(dockerfile)
        result_path = context / "result.json"
        command = [
            "docker",
            "buildx",
            "build",
            "--builder",
            config.BUILDER_NAME,
            "--platform",
            args.platform,
            # Force the copy to execute without --no-cache, which can reset
            # BuildKit cache mounts. A fresh ARG invalidates only the RUN layer.
            "--build-arg",
            f"PARETON_CACHE_NONCE={uuid.uuid4().hex}",
            "--network=none",
            "--progress=plain",
        ]
        if args.action == "backup":
            command += [
                "--push",
                "--provenance=false",
                "-t",
                args.image_ref,
                "--metadata-file",
                str(result_path),
            ]
        else:
            # Execute the cache mutation without loading/pushing a helper image.
            command += ["--output", "type=cacheonly"]
        subprocess.run(
            [*command, str(context)],
            check=True,
            stdout=sys.stderr,
            timeout=config.BUILD_TIMEOUT_S,
        )
        if args.action == "backup":
            digest = json.loads(result_path.read_text())["containerimage.digest"]
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("Docker returned an invalid snapshot digest")
            return digest_pinned_ref(args.image_ref, digest)
        return _ccache_mount_id(args.baseline_commit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backup", "restore"))
    parser.add_argument("--engine", required=True, choices=sorted(ENGINE_PRESETS))
    parser.add_argument(
        "--baseline-commit", required=True, help="Full target commit SHA"
    )
    parser.add_argument("--base-image", required=True, help="Digest-pinned build base")
    parser.add_argument(
        "--image-ref",
        required=True,
        help="Backup: destination tag. Restore: trusted digest-pinned snapshot.",
    )
    parser.add_argument(
        "--platform",
        choices=("linux/amd64", "linux/arm64"),
        default="linux/amd64",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.baseline_commit):
        parser.error(
            "--baseline-commit must be a full lowercase 40-character commit SHA"
        )
    if not re.fullmatch(_PINNED, args.base_image):
        parser.error(
            "--base-image must be pinned with @sha256:<64 lowercase hex digits>"
        )
    pattern = _TAGGED if args.action == "backup" else _PINNED
    if not re.fullmatch(pattern, args.image_ref):
        parser.error("--image-ref must be a tag for backup or a digest pin for restore")
    try:
        print(transfer(args))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Cache {args.action} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
