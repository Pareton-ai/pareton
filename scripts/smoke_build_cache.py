"""Test compiler-cache backup/restore with a tiny C program and disposable Docker resources.

Run from any directory: python /path/to/pareton/scripts/smoke_build_cache.py
Requires Docker Engine and Buildx. Pulls Alpine, registry and BuildKit from Docker
Hub; pushes only to a disposable loopback registry. No GPU or Python extras needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Synthetic commit labels; no engine source is downloaded or built.
SOURCE = hashlib.sha1(b"ccache smoke baseline").hexdigest()
FUTURE = hashlib.sha1(b"ccache smoke future baseline").hexdigest()


def run(*args: str, **kwargs) -> str:
    result = subprocess.run(
        args, check=True, text=True, stdout=subprocess.PIPE, timeout=600, **kwargs
    )
    return result.stdout.strip()


def main() -> None:
    name = "pareton-cache-smoke-" + uuid.uuid4().hex[:10]
    builders: list[str] = []
    registry_started = False
    architecture = run("docker", "info", "--format", "{{.Architecture}}")
    platform = {
        "aarch64": "linux/arm64",
        "arm64": "linux/arm64",
        "x86_64": "linux/amd64",
        "amd64": "linux/amd64",
    }[architecture]
    try:
        with tempfile.TemporaryDirectory(prefix=name) as directory:
            root = Path(directory)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            run(
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "-e",
                f"REGISTRY_HTTP_ADDR=0.0.0.0:{port}",
                "-p",
                f"127.0.0.1:{port}:{port}",
                "registry:2",
            )
            registry_started = True
            registry = f"localhost:{port}"
            buildkit_config = root / "buildkitd.toml"
            buildkit_config.write_text(f'[registry."{registry}"]\n  http = true\n')

            def create_builder(suffix: str) -> str:
                builder = f"{name}-{suffix}"
                # Separate daemons/volumes, never selected as the default. Sharing
                # the registry network makes its localhost ref work on Docker Desktop.
                run(
                    "docker",
                    "buildx",
                    "create",
                    "--name",
                    builder,
                    "--driver",
                    "docker-container",
                    "--driver-opt",
                    f"network=container:{name}",
                    "--buildkitd-config",
                    str(buildkit_config),
                )
                builders.append(builder)
                run("docker", "buildx", "inspect", builder, "--bootstrap")
                return builder

            def build(builder: str, context: Path, *options: str) -> None:
                run(
                    "docker",
                    "buildx",
                    "build",
                    "--builder",
                    builder,
                    "--platform",
                    platform,
                    "--progress=plain",
                    "--build-arg",
                    f"SMOKE_NONCE={uuid.uuid4().hex}",
                    *options,
                    str(context),
                )

            source = create_builder("source")
            base_context = root / "base"
            base_context.mkdir()
            (base_context / "Dockerfile").write_text(
                "FROM alpine:3.22\nRUN apk add --no-cache ccache gcc musl-dev\n"
            )
            base_tag = f"{registry}/base:test"
            result_file = root / "base.json"
            build(
                source,
                base_context,
                "--push",
                "--provenance=false",
                "-t",
                base_tag,
                "--metadata-file",
                str(result_file),
            )
            base = f"{registry}/base@{json.loads(result_file.read_text())['containerimage.digest']}"
            context = root / "compile"
            context.mkdir()
            (context / "hello.c").write_text(
                '#include <stdio.h>\nint main(void) { puts("cache works"); return 0; }\n'
            )

            def compile_c(builder: str, commit: str, hit: bool) -> None:
                access = "readonly" if hit else "sharing=locked"
                readonly = (
                    "CCACHE_READONLY=1 CCACHE_TEMPDIR=/tmp/ccache-tmp" if hit else ""
                )
                expected = "direct_cache_hit" if hit else "cache_miss"
                (context / "Dockerfile").write_text(f"""FROM {base}
ENV CCACHE_DIR=/root/.ccache CCACHE_NOHASHDIR=1 CCACHE_LOGFILE=/tmp/ccache.log
WORKDIR /src
COPY hello.c /src/hello.c
ARG SMOKE_NONCE
RUN --mount=type=cache,id=pareton-ccache-{commit},target=/root/.ccache,{access} \\
    {readonly} ccache gcc -c /src/hello.c -o /tmp/hello.o \\
    && gcc /tmp/hello.o -o /tmp/hello \\
    && test "$(/tmp/hello)" = "cache works" \\
    && cat /tmp/ccache.log \\
    && grep -q 'Result: {expected}' /tmp/ccache.log
""")
                build(builder, context, "--network=none", "--output", "type=cacheonly")

            def cache(
                action: str, builder: str, engine: str, commit: str, ref: str
            ) -> str:
                return run(
                    sys.executable,
                    "-m",
                    "builder.cache",
                    action,
                    "--engine",
                    engine,
                    "--baseline-commit",
                    commit,
                    "--base-image",
                    base,
                    "--platform",
                    platform,
                    "--image-ref",
                    ref,
                    cwd=REPO,
                    env={
                        **os.environ,
                        "PARETON_BUILDER_NAME": builder,
                        "PARETON_BUILDER_LOCK_PATH": str(root / "storage.lock"),
                        "PARETON_GHCR_USERNAME": "",
                        "PARETON_GHCR_TOKEN": "",
                    },
                )

            compile_c(source, SOURCE, hit=False)
            snapshots = {}
            for engine in ("vllm", "sglang"):
                snapshots[engine] = cache(
                    "backup", source, engine, SOURCE, f"{registry}/ccache:{engine}"
                )
                print(
                    f"PASS {engine}: backup published {snapshots[engine]}", flush=True
                )
            # Delete the source daemon and its cache volume before any restore.
            run("docker", "buildx", "rm", source)
            builders.remove(source)
            print("Source builder and local compiler cache removed.", flush=True)
            for engine, snapshot in snapshots.items():
                for commit in (SOURCE, FUTURE):
                    target = create_builder(f"{engine}-{commit[:7]}")
                    cache("restore", target, engine, commit, snapshot)
                    compile_c(target, commit, hit=True)
                    print(
                        f"PASS {engine}: restored {'same' if commit == SOURCE else 'future'} "
                        "commit, direct cache hit, executable output verified",
                        flush=True,
                    )
                    run("docker", "buildx", "rm", target)
                    builders.remove(target)
            print(
                "PASS: both engines, backup and restore, four fresh-builder cache hits.",
                flush=True,
            )
    finally:
        for builder in builders:
            subprocess.run(
                ["docker", "buildx", "rm", builder], check=False, timeout=120
            )
        if registry_started:
            subprocess.run(["docker", "stop", name], check=False, timeout=120)


if __name__ == "__main__":
    main()
