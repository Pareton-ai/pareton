"""Check the selected local Buildx builder's actual GC policy before starting."""

from __future__ import annotations

import re
import subprocess

try:
    import tomllib
except ImportError:  # Python 3.10 developer environments
    import tomli as tomllib

import config
from builder.gc_config import validate_daemon_gc_file


def _output(*args: str) -> str:
    return subprocess.check_output(args, text=True, timeout=120).strip()


def validate_builder() -> None:
    inspection = _output(
        "docker", "buildx", "inspect", config.BUILDER_NAME, "--bootstrap"
    )
    # Buildx 0.28 (bundled with the pinned CLI) has no inspect --format flag.
    driver_match = re.search(r"^Driver:\s*(\S+)", inspection, re.MULTILINE)
    if driver_match is None:
        raise ValueError("Buildx inspect did not report its driver")
    driver = driver_match[1]
    if driver == "docker":
        validate_daemon_gc_file(config.DOCKER_DAEMON_CONFIG_PATH)
    elif driver == "docker-container":
        nodes_section = inspection.partition("Nodes:")[2]
        nodes = re.findall(r"^Name:\s*(\S+)", nodes_section, re.MULTILINE)
        if len(nodes) != 1:
            raise ValueError("Pareton requires a single local Buildx node")
        # Buildx mounts this file when --buildkitd-config is supplied. A
        # missing file is a failure: the default BuildKit GC is unsafe here.
        text = _output(
            "docker",
            "exec",
            f"buildx_buildkit_{nodes[0]}",
            "cat",
            "/etc/buildkit/buildkitd.toml",
        )
        data = tomllib.loads(text)
        for worker in ("oci", "containerd"):
            if data.get("worker", {}).get(worker, {}).get("gc") is not False:
                raise ValueError(f"BuildKit worker.{worker}.gc must be false")
    else:
        raise ValueError(f"unsupported builder driver {driver!r}; use a local builder")


def main() -> None:
    validate_builder()
    print(f"builder {config.BUILDER_NAME}: application-managed GC verified", flush=True)


if __name__ == "__main__":
    main()
