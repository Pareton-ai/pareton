"""Docker integration tests for bench.lifecycle.

Skipped automatically when Docker is unavailable. Run with:
  pytest tests/test_lifecycle_docker.py -m docker
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

from bench.lifecycle import (
    NAME_PREFIX,
    BenchNetwork,
    EngineContainer,
    default_docker_runner,
)
from bench.mock_engine import post_completion
from bench.schemas import EngineSpec

ROOT = Path(__file__).resolve().parents[1]
MOCK_IMAGE = "pareton-mock-engine:local"
DOCKERFILE = ROOT / "images" / "mock-engine" / "Dockerfile"


def _docker_available() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _docker_available(), reason="docker not available"),
]


def _names_matching_prefix(kind: str) -> list[str]:
    """kind: 'container' | 'network' — return names starting with NAME_PREFIX."""
    if kind == "container":
        r = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    else:
        r = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    return [n for n in r.stdout.splitlines() if n.startswith(NAME_PREFIX)]


@pytest.fixture(scope="module")
def mock_engine_image() -> str:
    build = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(DOCKERFILE),
            "-t",
            MOCK_IMAGE,
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if build.returncode != 0:
        pytest.fail(f"mock engine image build failed:\n{build.stderr[-4000:]}")
    return MOCK_IMAGE


def _host_can_reach(ip: str, port: int = 8000) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=1.0):
            return True
    except OSError:
        return False


def test_publish_port_lifecycle_healthy_and_cleanup(
    mock_engine_image: str, tmp_path: Path
):
    """Works on Docker Desktop and Linux: random host port + completions."""
    run_id = "itpub" + "0" * 7
    logs = tmp_path / "logs"
    spec = EngineSpec(image=mock_engine_image, serve_args=[], env={})

    before_c = set(_names_matching_prefix("container"))
    before_n = set(_names_matching_prefix("network"))

    with BenchNetwork(
        run_id=run_id, internal=False, runner=default_docker_runner, cmd_timeout_s=60
    ) as net:
        with EngineContainer(
            spec=spec,
            network=net,
            role="baseline",
            pull=False,
            publish_port=True,
            port=8000,
            logs_dir=logs,
            health_timeout_s=30,
            health_poll_s=0.5,
            cmd_timeout_s=60,
        ) as handle:
            assert handle.base_url.startswith("http://127.0.0.1:")
            assert handle.image_digest.startswith("sha256:")
            # Local image → Id fallback (not a registry RepoDigest).
            resp = post_completion(
                f"{handle.base_url}/v1/completions",
                prompt="Hello world",
                max_tokens=1,
                echo=True,
                logprobs=1,
            )
            assert resp["object"] == "text_completion"
            assert resp["choices"][0]["logprobs"]["token_logprobs"][0] is None

    after_c = set(_names_matching_prefix("container"))
    after_n = set(_names_matching_prefix("network"))
    assert not (after_c - before_c), f"leaked containers: {after_c - before_c}"
    assert not (after_n - before_n), f"leaked networks: {after_n - before_n}"
    assert list(logs.glob("*.log")), "expected container log captured"


def test_internal_network_blocks_egress(mock_engine_image: str):
    """Outbound HTTPS from an --internal network container must fail."""
    run_id = "itneg" + "0" * 7
    # Pull alpine once so the probe itself does not need egress from the
    # internal network (image must already be local).
    pull = subprocess.run(
        ["docker", "pull", "alpine:3.20"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if pull.returncode != 0:
        pytest.skip(f"could not pull alpine:3.20 for egress probe: {pull.stderr}")

    with BenchNetwork(
        run_id=run_id, internal=True, runner=default_docker_runner, cmd_timeout_s=60
    ) as net:
        probe = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                net.name,
                "alpine:3.20",
                "wget",
                "-q",
                "-T",
                "3",
                "-O",
                "-",
                "https://example.com",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert probe.returncode != 0, (
            "expected egress blocked on --internal network, "
            f"but wget succeeded: {probe.stdout[:200]}"
        )

    assert not [n for n in _names_matching_prefix("network") if run_id in n], (
        "leaked network after egress test"
    )


def test_internal_network_lifecycle_when_ip_reachable(
    mock_engine_image: str, tmp_path: Path
):
    """Full internal-network path on Linux; self-skip on Docker Desktop."""
    run_id = "itlip" + "0" * 7
    # Preflight: start a short-lived container on an internal net and probe IP.
    with BenchNetwork(
        run_id=run_id + "p",
        internal=True,
        runner=default_docker_runner,
        cmd_timeout_s=60,
    ) as probe_net:
        run = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                f"{NAME_PREFIX}{run_id}p-probe",
                "--network",
                probe_net.name,
                mock_engine_image,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if run.returncode != 0:
            pytest.fail(f"probe container failed: {run.stderr}")
        cid = run.stdout.strip()
        try:
            ip_r = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    f'{{{{index .NetworkSettings.Networks "{probe_net.name}" "IPAddress"}}}}',
                    cid,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            ip = ip_r.stdout.strip()
            # Give the mock a moment to bind.
            import time

            time.sleep(1.0)
            if not ip or not _host_can_reach(ip, 8000):
                pytest.skip(
                    "container-bridge IP not reachable from host "
                    "(Docker Desktop); covered by publish_port test"
                )
        finally:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

    with BenchNetwork(
        run_id=run_id, internal=True, runner=default_docker_runner, cmd_timeout_s=60
    ) as net:
        with EngineContainer(
            spec=EngineSpec(image=mock_engine_image, serve_args=[], env={}),
            network=net,
            role="baseline",
            pull=False,
            publish_port=False,
            logs_dir=tmp_path,
            health_timeout_s=30,
            health_poll_s=0.5,
            cmd_timeout_s=60,
        ) as handle:
            assert handle.image_digest.startswith("sha256:")
            resp = post_completion(
                f"{handle.base_url}/v1/completions",
                prompt="ok",
                max_tokens=1,
            )
            assert resp["object"] == "text_completion"

    assert not [n for n in _names_matching_prefix("container") if run_id in n], (
        "leaked containers for run_id"
    )
    assert not [n for n in _names_matching_prefix("network") if run_id in n], (
        "leaked network for run_id"
    )


def test_two_publish_port_engines_no_port_collision(
    mock_engine_image: str, tmp_path: Path
):
    """Baseline + candidate concurrently must get distinct host ports."""
    run_id = "it2en" + "0" * 7
    spec = EngineSpec(image=mock_engine_image, serve_args=[], env={})
    with BenchNetwork(
        run_id=run_id, internal=False, runner=default_docker_runner, cmd_timeout_s=60
    ) as net:
        with (
            EngineContainer(
                spec=spec,
                network=net,
                role="baseline",
                pull=False,
                publish_port=True,
                logs_dir=tmp_path,
                health_timeout_s=30,
                health_poll_s=0.5,
                cmd_timeout_s=60,
            ) as base,
            EngineContainer(
                spec=spec,
                network=net,
                role="candidate",
                pull=False,
                publish_port=True,
                logs_dir=tmp_path,
                health_timeout_s=30,
                health_poll_s=0.5,
                cmd_timeout_s=60,
            ) as cand,
        ):
            assert base.base_url != cand.base_url
            assert base.container_name != cand.container_name
            post_completion(f"{base.base_url}/v1/completions", prompt="a", max_tokens=1)
            post_completion(f"{cand.base_url}/v1/completions", prompt="b", max_tokens=1)
