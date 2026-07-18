"""Docker integration: Module A via B3 lifecycle + containerized mock engines.

Skipped automatically when Docker is unavailable. Self-skips on Docker Desktop
where the host cannot reach container-bridge IPs (production path uses
``--internal`` + container IP; no published ports).
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

import pytest

from bench.lifecycle import NAME_PREFIX, BenchNetwork, default_docker_runner
from bench.main import EXIT_OK, main
from bench.validate import validate_report_dict

ROOT = Path(__file__).resolve().parents[1]
MOCK_IMAGE = "pareton-mock-engine:local"
DOCKERFILE = ROOT / "images" / "mock-engine" / "Dockerfile"
SAMPLE_REQUEST = ROOT / "fixtures" / "bench" / "sample_request.json"
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"


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


def _host_can_reach(ip: str, port: int = 8000) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def mock_image_digest() -> str:
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
    insp = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", MOCK_IMAGE],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if insp.returncode != 0:
        pytest.fail(f"docker inspect failed: {insp.stderr}")
    image_id = insp.stdout.strip()
    if not image_id.startswith("sha256:"):
        image_id = f"sha256:{image_id}"
    # Canonical 64-hex form for request validation.
    hex_part = image_id.split(":", 1)[1]
    if len(hex_part) > 64:
        # Docker may return untruncated id; digest pin needs 64 hex.
        hex_part = hex_part[:64]
    return f"sha256:{hex_part}"


def _require_container_ip_reachable(mock_image_digest: str) -> None:
    """Skip when host cannot reach bridge IPs (Docker Desktop)."""
    run_id = "itpre" + "0" * 7
    with BenchNetwork(
        run_id=run_id,
        internal=True,
        runner=default_docker_runner,
        cmd_timeout_s=60,
    ) as net:
        run = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                f"{NAME_PREFIX}{run_id}-probe",
                "--network",
                net.name,
                mock_image_digest,
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
                    f'{{{{index .NetworkSettings.Networks "{net.name}" "IPAddress"}}}}',
                    cid,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            ip = ip_r.stdout.strip()
            time.sleep(1.0)
            if not ip or not _host_can_reach(ip, 8000):
                pytest.skip(
                    "container-bridge IP not reachable from host "
                    "(Docker Desktop); production path is Linux GPU pods"
                )
        finally:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )


def test_cli_correctness_via_lifecycle_baseline_vs_baseline(
    mock_image_digest: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Docker lifecycle Module A self-check (no real HF download)."""
    from bench.weights import StagedWeights

    def fake_stage(model, *, token_env="HF_TOKEN", cache_dir=None):
        # Mock engines do not load HF weights; avoid multi-GB staging here.
        root = tmp_path / "staged-weights"
        root.mkdir(exist_ok=True)
        return StagedWeights(
            path=root,
            weights_sha256="sha256:" + ("d" * 64),
            num_files=1,
            total_bytes=1,
            manifest={
                "repo": model.hf_repo,
                "revision": model.hf_revision,
                "files": [],
            },
        )

    monkeypatch.setattr("bench.main.stage_weights", fake_stage)
    _require_container_ip_reachable(mock_image_digest)

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["mode"] = "correctness"
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    # Same local mock image for both sides (self-check through Docker).
    for role in ("baseline", "candidate"):
        req["engines"][role]["image"] = mock_image_digest
        req["engines"][role]["serve_args"] = []
        req["engines"][role]["env"] = {}
    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(req, indent=2) + "\n", encoding="utf-8")
    out = tmp_path / "out"

    code = main(["--request", str(req_path), "--output-dir", str(out)])
    assert code == EXIT_OK, (
        (out / "harness.log").read_text(encoding="utf-8")[-2000:]
        if (out / "harness.log").is_file()
        else "no harness log"
    )
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    validate_report_dict(report)
    assert report["verdict"] == "pass"
    assert report["correctness"]["verdict"] == "pass"
    assert report["inputs_fingerprint"]["model_weights_sha256"] == (
        "sha256:" + ("d" * 64)
    )
    assert (out / "evidence" / "correctness" / "logprob_diffs.jsonl").is_file()
