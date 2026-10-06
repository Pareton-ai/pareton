"""Opt-in PRO6000 model-volume transport for qualification and v5 shadow runs.

Adapted from PR #189's DockerModelVolume. This operator-only module copies weights
through Docker's API when the client and daemon do not share bind-mount paths.
It does not change the production harness, workload, scoring or serving flags.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

from bench.lifecycle import EngineError
from bench.validate import load_bench_request, sha256_file


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


class DockerModelVolume:
    """Copy locally verified weights through Docker's API, avoiding host binds."""

    def __init__(self, root):
        self.root = root
        self.name = "pareton-pro6000-model-" + uuid4().hex
        self.source = None
        self.created = False

    def command(self, *args):
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=7200, check=False
        )
        if result.returncode:
            raise EngineError(f"model volume {args[0]} failed: {result.stderr}")
        return result.stdout

    def prepare(self, source, image):
        source = source.resolve()
        if self.source is not None:
            if source != self.source:
                raise EngineError("model path changed during shadow round")
            return
        print(
            "Copying staged weights into a Docker-managed model volume...", flush=True
        )
        expected = {
            str(path.relative_to(source)): sha256_file(path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        }
        if not expected or any(path.is_symlink() for path in source.rglob("*")):
            raise EngineError(
                "model volume requires nonempty, symlink-free staged weights"
            )
        save(
            self.root / "model_volume.json",
            {
                "name": self.name,
                "source": str(source),
                "verified": False,
            },
        )
        self.command("pull", image)
        self.command("volume", "create", self.name)
        self.created = True
        helper = self.name + "-copy"
        try:
            self.command(
                "create",
                "--name",
                helper,
                "--network",
                "none",
                "--mount",
                f"type=volume,src={self.name},dst=/model,volume-nocopy",
                "--entrypoint",
                "/bin/true",
                image,
            )
            self.command("cp", str(source) + "/.", helper + ":/model")
        finally:
            self.command("rm", "-f", helper)
        # Read back every copied file inside the daemon's mount namespace.
        verify = """import hashlib, json, pathlib
root = pathlib.Path('/model')
result = {}
for path in sorted(root.rglob('*')):
    if path.is_symlink():
        raise RuntimeError('symlink in copied model volume')
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        result[str(path.relative_to(root))] = "sha256:" + digest.hexdigest()
print(json.dumps(result))
"""
        actual = json.loads(
            self.command(
                "run",
                "--rm",
                "--network",
                "none",
                "--mount",
                f"type=volume,src={self.name},dst=/model,readonly,volume-nocopy",
                "--entrypoint",
                "python3",
                image,
                "-c",
                verify,
            )
        )
        if actual != expected:
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            mismatched = sorted(
                key
                for key in expected.keys() & actual.keys()
                if expected[key] != actual[key]
            )
            save(
                self.root / "model_volume.json",
                {
                    "name": self.name,
                    "source": str(source),
                    "verified": False,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "missing": missing,
                    "extra": extra,
                    "mismatched": mismatched,
                },
            )
            raise EngineError(
                "Docker model volume file hashes differ from staged weights: "
                f"{len(missing)} missing, {len(extra)} extra, "
                f"{len(mismatched)} mismatched; see model_volume.json"
            )
        self.source = source
        save(
            self.root / "model_volume.json",
            {
                "name": self.name,
                "source": str(source),
                "sha256": expected,
                "verified": True,
            },
        )
        print("Docker model volume verified; starting engine lifecycle.", flush=True)

    @property
    def mount(self):
        if self.source is None:
            raise EngineError("model volume has not been verified")
        return f"type=volume,src={self.name},dst=/model,readonly,volume-nocopy"

    def wrap_runner(self, runner):
        if self.source is None:
            raise EngineError("model volume has not been verified")

        def run(cmd, **kwargs):
            cmd = list(cmd)
            if cmd[:2] == ["docker", "run"]:
                old = f"{self.source}:/model:ro"
                matches = [
                    i
                    for i in range(1, len(cmd))
                    if cmd[i - 1] == "-v" and cmd[i] == old
                ]
                if len(matches) != 1:
                    raise EngineError("expected exactly one staged model bind mount")
                i = matches[0]
                cmd[i - 1 : i + 1] = [
                    "--mount",
                    f"type=volume,src={self.name},dst=/model,readonly,volume-nocopy",
                ]
            return runner(cmd, **kwargs)

        return run

    def close(self):
        if self.created:
            self.command("volume", "rm", self.name)
            self.created = False


def run_shadow(request_path, output_dir):
    import bench.main as harness

    req, _ = load_bench_request(request_path)
    if (
        req.hardware.gpu_count != 1
        or req.hardware.gpu_sku_expected != "RTXPRO6000"
        or req.model.hf_repo != "Qwen/Qwen3.8-27B-FP8"
        or req.model.quantization != "fp8"
        or req.engines.baseline.name != "sglang"
        or any(engine.name != "sglang" for engine in req.engines.candidates)
    ):
        raise EngineError("volume shadow helper requires the TP1 PRO6000 FP8 request")
    if os.environ.get("PARETON_BENCH_ENGINE_CACHE_DIR", "").strip():
        raise EngineError(
            "unset PARETON_BENCH_ENGINE_CACHE_DIR for this portable run; "
            "the helper replaces only the /model bind mount"
        )
    if output_dir.exists():
        raise EngineError("use a fresh shadow output directory")
    volume = DockerModelVolume(output_dir)
    original = harness.EngineContainer

    class VolumeContainer(original):
        def __enter__(self):
            if self.weights_dir is None:
                raise EngineError("shadow engine has no staged model weights")
            volume.prepare(self.weights_dir, self.spec.image)
            self.runner = volume.wrap_runner(self.runner)
            return super().__enter__()

    # Only this explicit CLI process replaces the model transport. The worker's
    # request, v5 trace, normal EOS, C4 scheduling and scorer remain untouched.
    try:
        harness.EngineContainer = VolumeContainer
        return harness.run_bench(request_path, output_dir)
    finally:
        harness.EngineContainer = original
        volume.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_shadow(args.request, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
