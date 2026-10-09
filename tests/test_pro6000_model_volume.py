"""Operator-only Docker volume transport; Docker is mocked, not runtime-tested."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.lifecycle import EngineError

spec = importlib.util.spec_from_file_location(
    "pro6000_model_volume",
    Path(__file__).resolve().parents[1] / "ops/pro6000_model_volume.py",
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
DockerModelVolume = helper.DockerModelVolume

pytestmark = pytest.mark.unit


def test_model_volume_copies_verifies_reuses_and_rewrites(tmp_path, monkeypatch):
    source = tmp_path / "weights"
    source.mkdir()
    (source / "config.json").write_text("{}")
    volume = DockerModelVolume(tmp_path)
    calls = []

    def command(*args):
        calls.append(args)
        if args[0] == "run":
            # Execute the actual container-side hashing code on a local fixture.
            # A canned expected response would hide hash-format differences.
            script = args[-1].replace(
                "pathlib.Path('/model')", f"pathlib.Path({str(source)!r})"
            )
            return subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        return ""

    monkeypatch.setattr(volume, "command", command)
    volume.prepare(source, "pinned-image")
    volume.prepare(source, "pinned-image")
    assert sum(c[0] == "cp" for c in calls) == 1
    assert ("cp", str(source) + "/.", volume.name + "-copy:/model") in calls
    assert json.loads((tmp_path / "model_volume.json").read_text())["verified"]
    seen = []
    runner = volume.wrap_runner(lambda cmd, **kw: seen.append(cmd))
    runner(["docker", "run", "-v", f"{source}:/model:ro", "pinned-image"])
    assert seen[0][2:] == [
        "--mount",
        f"type=volume,src={volume.name},dst=/model,readonly,volume-nocopy",
        "pinned-image",
    ]
    runner(["docker", "inspect", "engine"])
    assert seen[-1] == ["docker", "inspect", "engine"]
    with pytest.raises(EngineError, match="exactly one"):
        runner(["docker", "run", "image"])
    volume.close()
    assert calls[-1] == ("volume", "rm", volume.name)


def test_model_volume_rejects_mismatch_and_cleans_up(tmp_path, monkeypatch):
    source = tmp_path / "weights"
    source.mkdir()
    (source / "config.json").write_text("{}")
    volume = DockerModelVolume(tmp_path)
    calls = []

    def command(*args):
        calls.append(args)
        return "{}" if args[0] == "run" else ""

    monkeypatch.setattr(volume, "command", command)
    with pytest.raises(EngineError, match="hashes differ"):
        volume.prepare(source, "image")
    assert volume.source is None
    evidence = json.loads((tmp_path / "model_volume.json").read_text())
    assert evidence["verified"] is False
    assert evidence["missing"] == ["config.json"]
    assert evidence["extra"] == evidence["mismatched"] == []
    volume.close()
    assert calls[-1] == ("volume", "rm", volume.name)


@pytest.mark.parametrize("invalid", ["empty", "symlink"])
def test_model_volume_rejects_unsafe_source_before_docker(
    tmp_path, monkeypatch, invalid
):
    source = tmp_path / "source"
    source.mkdir()
    if invalid == "symlink":
        (tmp_path / "weights").write_text("data")
        (source / "weights").symlink_to(tmp_path / "weights")
    volume = DockerModelVolume(tmp_path)
    monkeypatch.setattr(volume, "command", lambda *a: pytest.fail("Docker called"))
    with pytest.raises(EngineError, match="nonempty, symlink-free"):
        volume.prepare(source, "image")
    with pytest.raises(EngineError, match="not been verified"):
        _ = volume.mount


@pytest.mark.parametrize("fail", [False, True])
def test_shadow_uses_volume_only_and_restores_harness(tmp_path, monkeypatch, fail):
    import bench.main as harness

    monkeypatch.delenv("PARETON_BENCH_ENGINE_CACHE_DIR", raising=False)
    source = tmp_path / "weights"
    source.mkdir()
    (source / "config.json").write_text("{}")
    output = tmp_path / "shadow"
    request_path = tmp_path / "request.json"
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json"
    )
    fields = json.loads(fixture.read_text())
    # These pins are passed through untouched to the actual run_bench entry point.
    request = SimpleNamespace(
        hardware=SimpleNamespace(gpu_count=1, gpu_sku_expected="RTXPRO6000"),
        model=SimpleNamespace(hf_repo="Qwen/Qwen3.8-27B-FP8", quantization="fp8"),
        engines=SimpleNamespace(baseline=SimpleNamespace(name="sglang"), candidates=[]),
    )
    monkeypatch.setattr(helper, "load_bench_request", lambda path: (request, {}))
    calls = []

    def command(self, *args):
        calls.append(args)
        if args[0] == "run":
            return json.dumps(
                {"config.json": helper.sha256_file(source / "config.json")}
            )
        return ""

    monkeypatch.setattr(DockerModelVolume, "command", command)
    launches = []

    class Container:
        def __init__(self):
            self.weights_dir = source
            self.spec = SimpleNamespace(
                image=fields["base_image_digest"],
                serve_args=fields["bench"]["serve_args"],
            )
            self.runner = lambda cmd, **kw: launches.append(cmd)

        def __enter__(self):
            self.runner(
                [
                    "docker",
                    "run",
                    "-v",
                    f"{source}:/model:ro",
                    self.spec.image,
                    *self.spec.serve_args,
                ]
            )
            return self

    monkeypatch.setattr(harness, "EngineContainer", Container)

    def run_bench(path, out):
        assert (path, out) == (request_path, output)
        for _ in range(2):
            harness.EngineContainer().__enter__()
        if fail:
            raise EngineError("test startup failure")
        return 3  # Preserve harness nonzero results as well as exceptions.

    monkeypatch.setattr(harness, "run_bench", run_bench)
    if fail:
        with pytest.raises(EngineError, match="test startup failure"):
            helper.run_shadow(request_path, output)
    else:
        assert helper.run_shadow(request_path, output) == 3
    assert harness.EngineContainer is Container
    assert sum(c[0] == "cp" for c in calls) == 1
    assert calls[-1][:2] == ("volume", "rm")
    for launch in launches:
        assert "-v" not in launch
        assert "readonly,volume-nocopy" in launch[3]
        assert launch[5:] == fields["bench"]["serve_args"]


def test_shadow_rejects_engine_cache_bind_before_running(tmp_path, monkeypatch):
    monkeypatch.setenv("PARETON_BENCH_ENGINE_CACHE_DIR", "/invisible/cache")
    req = SimpleNamespace(
        hardware=SimpleNamespace(gpu_count=1, gpu_sku_expected="RTXPRO6000"),
        model=SimpleNamespace(hf_repo="Qwen/Qwen3.8-27B-FP8", quantization="fp8"),
        engines=SimpleNamespace(baseline=SimpleNamespace(name="sglang"), candidates=[]),
    )
    monkeypatch.setattr(helper, "load_bench_request", lambda p: (req, {}))
    with pytest.raises(EngineError, match="unset PARETON_BENCH_ENGINE_CACHE_DIR"):
        helper.run_shadow(tmp_path / "request.json", tmp_path / "out")
