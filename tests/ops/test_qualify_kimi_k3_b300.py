"""CPU-only tests for the Kimi K3 B300 qualification runner."""

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "qualify_kimi_k3_b300", ROOT / "ops/qualify_kimi_k3_b300.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
pytestmark = pytest.mark.unit

FIELDS = ROOT / "fixtures/campaigns/sglang_kimi_k3_b300/campaign-fields.json"


def fields():
    return json.loads(FIELDS.read_text())


def smi(names):
    return lambda *a, **k: SimpleNamespace(stdout="\n".join(names) + "\n")


def test_fixture_is_accepted_and_qualified_rules_are_refused():
    runner.check_fields(fields())
    qualified = fields()
    qualified["sampling_rule"]["eligible_row_indices"] = [1, 2]
    with pytest.raises(ValueError, match="unqualified"):
        runner.check_fields(qualified)
    other = fields()
    other["bench"]["model"]["hf_repo"] = "Qwen/Qwen3.8-27B-FP8"
    with pytest.raises(ValueError, match="Kimi"):
        runner.check_fields(other)


def test_gpu_check_requires_eight_b300_and_nothing_else():
    names = ["NVIDIA B300 SXM6 AC"] * 8
    assert runner.check_gpus("B300", 8, runner=smi(names)) == names
    with pytest.raises(runner.EngineError):
        runner.check_gpus("B300", 8, runner=smi(names[:7]))
    with pytest.raises(runner.EngineError):
        runner.check_gpus("B300", 8, runner=smi([*names, "NVIDIA H200"]))


def test_baseline_start_matches_the_round(tmp_path):
    req, start = runner.baseline_request(fields(), tmp_path)
    args = start.spec.serve_args
    assert args[:4] == ["--model-path", "/model", "--context-length", "1048576"]
    assert args.count("--model-path") == 1
    assert args[args.index("--speculative-draft-model-path") + 1] == "/draft"
    assert args[args.index("--mem-fraction-static") + 1] == "0.88"
    assert req.hardware.gpu_count == 8
    assert req.draft_model.hf_repo == "RadixArk/Kimi-K3-DSpark"


def test_main_stages_both_models_and_qualifies_the_published_baseline(
    monkeypatch, tmp_path
):
    staged, containers, qualified = [], [], []
    monkeypatch.setattr(runner, "check_gpus", lambda sku, count: [sku] * count)

    def stage(model, **kwargs):
        staged.append((model.hf_repo, kwargs.get("require_tokenizer", True)))
        return SimpleNamespace(path=tmp_path / model.hf_repo.replace("/", "--"))

    @contextmanager
    def network(**kwargs):
        yield SimpleNamespace(internal=kwargs["internal"])

    class Container:
        def __init__(self, **kwargs):
            containers.append(kwargs)

        def __enter__(self):
            return SimpleNamespace(base_url="http://127.0.0.1:41234", container_id="c1")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(runner, "stage_weights", stage)
    monkeypatch.setattr(runner, "BenchNetwork", network)
    monkeypatch.setattr(runner, "EngineContainer", Container)
    monkeypatch.setattr(runner, "qualify", lambda **kw: qualified.append(kw))
    out = tmp_path / "run"
    assert (
        runner.main(["--output-dir", str(out), "--campaign-fields", str(FIELDS)]) == 0
    )
    assert staged == [
        ("moonshotai/Kimi-K3", True),
        ("RadixArk/Kimi-K3-DSpark", False),
    ]
    (container,) = containers
    assert container["publish_port"] is True and container["gpu_count"] == 8
    assert container["network"].internal is False
    assert container["draft_dir"].name == "RadixArk--Kimi-K3-DSpark"
    (call,) = qualified
    assert call["base_url"] == "http://127.0.0.1:41234"
    assert call["container"] == "c1"
    assert call["engine_ref"] == fields()["bench"]["baseline_engine_image_digest"]
    assert (call["pool_size"], call["repetitions"], call["concurrency"]) == (32, 2, 4)
    assert call["output_dir"] == out.resolve() / "qualification"


def test_main_reports_failures_without_a_rule(monkeypatch, tmp_path, capsys):
    def no_gpus(sku, count):
        raise runner.EngineError("need 8 B300 GPUs")

    monkeypatch.setattr(runner, "check_gpus", no_gpus)
    assert runner.main(["--output-dir", str(tmp_path / "run")]) == 1
    assert "need 8 B300" in capsys.readouterr().err
