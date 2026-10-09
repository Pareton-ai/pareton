"""CPU-only tests for the temperature-extremes logprob check."""

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.correctness import _PositionScore
from bench.longform import length_groups
from bench.trajectory import token_ids_sha256

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "temperature_logprob_check", ROOT / "ops/temperature_logprob_check.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
pytestmark = pytest.mark.unit

FIELDS = ROOT / "fixtures/campaigns/sglang_kimi_k3_b300/campaign-fields.json"


def trace():
    groups = length_groups(4)
    return {
        "schema_version": 1,
        "meta": {"name": "test"},
        "requests": [
            {
                "id": "hf-0",
                "prompt": "Write a long story.",
                "input_tokens": groups[2]["max_tokens"],
                "input_length_group": groups[2]["name"],
                "input_ids_sha256": token_ids_sha256([1]),
                "arrival_offset_ms": 0,
                "max_tokens": 5120,
                "sampling": {"temperature": 0.5, "top_p": 1.0},
            }
        ],
    }


@pytest.fixture
def harness(monkeypatch, tmp_path):
    staged, starts, completions = [], [], []
    monkeypatch.setattr(
        check, "sample_prompt", lambda f, i: (trace(), trace()["requests"][0])
    )

    def stage(model, **kwargs):
        staged.append((model.hf_repo, kwargs.get("require_tokenizer", True)))
        path = tmp_path / model.hf_repo.replace("/", "--")
        path.mkdir(exist_ok=True)
        return SimpleNamespace(path=path)

    monkeypatch.setattr(check, "stage_weights", stage)

    @contextmanager
    def start(self, engine_start, *, phase):
        starts.append((engine_start.kind, list(engine_start.spec.serve_args)))
        assert self.draft_dir is not None
        yield f"http://{engine_start.kind}"

    monkeypatch.setattr(check._EngineProvider, "start", start)

    def complete(url, **kwargs):
        assert url == "http://baseline"
        completions.append(kwargs)
        return {
            "choices": [{"text": "word " * 10, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10},
        }

    monkeypatch.setattr(check, "post_completion", complete)
    scores = {"value": -1.0}

    def score(url, captured, **kwargs):
        assert url == "http://scorer" and kwargs["engine_name"] == "sglang"
        value = scores["value"]
        if callable(value):
            value = value(captured)
        return [_PositionScore(i, "w", -1, value, None) for i in range(10)], 10, ""

    monkeypatch.setattr(check, "score_captured_output", score)
    return SimpleNamespace(
        staged=staged, starts=starts, completions=completions, scores=scores
    )


def run(tmp_path, name="out"):
    out = tmp_path / name
    code = check.main(["--campaign-fields", str(FIELDS), "--output-dir", str(out)])
    return code, json.loads((out / "summary.json").read_text())


def test_ten_samples_at_each_temperature_extreme_pass(tmp_path, harness):
    code, summary = run(tmp_path)
    assert code == 0, summary
    assert summary["status"] == "passed"
    assert summary["temperatures"] == [0.1, 1.01]
    assert [(c["temperature"], c["seed"]) for c in harness.completions] == [
        (t, s) for t in (0.1, 1.01) for s in range(10)
    ]
    assert all(c["max_tokens"] == 5120 for c in harness.completions)
    assert harness.staged == [
        ("moonshotai/Kimi-K3", True),
        ("RadixArk/Kimi-K3-DSpark", False),
    ]
    kinds = [kind for kind, _ in harness.starts]
    assert kinds == ["baseline", "scorer"]
    scorer_args = harness.starts[1][1]
    fractions = [
        scorer_args[i + 1]
        for i, a in enumerate(scorer_args)
        if a == "--mem-fraction-static"
    ]
    assert fractions[-1] == "0.80"
    assert summary["by_temperature"]["0.1"]["passed"] == 10
    assert summary["by_temperature"]["1.01"]["passed"] == 10


def test_one_low_scoring_sample_fails_the_check(tmp_path, harness):
    harness.scores["value"] = lambda c: -20.0 if c.request_id == "t1.01-r3" else -1.0
    code, summary = run(tmp_path)
    assert code == 1
    assert summary["status"] == "failed_thresholds"
    assert summary["failed_samples"] == ["t1.01-r3"]
    results = json.loads((tmp_path / "out/results.json").read_text())
    failed = next(r for r in results if r["request_id"] == "t1.01-r3")
    assert "mean logprob" in failed["reason"]
    assert "quantile" in failed["reason"]
    assert failed["raw_min_below_min_token_logprob"] is True


def test_existing_output_directory_is_never_reused(tmp_path, harness):
    assert run(tmp_path)[0] == 0
    with pytest.raises(FileExistsError):
        run(tmp_path)
