"""CPU-only tests for the pre-launch campaign checks."""

import importlib.util
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from bench.correctness import _PositionScore
from bench.http import StreamResult
from bench.longform import qualification_contract
from bench.sampler import parse_sampling_rule

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "campaign_preflight", ROOT / "ops/campaign_preflight.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
pytestmark = pytest.mark.unit

FIELDS = ROOT / "fixtures/campaigns/sglang_kimi_k3_b300/campaign-fields.json"
INPUT_TOKENS = 16000


@pytest.fixture(autouse=True)
def restore_health_timeout(monkeypatch):
    monkeypatch.setattr(config, "BENCH_HEALTH_TIMEOUT_S", config.BENCH_HEALTH_TIMEOUT_S)


def fields():
    return json.loads(FIELDS.read_text())


def qualified_rule():
    data = fields()
    rule = dict(
        parse_sampling_rule(data["sampling_rule"]),
        eligible_row_indices=list(range(32)),
    )
    rule["qualification"] = {
        "contract_sha256": qualification_contract(rule, data["bench"], data["engine"]),
        "repetitions": 2,
    }
    return rule


def request():
    return {
        "id": "hf-7",
        "prompt": "Write a long story.",
        "input_tokens": INPUT_TOKENS,
        "input_length_group": "16k",
        "max_tokens": 5120,
        "sampling": {"temperature": 0.5, "top_p": 1.0},
    }


@pytest.fixture
def harness(monkeypatch, tmp_path):
    h = SimpleNamespace(
        staged=[],
        containers=[],
        qualified=[],
        streams=[],
        starts=[],
        ttft_s=0.5,
        greedy_text=lambda label: "a b c d",
        scores=-1.0,
        lock=threading.Lock(),
    )
    monkeypatch.setattr(check, "check_gpus", lambda sku, count: [sku] * count)

    def stage(model, **kwargs):
        h.staged.append((model.hf_repo, kwargs.get("require_tokenizer", True)))
        return SimpleNamespace(path=tmp_path / model.hf_repo.replace("/", "--"))

    @contextmanager
    def network(**kwargs):
        yield SimpleNamespace(internal=kwargs["internal"])

    class Container:
        def __init__(self, **kwargs):
            h.containers.append(kwargs)

        def __enter__(self):
            return SimpleNamespace(base_url="http://baseline", container_id="c1")

        def __exit__(self, *exc):
            return False

    def qualify(**kwargs):
        h.qualified.append(kwargs)
        out = kwargs["output_dir"]
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps({"qualified_rows": 32}))
        (out / "sampling_rule.json").write_text(json.dumps(qualified_rule()))
        return qualified_rule()

    formatter = SimpleNamespace(encode=lambda text: text.split())
    monkeypatch.setattr(
        check,
        "highest_tier_request",
        lambda f, rule, i: ({"requests": [request()]}, "16k", request(), formatter),
    )

    def stream(url, **kwargs):
        assert url == "http://baseline"
        with h.lock:
            h.streams.append(kwargs)
        greedy = kwargs["temperature"] == 0.0
        return StreamResult(
            text=h.greedy_text(len(h.streams)) if greedy else "word " * 10,
            finish_reason="stop",
            completion_tokens=10,
            ttft_s=h.ttft_s,
            itl_s=[0.02] * 9,
            e2e_s=h.ttft_s + 0.18,
            prompt_tokens=INPUT_TOKENS,
        )

    @contextmanager
    def start(self, engine_start, *, phase):
        h.starts.append((engine_start.kind, list(engine_start.spec.serve_args)))
        assert self.draft_dir is not None
        yield "http://scorer"

    def score(url, captured, **kwargs):
        value = h.scores(captured) if callable(h.scores) else h.scores
        return [_PositionScore(i, "w", -1, value, None) for i in range(10)], 10, ""

    monkeypatch.setattr(check, "stage_weights", stage)
    monkeypatch.setattr(check, "BenchNetwork", network)
    monkeypatch.setattr(check, "EngineContainer", Container)
    monkeypatch.setattr(check, "qualify", qualify)
    monkeypatch.setattr(check, "post_completion_stream", stream)
    monkeypatch.setattr(check._EngineProvider, "start", start)
    monkeypatch.setattr(check, "score_captured_output", score)
    return h


def run(tmp_path, *extra, name="out"):
    out = tmp_path / name
    code = check.main(
        ["--campaign-fields", str(FIELDS), "--output-dir", str(out), *extra]
    )
    return code, json.loads((out / "summary.json").read_text())


def test_all_checks_pass_on_one_baseline_start(tmp_path, harness, capsys):
    code, summary = run(tmp_path)
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == (
        f"Preflight status: passed; summary: {(tmp_path / 'out/summary.json').resolve()}"
    )
    assert code == 0, summary
    assert summary["status"] == "passed"
    assert harness.staged == [
        ("moonshotai/Kimi-K3", True),
        ("RadixArk/Kimi-K3-DSpark", False),
    ]
    (container,) = harness.containers
    assert container["publish_port"] is True and container["gpu_count"] == 8
    assert container["network"].internal is False
    args = container["spec"].serve_args
    assert args.count("--model-path") == 1
    assert args[args.index("--speculative-draft-model-path") + 1] == "/draft"
    (qualified,) = harness.qualified
    assert qualified["container"] == "c1"
    assert (qualified["repetitions"], qualified["concurrency"]) == (2, 4)
    # 10 at each extreme, one greedy reference, then a C4 greedy wave.
    temps = [(s["temperature"], s["seed"]) for s in harness.streams]
    assert sorted(temps[:20]) == sorted((t, r) for t in (0.1, 1.01) for r in range(10))
    assert temps[20:] == [(0.0, 0)] * 5
    assert summary["tier"] == "16k" and summary["input_tokens"] == INPUT_TOKENS
    assert summary["sla"]["requests"] == 25 and summary["sla"]["failures"] == []
    assert set(summary["greedy"]["matches"].values()) == {1.0}
    assert [kind for kind, _ in harness.starts] == ["scorer"]
    assert summary["by_temperature"]["1.01"]["passed"] == 10
    rule = json.loads((tmp_path / "out/qualification/sampling_rule.json").read_text())
    assert rule["eligible_row_indices"] == list(range(32))


def test_health_timeout_reaches_the_scorer_start(tmp_path, harness):
    code, _ = run(tmp_path, "--health-timeout", "5400")
    assert code == 0
    (container,) = harness.containers
    assert container["health_timeout_s"] == 5400
    # The scorer's EngineContainer reads config, like every round start.
    assert config.BENCH_HEALTH_TIMEOUT_S == 5400


def test_low_logprob_slow_ttft_and_greedy_drift_each_fail(tmp_path, harness):
    harness.scores = lambda c: -20.0 if c.request_id == "t1.01-r3" else -1.0
    harness.ttft_s = 2.5
    harness.greedy_text = lambda n: "a b c d" if n == 21 else "a b x y"
    code, summary = run(tmp_path)
    assert code == 1
    assert summary["status"] == "failed_checks"
    assert summary["failed_samples"] == ["t1.01-r3"]
    assert any(f.startswith("sla: p99 TTFT") for f in summary["failures"])
    assert sum(f.startswith("greedy:") for f in summary["failures"]) == 4
    results = json.loads((tmp_path / "out/results.json").read_text())
    failed = next(r for r in results if r["request_id"] == "t1.01-r3")
    assert "mean logprob" in failed["reason"] and "quantile" in failed["reason"]
    assert failed["raw_min_below_min_token_logprob"] is True


def test_reused_rule_skips_qualification(tmp_path, harness):
    saved = tmp_path / "rule.json"
    saved.write_text(json.dumps(qualified_rule()))
    code, summary = run(tmp_path, "--qualified-rule", str(saved))
    assert code == 0, summary
    assert harness.qualified == []
    assert summary["qualification"] == {"reused": str(saved)}


def test_reused_rule_must_match_the_campaign(tmp_path):
    rule = qualified_rule()
    rule["max_tokens"] = 4096
    saved = tmp_path / "rule.json"
    saved.write_text(json.dumps(rule))
    with pytest.raises(check.SamplerError):
        check.load_qualified_rule(saved, fields())


def test_unfillable_qualification_is_a_check_failure(tmp_path, harness, monkeypatch):
    def qualify(**kwargs):
        raise check.SamplerError("input tiers: 16k cannot fill 16 slots")

    monkeypatch.setattr(check, "qualify", qualify)
    code, summary = run(tmp_path)
    assert code == 1 and summary["status"] == "failed_checks"
    assert harness.streams == []


def test_engine_token_count_mismatch_is_an_engine_error(tmp_path, harness, monkeypatch):
    original = check.post_completion_stream

    def miscounted(url, **kwargs):
        result = original(url, **kwargs)
        result.prompt_tokens = INPUT_TOKENS - 1
        return result

    monkeypatch.setattr(check, "post_completion_stream", miscounted)
    code, summary = run(tmp_path)
    assert code == 3 and "prompt tokens" in summary["error"]


def test_highest_tier_request_uses_the_last_tier(monkeypatch):
    trace = {
        "requests": [
            dict(request(), id="a", input_length_group="8k"),
            dict(request(), id="b", input_length_group="16k"),
            dict(request(), id="c", input_length_group="16k"),
        ]
    }
    monkeypatch.setattr(check, "build_prompt_formatter", lambda *a, **k: "fmt")
    monkeypatch.setattr(
        check,
        "generate_trace",
        lambda **k: SimpleNamespace(body=json.dumps(trace).encode()),
    )
    _, tier, chosen, formatter = check.highest_tier_request(
        fields(), qualified_rule(), 1
    )
    assert (tier, chosen["id"], formatter) == ("16k", "c", "fmt")
    with pytest.raises(ValueError, match="16k"):
        check.highest_tier_request(fields(), qualified_rule(), 2)


def test_gpu_check_requires_the_campaign_sku_and_nothing_else():
    def smi(names):
        return lambda *a, **k: SimpleNamespace(stdout="\n".join(names) + "\n")

    names = ["NVIDIA B300 SXM6 AC"] * 8
    assert check.check_gpus("B300", 8, runner=smi(names)) == names
    with pytest.raises(check.EngineError):
        check.check_gpus("B300", 8, runner=smi(names[:7]))
    with pytest.raises(check.EngineError):
        check.check_gpus("B300", 8, runner=smi([*names, "NVIDIA H200"]))


def test_token_match_is_positional_over_the_longer_output():
    assert check.token_match([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert check.token_match([1, 2, 3, 4], [1, 2, 9, 4]) == 0.75
    assert check.token_match([1, 2], [1, 2, 3, 4]) == 0.5


def test_existing_output_directory_is_never_reused(tmp_path, harness):
    assert run(tmp_path)[0] == 0
    with pytest.raises(FileExistsError):
        run(tmp_path)
