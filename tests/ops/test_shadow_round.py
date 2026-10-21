"""CPU-only tests for the shadow-round runner."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from bench.longform import qualification_contract
from bench.sampler import parse_sampling_rule

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "shadow_round", ROOT / "ops/shadow_round.py"
)
shadow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shadow)
pytestmark = pytest.mark.unit

TEMPLATE = ROOT / "fixtures/campaigns/sglang_kimi_k3_b300_exgentic/campaign-fields.json"


@pytest.fixture(autouse=True)
def restore_health_timeout(monkeypatch):
    monkeypatch.setattr(config, "BENCH_HEALTH_TIMEOUT_S", config.BENCH_HEALTH_TIMEOUT_S)


@pytest.fixture(autouse=True)
def prepared(tmp_path):
    """The CPU audit's prepared fields: the template with its snapshot count."""
    fields = json.loads(TEMPLATE.read_text())
    fields["sampling_rule"]["n_rows"] = 64
    (tmp_path / "campaign-fields.json").write_text(json.dumps(fields))


def qualified_rule_file(tmp_path):
    fields = json.loads((tmp_path / "campaign-fields.json").read_text())
    rule = dict(
        parse_sampling_rule(fields["sampling_rule"]),
        eligible_row_indices=list(range(32)),
    )
    rule["qualification"] = {
        "contract_sha256": qualification_contract(
            rule, fields["bench"], fields["engine"]
        ),
        "repetitions": 2,
    }
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(rule))
    return path


@pytest.fixture
def harness(monkeypatch):
    calls = {
        "report": {
            "verdict": "pass",
            "entries": [
                {
                    "index": 0,
                    "status": "scored",
                    "score": 0.01,
                    "score_report": {
                        "prompts": [
                            {"request_id": "ex-0", "candidate_failed": False},
                            {"request_id": "ex-1", "candidate_failed": True},
                        ]
                    },
                }
            ],
        },
        "code": 0,
    }
    monkeypatch.setattr(shadow, "check_gpus", lambda sku, count: [sku] * count)

    def sample_round(fields, output_dir):
        calls["rule"] = fields["sampling_rule"]
        output_dir.mkdir(parents=True)
        path = output_dir / "workload_trace.json"
        path.write_text(json.dumps({"schema_version": 1, "requests": [{"id": "ex-0"}]}))
        return path

    def run_bench(request_path, out):
        calls["request"] = json.loads(Path(request_path).read_text())
        out.mkdir(parents=True)
        (out / "bench_report.json").write_text(json.dumps(calls["report"]))
        return calls["code"]

    monkeypatch.setattr(shadow, "sample_round", sample_round)
    monkeypatch.setattr(shadow.harness, "run_bench", run_bench)
    return calls


def run(tmp_path, rule, *extra):
    out = tmp_path / "out"
    code = shadow.main(
        [
            "--campaign-fields",
            str(tmp_path / "campaign-fields.json"),
            "--qualified-rule",
            str(rule),
            "--output-dir",
            str(out),
            *extra,
        ]
    )
    return code, json.loads((out / "summary.json").read_text())


def test_unchanged_candidate_scored_passes(tmp_path, harness, capsys):
    code, summary = run(tmp_path, qualified_rule_file(tmp_path))
    assert code == 0, summary
    assert harness["rule"]["eligible_row_indices"] == list(range(32))
    request = harness["request"]
    assert request["draft_model"]["hf_repo"] == "RadixArk/Kimi-K3-DSpark"
    assert (
        request["engines"]["baseline"]["image"]
        == (request["engines"]["candidates"][0]["image"])
    )
    assert summary["entries"][0]["status"] == "scored"
    # The prompts the baseline lost against itself are reported, not hidden.
    assert summary["entries"][0]["prompt_counts"]["zeroed"] == 1
    assert summary["entries"][0]["prompt_counts"]["total"] == 2
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("Shadow status: passed; summary: ")


def test_round_engines_get_the_long_health_timeout(tmp_path, harness):
    rule = qualified_rule_file(tmp_path)
    assert run(tmp_path, rule)[0] == 0
    assert config.BENCH_HEALTH_TIMEOUT_S == shadow._preflight.DEFAULT_HEALTH_TIMEOUT_S
    assert shadow._preflight.DEFAULT_HEALTH_TIMEOUT_S >= 3600
    (tmp_path / "out").rename(tmp_path / "first")
    assert run(tmp_path, rule, "--health-timeout", "5400")[0] == 0
    assert config.BENCH_HEALTH_TIMEOUT_S == 5400


def test_disqualified_candidate_fails(tmp_path, harness):
    harness["report"] = {
        "verdict": "pass",
        "entries": [{"index": 0, "status": "disqualified", "reason": "correctness"}],
    }
    code, summary = run(tmp_path, qualified_rule_file(tmp_path))
    assert code == 1
    assert summary["failures"] == ["candidate 0 disqualified: correctness"]


def test_unqualified_rule_is_refused(tmp_path, harness):
    code, summary = run(tmp_path, tmp_path / "campaign-fields.json")
    assert code == 2 and "error" in summary


def test_drift_past_the_campaign_ceiling_fails(tmp_path, harness):
    harness["report"] = {
        "verdict": "pass",
        "baseline_drift": 0.2,
        "baseline": {"tier_completion": {"128k": {"completion_s": 900.0}}},
        "entries": [{"index": 0, "status": "scored", "score": 0.0}],
    }
    code, summary = run(tmp_path, qualified_rule_file(tmp_path))
    assert code == 1
    assert summary["failures"] == ["baseline drift 0.200 > ceiling 0.15"]
    assert summary["baseline_tier_completion_s"] == {"128k": 900.0}


def test_round_sampling_replays_its_receipt(tmp_path, monkeypatch):
    calls = []

    def generate_trace(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(body=b"{}", receipt={"rows": [1]})

    monkeypatch.setattr(shadow, "generate_trace", generate_trace)
    monkeypatch.setattr(shadow, "build_prompt_formatter", lambda *a, **k: "fmt")
    path = shadow.sample_round(
        json.loads((tmp_path / "campaign-fields.json").read_text()), tmp_path / "p"
    )
    assert path.read_bytes() == b"{}"
    assert calls[1]["sampling_receipt"] == {"rows": [1]}
    assert calls[0]["prompt_formatter"] == "fmt"
