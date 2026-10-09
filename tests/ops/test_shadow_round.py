"""CPU-only tests for the shadow-round runner."""

import importlib.util
import json
from pathlib import Path

import pytest

from bench.longform import qualification_contract
from bench.sampler import parse_sampling_rule

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "shadow_round", ROOT / "ops/shadow_round.py"
)
shadow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shadow)
pytestmark = pytest.mark.unit

FIELDS = ROOT / "fixtures/campaigns/sglang_kimi_k3_b300/campaign-fields.json"


def qualified_rule_file(tmp_path):
    fields = json.loads(FIELDS.read_text())
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
            "entries": [{"index": 0, "status": "scored", "score": 0.01}],
        },
        "code": 0,
    }
    monkeypatch.setattr(shadow, "check_gpus", lambda sku, count: [sku] * count)

    def preview(*, fields, output_dir, **kwargs):
        calls["rule"] = fields["sampling_rule"]
        output_dir.mkdir(parents=True)
        (output_dir / "workload_trace.json").write_text(
            json.dumps({"schema_version": 1, "requests": [{"id": "hf-0"}]})
        )

    def run_bench(request_path, out):
        calls["request"] = json.loads(Path(request_path).read_text())
        out.mkdir(parents=True)
        (out / "bench_report.json").write_text(json.dumps(calls["report"]))
        return calls["code"]

    monkeypatch.setattr(shadow, "preview", preview)
    monkeypatch.setattr(shadow.harness, "run_bench", run_bench)
    return calls


def run(tmp_path, rule):
    out = tmp_path / "out"
    code = shadow.main(
        [
            "--campaign-fields",
            str(FIELDS),
            "--qualified-rule",
            str(rule),
            "--output-dir",
            str(out),
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
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("Shadow status: passed; summary: ")


def test_disqualified_candidate_fails(tmp_path, harness):
    harness["report"] = {
        "verdict": "pass",
        "entries": [{"index": 0, "status": "disqualified", "reason": "correctness"}],
    }
    code, summary = run(tmp_path, qualified_rule_file(tmp_path))
    assert code == 1
    assert summary["failures"] == ["candidate 0 disqualified: correctness"]


def test_unqualified_rule_is_refused(tmp_path, harness):
    code, summary = run(tmp_path, FIELDS)
    assert code == 2 and "error" in summary
