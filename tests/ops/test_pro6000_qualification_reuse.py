"""Reuse only the recorded, unchanged GPU-qualified 8k/16k source outputs."""

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_longform_sampling import formatter, response, row

from bench.longform import digest, require_qualification
from bench.qualify_longform import qualify
from bench.sampler import generate_trace, sampling_context_for_rule
from bench.validate import validate_workload_trace_dict
from worker.round_job import materialize_round_trace

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "reuse_pro6000", ROOT / "ops/reuse_pro6000_qualification.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
SOURCE_FILES, reuse, sha256 = helper.SOURCE_FILES, helper.reuse, helper.sha256


@pytest.fixture
def source_pool(tmp_path, monkeypatch):
    target = json.loads(
        (
            ROOT / "fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json"
        ).read_text()
    )
    # Short synthetic outputs, real formatter/source bands and real qualification writer.
    target["sampling_rule"].update(max_tokens=10, min_output_tokens=6, n_rows=64)
    source_fields = copy.deepcopy(target)
    source_fields["bench"]["correctness"]["num_prompts"] = 32
    old = source_fields["sampling_rule"]
    old["n_prompts"] = 32
    del old["input_tiers"], old["max_baseline_prompt_drops"]
    fmt = formatter(old)
    monkeypatch.setattr(
        "bench.qualify_longform.verify_baseline_image",
        lambda **kw: {
            "engine_ref": target["base_image_digest"],
            "container_id": "baseline",
        },
    )
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **kw: None
    )

    def post(url, path, body, **kw):
        result = response()
        result["usage"]["prompt_tokens"] = len(fmt.encode(body["prompt"]))
        return result

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    source = tmp_path / "source"
    qualify(
        fields=source_fields,
        base_url="http://baseline",
        container="baseline",
        engine_ref=target["base_image_digest"],
        output_dir=source,
        pool_size=64,
        max_rows=64,
        repetitions=2,
        concurrency=4,
        row_fetcher=row,
        formatter=fmt,
    )

    def forbidden(*a, **kw):
        raise AssertionError("conversion must not contact an engine")

    monkeypatch.setattr("bench.qualify_longform.post_json", forbidden)
    monkeypatch.setattr("bench.qualify_longform.verify_baseline_image", forbidden)
    monkeypatch.setattr("bench.qualify_longform.validate_engine_workload", forbidden)
    return source, target, fmt


def convert(source_pool, output, **kw):
    source, target, fmt = source_pool
    return reuse(
        source_dir=source,
        output_dir=output,
        fields=target,
        formatter=fmt,
        row_fetcher=kw.pop("row_fetcher", row),
        **kw,
    )


def test_reuse_preserves_provenance_and_round_receipts(source_pool, tmp_path):
    source, target, fmt = source_pool
    before = {name: (source / name).read_bytes() for name in SOURCE_FILES}
    output = tmp_path / "converted"
    new = convert(source_pool, output)
    assert new["n_prompts"] == 16 and new["max_baseline_prompt_drops"] == 4
    assert len(new["eligible_row_indices"]) == 32
    assert all(i % 4 in (2, 3) for i in new["eligible_row_indices"])
    require_qualification(new, target["bench"], target["engine"])
    for name, data in before.items():
        assert (
            (source / name).read_bytes()
            == (output / "source" / name).read_bytes()
            == data
        )
    provenance = json.loads((output / "reuse.json").read_text())
    assert provenance["gpu_requests_made"] == 0
    assert provenance["source_files"] == {
        name: sha256(data) for name, data in before.items()
    }
    records = [
        json.loads(line)
        for line in (output / "qualification.jsonl").read_text().splitlines()
    ]
    assert Counter(r["input_length_group"] for r in records[1:]) == {
        "8k": 32,
        "16k": 32,
    }
    old_records = [
        json.loads(line) for line in before["qualification.jsonl"].splitlines()
    ]
    assert records[1:] == [
        r for r in old_records[1:] if r["input_length_group"] in ("8k", "16k")
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["evidence_sha256"] == sha256(
        (output / "qualification.jsonl").read_bytes()
    )
    assert summary["sampling_rule_sha256"] == digest(new)
    sampled = generate_trace(
        rule=new,
        seed_hex="a" * 64,
        row_fetcher=row,
        prompt_formatter=fmt,
        sampling_context=sampling_context_for_rule(
            new, target["bench"], target["engine"]
        ),
    )
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert Counter(r.input_length_group for r in trace.requests) == {"8k": 8, "16k": 8}
    restored = materialize_round_trace(
        {"sampled_trace_sha256": sampled.sha256, "sampling_receipt": sampled.receipt},
        SimpleNamespace(bench=target["bench"], engine=target["engine"]),
        tmp_path / "worker",
        row_fetcher=row,
        prompt_formatter=fmt,
    )
    assert restored.read_bytes() == sampled.body
    with pytest.raises(ValueError, match="fresh"):
        convert(source_pool, output)


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "image",
        "serve_args",
        "scorer",
        "gpu_count",
        "temperature",
        "timeout",
        "followup",
        "dataset",
        "exclusions",
    ],
)
def test_reuse_rejects_changed_execution_or_generation_pins(
    source_pool, tmp_path, change
):
    _, target, _ = source_pool
    if change == "model":
        target["bench"]["model"]["hf_revision"] = "a" * 40
    elif change == "image":
        target["bench"]["baseline_engine_image_digest"] = "sha256:" + "a" * 64
    elif change == "serve_args":
        target["bench"]["serve_args"] += ["--different"]
    elif change == "scorer":
        target["bench"]["correctness"]["serve_args"][-1] = "0.7"
    elif change == "gpu_count":
        target["bench"]["gpu_count"] = 2
    else:
        key, value = {
            "temperature": ("temperature_range", [0.2, 1.01]),
            "timeout": ("request_timeout_s", 601),
            "followup": ("followup_prompt", "Different work"),
            "dataset": ("revision", "b" * 40),
            "exclusions": ("max_baseline_prompt_drops", 8),
        }[change]
        target["sampling_rule"][key] = value
    output = tmp_path / "converted"
    with pytest.raises(ValueError, match="refused"):
        convert(source_pool, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "change",
    ["checksum", "rule", "missing", "duplicate", "short", "sampling", "tokens", "tier"],
)
def test_reuse_checks_evidence_before_writing(source_pool, tmp_path, change):
    source, _, _ = source_pool
    evidence = source / "qualification.jsonl"
    records = [json.loads(line) for line in evidence.read_text().splitlines()]
    if change == "checksum":
        evidence.write_text(evidence.read_text() + "\n")
    elif change == "rule":
        path = source / "sampling_rule.json"
        rule = json.loads(path.read_text())
        rule["qualification"]["contract_sha256"] = "sha256:" + "a" * 64
        path.write_text(json.dumps(rule))
    else:
        if change == "missing":
            records.pop(1)
        elif change == "duplicate":
            records.append(records[1])
        else:
            selected = records[1]["row_index"]
            for record in records[1:]:
                if record["row_index"] != selected:
                    continue
                if change == "short":
                    record["completion_tokens"] = 5
                    record["finish_reason"] = "stop"
                elif change == "sampling":
                    record["sampling"]["temperature"] = 0.7
                elif change == "tokens":
                    record["input_ids_sha256"] = "sha256:" + "a" * 64
                elif change == "tier":
                    record["input_length_group"] = "8k"
        evidence.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        path = source / "summary.json"
        summary = json.loads(path.read_text())
        summary["evidence_sha256"] = sha256(evidence.read_bytes())
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="refused"):
        convert(source_pool, tmp_path / "converted")
    assert not (tmp_path / "converted").exists()


def test_reuse_rechecks_pinned_prompt_bytes(source_pool, tmp_path):
    with pytest.raises(ValueError, match="prompt/tokenization changed"):
        convert(
            source_pool,
            tmp_path / "converted",
            row_fetcher=lambda i: row(i, references=15000 if i % 4 == 3 else 7500),
        )
