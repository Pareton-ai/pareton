"""Snapshot import, target-token tiers, qualification and worker replay for v6."""

import copy
import json
import re
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.concurrency import WEIGHTED_RULE, request_groups, validate_tier_contract
from bench.exgentic import (
    DATASET,
    REVISION,
    flatten_dataset,
    normalize_snapshot,
    prompt_kv_tokens,
)
from bench.longform import (
    RENDER_FAILED,
    candidate_for_row,
    digest,
    require_qualification,
    sampling_context_for_campaign,
    scan_candidates,
)
from bench.qualify_exgentic import prepare
from bench.qualify_longform import qualify
from bench.sampler import (
    PromptFormatter,
    PromptRenderError,
    SamplerError,
    generate_trace,
    parse_sampling_rule,
)
from bench.score import PromptTiming, score_candidate
from bench.validate import validate_workload_trace_dict
from worker.round_job import materialize_round_trace

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).parents[1] / "fixtures/campaigns/sglang_kimi_k3_b300_exgentic"
TIERS = tuple(
    json.loads((FIXTURE / "campaign-fields.json").read_text())["sampling_rule"][
        "input_tiers"
    ]
)
SIZES = tuple(int(tier[:-1]) * 1024 for tier in TIERS)


def fields(n=8):
    value = json.loads((FIXTURE / "campaign-fields.json").read_text())
    value["sampling_rule"].update(
        n_rows=n, n_prompts=4, max_tokens=32, min_output_tokens=1
    )
    return value


def formatter():
    def render(messages, *, tools=None):
        return json.dumps({"messages": messages, "tools": tools})

    def encode(prompt):
        n, identifier = map(int, re.search(r"n=(\d+);id=(\d+)", prompt).groups())
        return [identifier + 1] + [0] * (n - 1)

    return PromptFormatter(
        render,
        {
            "chat_template": {"enable_thinking": False},
            "tokenizer": {"sha256": "sha256:" + "a" * 64},
        },
        encode,
    )


def snapshots(n=8):
    return [
        {
            "snapshot_json": json.dumps(
                {
                    "session_id": f"session-{i}",
                    "source_row": i,
                    "span_index": 2,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"n={SIZES[i % 4]};id={i}",
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "search", "parameters": {}},
                        }
                    ],
                }
            )
        }
        for i in range(n)
    ]


def sample(f=None, rows=None, **kwargs):
    f = f or fields()
    rows = rows or snapshots()
    return generate_trace(
        rule=f["sampling_rule"],
        seed_hex="a" * 64,
        row_fetcher=rows.__getitem__,
        prompt_formatter=formatter(),
        sampling_context=sampling_context_for_campaign(f["bench"], f["engine"]),
        **kwargs,
    )


def fake_baseline(monkeypatch, f):
    monkeypatch.setattr(
        "bench.qualify_longform.verify_baseline_image",
        lambda **kw: {
            "engine_ref": f["bench"]["baseline_engine_image_digest"],
            "container_id": "baseline",
        },
    )
    checks = []
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload",
        lambda *a, **kw: checks.append(kw),
    )
    return checks


def source_row():
    return {
        "session_id": "source-session",
        "benchmark": "swebench",
        "spans": [
            {
                "type": "llm_call",
                "status": {"code": 0},
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.usage.input_tokens": 99999,
                    "gen_ai.usage.output_tokens": 4,
                    "gen_ai.request.model": "source-model",
                    "gen_ai.system_instructions": "System instructions",
                    "gen_ai.input.messages": json.dumps(
                        [
                            {
                                "role": "developer",
                                "parts": [
                                    {
                                        "type": "text",
                                        "content": "Developer instructions",
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "parts": [
                                    {"type": "text", "content": "Inspect repository"}
                                ],
                            },
                            {
                                "role": "assistant",
                                "parts": [
                                    {"type": "thinking", "thinking": "Plan"},
                                    {
                                        "type": "tool_call",
                                        "id": "call-1",
                                        "name": "read",
                                        "arguments": {"path": "file.py"},
                                    },
                                ],
                            },
                            {
                                "role": "tool",
                                "parts": [
                                    {
                                        "type": "tool_call_response",
                                        "id": "call-1",
                                        "result": "Contents",
                                    }
                                ],
                            },
                        ]
                    ),
                    "gen_ai.tool.definitions": json.dumps(
                        [{"type": "function", "name": "read", "parameters": {}}]
                    ),
                    "gen_ai.output.messages": "DO NOT REPLAY THE SOURCE ANSWER",
                },
            }
        ],
    }


def test_import_preserves_system_tools_and_history_without_source_answer():
    snapshot = normalize_snapshot(source_row(), 7, 0)
    assert [m["role"] for m in snapshot["messages"]] == [
        "system",
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert snapshot["messages"][3]["reasoning_content"] == "Plan"
    assert json.loads(
        snapshot["messages"][3]["tool_calls"][0]["function"]["arguments"]
    ) == {"path": "file.py"}
    assert snapshot["messages"][4] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "Contents",
    }
    assert snapshot["tools"][0]["function"]["name"] == "read"
    assert "DO NOT REPLAY" not in json.dumps(snapshot)
    assert (snapshot["source_row"], snapshot["span_index"]) == (7, 0)


@pytest.mark.parametrize("change", ["failed", "zero", "image", "error"])
def test_import_rejects_failed_zero_token_and_unsupported_requests(change):
    row = source_row()
    span = row["spans"][0]
    if change == "failed":
        span["status"]["code"] = 2
    elif change == "zero":
        span["attributes"]["gen_ai.usage.output_tokens"] = 0
    elif change == "error":
        span["attributes"]["error.type"] = "rate_limit"
    else:
        span["attributes"]["gen_ai.input.messages"] = json.dumps(
            [{"role": "user", "parts": [{"type": "image", "url": "ignored"}]}]
        )
    assert normalize_snapshot(row, 0, 0) is None


def test_arrow_flattening_preserves_session_span_order():
    from datasets import Dataset

    first, second = source_row(), source_row()
    first["spans"].append(copy.deepcopy(first["spans"][0]))
    second["session_id"] = "second"
    dataset = flatten_dataset(Dataset.from_list([first, second]))
    values = [json.loads(row["snapshot_json"]) for row in dataset]
    assert [(row["source_row"], row["span_index"]) for row in values] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]


def test_exact_tiers_and_receipt_replay_through_worker(tmp_path):
    f = fields()
    value = sample(f)
    trace = validate_workload_trace_dict(json.loads(value.body))
    assert Counter(r.input_length_group for r in trace.requests) == dict.fromkeys(
        TIERS, 1
    )
    assert sorted(r.input_tokens for r in trace.requests) == list(SIZES)
    for request in trace.requests:
        rendered = json.loads(request.prompt)
        assert rendered["messages"][-1] == {
            "role": "user",
            "content": f["sampling_rule"]["followup_prompt"],
        }
        assert rendered["tools"][0]["function"]["name"] == "search"
    assert value.receipt["followup_prompt"] == f["sampling_rule"]["followup_prompt"]
    assert all(0.1 <= r.sampling.temperature <= 1.01 for r in trace.requests)
    assert len({r.sampling.temperature for r in trace.requests}) == 4
    assert all("snapshot_sha256" in r for r in value.receipt["requests"])
    assert sample(f, sampling_receipt=value.receipt).body == value.body
    path = materialize_round_trace(
        {"sampled_trace_sha256": value.sha256, "sampling_receipt": value.receipt},
        SimpleNamespace(bench=f["bench"], engine=f["engine"]),
        tmp_path / "worker",
        row_fetcher=snapshots().__getitem__,
        prompt_formatter=formatter(),
    )
    assert path.read_bytes() == value.body
    assert len(request_groups(trace.requests, 32, TIERS, size=1)) == 1
    changed = copy.deepcopy(value.receipt)
    changed["requests"][0]["snapshot_sha256"] = "sha256:" + "b" * 64
    with pytest.raises(SamplerError, match="receipt does not reproduce"):
        sample(f, sampling_receipt=changed)


def test_missing_tier_fails_without_truncation_or_fallback():
    rows = snapshots()
    for i in (3, 7):
        rows[i]["snapshot_json"] = rows[i]["snapshot_json"].replace(
            f"n={SIZES[-1]}", f"n={SIZES[-1] + 1}"
        )
    with pytest.raises(SamplerError, match=f"{TIERS[-1]}.*no shorter fallback"):
        sample(rows=rows)


@pytest.mark.parametrize("indices", [(0, 4), (0, 1)])
def test_one_snapshot_per_session_per_round(indices):
    # (0, 1) puts one session in two tiers, where they would share a prefix.
    rows = snapshots(8)
    for index in indices:
        row = json.loads(rows[index]["snapshot_json"])
        row["session_id"] = "same"
        rows[index]["snapshot_json"] = json.dumps(row)
    f = fields()
    f["sampling_rule"]["n_prompts"] = 8
    with pytest.raises(SamplerError, match="insufficient distinct"):
        sample(f, rows)


def test_weights_and_legacy_guard():
    f = fields()
    validate_tier_contract(f["sampling_rule"], f["scoring_rule"])
    assert f["scoring_rule"]["tier_weights"] == dict(
        zip(TIERS, [0.25] * 4, strict=True)
    )
    baseline = {t: PromptTiming(1, [], 8, "stop") for t in TIERS}
    bt = {t: {"request_ids": [t], "completion_s": 10} for t in TIERS}
    ct = {
        t: {"request_ids": [t], "completion_s": v}
        for t, v in zip(TIERS, [5, 10, 20, 10], strict=True)
    }
    result = score_candidate(
        f["scoring_rule"],
        baseline=baseline,
        candidate=baseline,
        baseline_tiers=bt,
        candidate_tiers=ct,
    )
    assert result.rule == WEIGHTED_RULE
    assert result.score == pytest.approx(0.25 * 0.5 - 0.25)
    with pytest.raises(SamplerError, match="input_tiers"):
        parse_sampling_rule({**f["sampling_rule"], "algo_version": 5})
    missing_followup = {
        k: v for k, v in f["sampling_rule"].items() if k != "followup_prompt"
    }
    with pytest.raises(SamplerError, match="explicit followup_prompt"):
        parse_sampling_rule(missing_followup)
    with pytest.raises(SamplerError, match="pinned v2"):
        parse_sampling_rule({**f["sampling_rule"], "revision": "b" * 40})


def test_cpu_preparation_writes_audit_and_replay(tmp_path):
    f = fields()
    f["sampling_rule"]["n_rows"] = 0
    output = tmp_path / "prepared"
    prepared = prepare(f, output, dataset=snapshots(), formatter=formatter())
    assert prepared["sampling_rule"]["n_rows"] == 8
    assert "qualification" not in prepared["sampling_rule"]
    assert json.loads((output / "audit.json").read_text())[
        "distinct_sessions_by_tier"
    ] == dict.fromkeys(TIERS, 2)
    assert (output / "workload_trace.json").exists()
    audit = json.loads((output / "audit.json").read_text())
    assert audit["followup_prompt_sha256"] == digest(
        f["sampling_rule"]["followup_prompt"]
    )
    assert "follow-up" in audit["input_length_basis"]
    # The test encoder gives every prompt a distinct first token.
    kv = audit["preview_round_kv"]
    assert kv["unique_prefix_tokens"] == kv["prompt_tokens"] == sum(SIZES)


def test_prompt_kv_tokens_counts_a_shared_prefix_once():
    assert prompt_kv_tokens([[1, 2, 3, 4], [1, 2, 9], [7], [1, 2, 3, 4]]) == {
        "prompt_tokens": 12,
        "unique_prefix_tokens": 6,
    }


def test_baseline_qualification_temperature_extremes_and_stale_contract(
    tmp_path, monkeypatch
):
    f = fields()
    f["sampling_rule"].update(max_tokens=5120, min_output_tokens=1500)
    fmt = formatter()
    fake_baseline(monkeypatch, f)
    calls = []

    def post(url, path, body, **kwargs):
        assert path == "/v1/completions" and body["ignore_eos"] is False
        assert "min_tokens" not in body
        assert body["temperature"] in (0.1, 1.01)
        n = len(fmt.encode(body["prompt"]))
        calls.append((body["prompt"], n, body["temperature"]))
        return {
            "choices": [
                {
                    "text": '{"tool": "search", "query": "result"}',
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": n, "completion_tokens": 1500},
        }

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    qualified = qualify(
        fields=f,
        base_url="http://127.0.0.1:8000",
        container="baseline",
        engine_ref=f["bench"]["baseline_engine_image_digest"],
        output_dir=tmp_path / "qualified",
        pool_size=8,
        max_rows=8,
        row_fetcher=snapshots().__getitem__,
        formatter=fmt,
    )
    assert len(calls) == 16
    assert {n for _, n, _ in calls} == set(SIZES)
    for prompt, _, _ in calls:
        assert [temp for p, _, temp in calls if p == prompt] == [0.1, 1.01]
    require_qualification(qualified, f["bench"], f["engine"])
    assert len(qualified["eligible_row_indices"]) == 8
    for changed in (
        {**qualified, "max_tokens": 5119},
        {**qualified, "followup_prompt": "A different long-form task"},
    ):
        with pytest.raises(SamplerError, match="baseline qualification"):
            require_qualification(changed, f["bench"], f["engine"])


def test_fixture_changes_only_the_workload_and_scorer_batch():
    original = json.loads(
        (FIXTURE.parent / "sglang_kimi_k3_b300/campaign-fields.json").read_text()
    )
    new = json.loads((FIXTURE / "campaign-fields.json").read_text())
    # Grading is sequential, so the scorer keeps one request slot.
    scorer = new["bench"]["correctness"]["serve_args"]
    assert scorer[-2:] == ["--max-running-requests", "1"]
    scorer[-1] = "8"
    workload = {
        "algo_version",
        "dataset",
        "revision",
        "n_rows",
        "input_tiers",
        "followup_prompt",
    }
    for key, ignored in (
        ("sampling_rule", workload),
        ("scoring_rule", {"tier_weights"}),
    ):
        assert {k: v for k, v in new.pop(key).items() if k not in ignored} == {
            k: v for k, v in original.pop(key).items() if k not in ignored
        }
    assert new == original
    rule = json.loads((FIXTURE / "campaign-fields.json").read_text())["sampling_rule"]
    assert (rule["dataset"], rule["revision"], rule["n_rows"]) == (DATASET, REVISION, 0)
    assert (
        json.loads((FIXTURE / "scoring_rule.json").read_text())
        == json.loads((FIXTURE / "campaign-fields.json").read_text())["scoring_rule"]
    )
    # An unprepared template cannot be launched.
    with pytest.raises(SamplerError):
        parse_sampling_rule(rule)


@pytest.mark.parametrize("stale", [False, True])
def test_v6_cannot_seed_without_fresh_qualification(monkeypatch, stale):
    from test_seed import _patch_store

    from campaign.seed import seed_synthetic_campaign

    captured = _patch_store(monkeypatch)
    f = fields(64)
    if stale:
        f["sampling_rule"]["eligible_row_indices"] = list(range(8))
        f["sampling_rule"]["qualification"] = {
            "contract_sha256": "sha256:" + "a" * 64,
            "repetitions": 2,
        }
    with pytest.raises(SamplerError, match="baseline qualification"):
        seed_synthetic_campaign(
            submission_fee_tao="0.35",
            allow_placeholders=True,
            status="draft",
            sampling_rule=f["sampling_rule"],
            scoring_rule=f["scoring_rule"],
            bench_max_model_len=1048576,
        )
    assert captured["inserts"] == 0 and captured["profile_data"] is None


def test_qualification_pool_needs_distinct_sessions_before_gpu_calls(
    tmp_path, monkeypatch
):
    f = fields()
    rows = snapshots()
    for index in (0, 4):
        snapshot = json.loads(rows[index]["snapshot_json"])
        snapshot["session_id"] = "one-session"
        rows[index]["snapshot_json"] = json.dumps(snapshot)
    fake_baseline(monkeypatch, f)
    monkeypatch.setattr(
        "bench.qualify_longform.post_json",
        lambda *a, **kw: pytest.fail("GPU called before input coverage check"),
    )
    with pytest.raises(SamplerError, match="distinct sessions"):
        qualify(
            fields=f,
            base_url="http://127.0.0.1:8000",
            container="baseline",
            engine_ref=f["bench"]["baseline_engine_image_digest"],
            output_dir=tmp_path / "failed",
            pool_size=8,
            max_rows=8,
            row_fetcher=rows.__getitem__,
            formatter=formatter(),
        )
    assert (tmp_path / "failed/qualification.jsonl").exists()
    assert not (tmp_path / "failed/sampling_rule.json").exists()


def test_qualification_defaults_to_full_split_and_32_in_flight_across_tiers(
    tmp_path, monkeypatch
):
    f = fields(64)
    f["sampling_rule"]["n_prompts"] = 32
    fmt = formatter()
    checks = fake_baseline(monkeypatch, f)
    barrier = threading.Barrier(32, timeout=20)
    lock = threading.Lock()
    calls = []

    def post(url, path, body, **kwargs):
        n = len(fmt.encode(body["prompt"]))
        with lock:
            calls.append((n, body["temperature"]))
        # A single-tier batch or C4 default cannot complete this barrier.
        barrier.wait()
        return {
            "choices": [{"text": "A valid response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n, "completion_tokens": 8},
        }

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    qualified = qualify(
        fields=f,
        base_url="http://127.0.0.1:8000",
        container="baseline",
        engine_ref=f["bench"]["baseline_engine_image_digest"],
        output_dir=tmp_path,
        row_fetcher=snapshots(64).__getitem__,
        formatter=fmt,
    )
    assert len(calls) == 128
    for offset in range(0, 128, 32):
        assert Counter(n for n, _ in calls[offset : offset + 32]) == dict.fromkeys(
            SIZES, 8
        )
        assert {temp for _, temp in calls[offset : offset + 32]} == {
            0.1 if offset % 64 == 0 else 1.01
        }
    assert len(qualified["eligible_row_indices"]) == 64
    assert json.loads((tmp_path / "summary.json").read_text())["concurrency"] == 32
    # Kimi's /tokenize check must send prompts in its 400k-character windows.
    assert {c["tokenize_window"] for c in checks} == {400_000}


def test_exgentic_preserves_kimi_output_floor():
    from bench.qualify_longform import evaluate_response

    rule = fields()["sampling_rule"]
    rule.update(max_tokens=5120)
    rule.pop("min_output_tokens")
    rule = parse_sampling_rule(rule)
    assert rule["min_output_tokens"] == 1500
    for count, rejection in ((1499, "short_output"), (1500, None)):
        response = {
            "choices": [{"text": "A valid response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 16384, "completion_tokens": count},
        }
        assert evaluate_response(response, rule, 16384)["rejection"] == rejection


@pytest.mark.parametrize("n_rows", [64, 68])
def test_mixed_tier_qualification_refills_rejections_or_fails_closed(
    tmp_path, monkeypatch, n_rows
):
    f = fields(n_rows)
    f["sampling_rule"].update(n_prompts=32, max_tokens=5120, min_output_tokens=1500)
    fmt = formatter()
    fake_baseline(monkeypatch, f)
    lock = threading.Lock()
    rejected = False

    def post(url, path, body, **kwargs):
        nonlocal rejected
        n = len(fmt.encode(body["prompt"]))
        count = 1500
        with lock:
            if n == 16384 and not rejected:
                rejected = True
                count = 1499
        return {
            "choices": [{"text": "A valid response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n, "completion_tokens": count},
        }

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    kwargs = dict(
        fields=f,
        base_url="http://127.0.0.1:8000",
        container="baseline",
        engine_ref=f["bench"]["baseline_engine_image_digest"],
        output_dir=tmp_path,
        max_rows=n_rows,
        row_fetcher=snapshots(n_rows).__getitem__,
        formatter=fmt,
    )
    if n_rows == 64:
        with pytest.raises(SamplerError, match="16k cannot fill 16 slots"):
            qualify(**kwargs)
        assert not (tmp_path / "sampling_rule.json").exists()
    else:
        result = qualify(**kwargs)
        assert len(result["eligible_row_indices"]) == 64
        assert json.loads((tmp_path / "summary.json").read_text())[
            "qualified_rows_by_input_tier"
        ] == dict.fromkeys(TIERS, 16)
    records = [
        json.loads(line)
        for line in (tmp_path / "qualification.jsonl").read_text().splitlines()
    ]
    assert sum(row.get("rejection") == "short_output" for row in records) == 1


@pytest.mark.parametrize("prompt", [None, "", "   ", []])
def test_followup_must_be_nonempty_text(prompt):
    with pytest.raises(SamplerError, match="followup_prompt"):
        parse_sampling_rule({**fields()["sampling_rule"], "followup_prompt": prompt})


def test_followup_tokens_determine_tier_and_capacity_without_mutating_snapshot():
    f = fields()
    rule = parse_sampling_rule(
        {**f["sampling_rule"], "followup_prompt": "one two three four"}
    )
    fmt = formatter()
    original_encode = fmt.encode

    def encode(prompt):
        value = json.loads(prompt)
        assert value["messages"][-1] == {
            "role": "user",
            "content": rule["followup_prompt"],
        }
        return original_encode(prompt) + [9] * len(
            value["messages"][-1]["content"].split()
        )

    fmt = PromptFormatter(fmt.render, fmt.receipt, encode)
    row = snapshots()[0]
    original_row = copy.deepcopy(row)
    context = sampling_context_for_campaign(f["bench"], f["engine"])
    # Source alone fills the tier; appending the task pushes it outside the band.
    assert candidate_for_row(0, row, fmt, rule, context) is None
    assert row == original_row
    row["snapshot_json"] = row["snapshot_json"].replace(
        f"n={SIZES[0]}", f"n={SIZES[0] - 4}"
    )
    candidate = candidate_for_row(0, row, fmt, rule, context)
    assert candidate["input_tokens"] == SIZES[0]
    assert candidate["input_length_group"] == TIERS[0]
    rendered = json.loads(candidate["prompt"])
    source = json.loads(row["snapshot_json"])
    assert rendered["messages"][:-1] == source["messages"]
    assert rendered["tools"] == source["tools"]
    assert (
        candidate_for_row(
            0, row, fmt, rule, {**context, "max_input_tokens": SIZES[0] - 1}
        )
        is None
    )


def test_changed_followup_cannot_replay_old_receipt():
    f = fields()
    original = sample(f)
    f["sampling_rule"]["followup_prompt"] = "Write a detailed mathematical exposition."
    with pytest.raises(SamplerError, match="receipt does not reproduce"):
        sample(f, sampling_receipt=original.receipt)


def test_sampler_reaches_eligible_sessions_beyond_benchmark_grouped_prefix():
    rows = snapshots(308)
    for i in range(300):
        snapshot = json.loads(rows[i]["snapshot_json"])
        snapshot["benchmark"] = "appworld"
        snapshot["messages"][0]["content"] = f"n=100;id={i}"
        rows[i]["snapshot_json"] = json.dumps(snapshot)
    result = sample(fields(308), rows)
    assert len(result.row_indices) == 4
    assert all(index >= 300 for index in result.row_indices)
    assert (
        sample(fields(308), rows, sampling_receipt=result.receipt).body == result.body
    )


def test_session_in_flight_is_deferred_not_dropped(tmp_path, monkeypatch):
    # Rows 60 (16k) and 21 (32k) share a session and are each tier's first
    # candidate, so they meet in the first batch. 32k has exactly its quota.
    # Row 60 is rejected, so row 21 must still qualify in a later batch.
    f = fields(68)
    f["sampling_rule"].update(n_prompts=32, max_tokens=5120, min_output_tokens=1500)
    fmt = formatter()
    rows = snapshots(68)
    for index, update in ((60, {}), (21, {}), (65, {"size": SIZES[2]})):
        snapshot = json.loads(rows[index]["snapshot_json"])
        if not update:
            snapshot["session_id"] = "shared"
        else:
            snapshot["messages"][0]["content"] = f"n={update['size']};id={index}"
        rows[index]["snapshot_json"] = json.dumps(snapshot)
    fake_baseline(monkeypatch, f)

    def post(url, path, body, **kwargs):
        tokens = fmt.encode(body["prompt"])
        return {
            "choices": [{"text": "A valid response", "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": 1499 if tokens[0] == 61 else 1500,
            },
        }

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    result = qualify(
        fields=f,
        base_url="http://127.0.0.1:8000",
        container="baseline",
        engine_ref=f["bench"]["baseline_engine_image_digest"],
        output_dir=tmp_path,
        max_rows=68,
        row_fetcher=rows.__getitem__,
        formatter=fmt,
    )
    assert 60 not in result["eligible_row_indices"]
    assert 21 in result["eligible_row_indices"]


def test_parallel_scan_matches_one_row_at_a_time():
    f = fields(n=40)
    rule = parse_sampling_rule(f["sampling_rule"])
    context = sampling_context_for_campaign(f["bench"], f["engine"])
    rows = snapshots(40)
    base = formatter()

    def render(messages, *, tools=None):
        # One unrenderable row keeps its place in the output.
        if "id=5" in json.dumps(messages):
            raise PromptRenderError("unrenderable")
        return base.render(messages, tools=tools)

    fmt = PromptFormatter(render, base.receipt, base.encode)
    serial = list(
        scan_candidates(range(40), rows.__getitem__, fmt, rule, context, workers=1)
    )
    parallel = list(
        scan_candidates(range(40), rows.__getitem__, fmt, rule, context, workers=3)
    )
    assert parallel == serial
    assert [i for i, _ in parallel] == list(range(40))
    assert serial[5][1] == RENDER_FAILED
    assert serial[0][1]["input_tokens"] == SIZES[0]
    assert all("prompt" not in c for _, c in serial if isinstance(c, dict))
