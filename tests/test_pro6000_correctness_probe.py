"""CPU-only regression tests for the opt-in PAR-144 diagnostic."""

import copy
import json
import runpy
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import bench.main as harness
from bench.correctness import BASELINE_INDEX, CapturedOutput, PendingCorrectness
from bench.lifecycle import EngineError
from bench.longform import length_groups, sampling_context_for_campaign
from bench.schemas import CorrectnessReport
from bench.trajectory import token_ids_sha256

ROOT = Path(__file__).resolve().parents[1]
PROBE = runpy.run_path(str(ROOT / "ops/pro6000-correctness-probe.py"))
pytestmark = pytest.mark.unit


def fields():
    value = json.loads(
        (ROOT / "fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json").read_text()
    )
    value["gpu_skus"] = ["RTXPRO6000"]
    value["bench"]["gpu_count"] = 1
    value["bench"]["model"].update(
        hf_repo="Qwen/Qwen3.8-27B-FP8",
        quantization="fp8",
        hf_revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    )
    value["bench"]["serve_args"] = ["--tp", "1", "--mem-fraction-static", "0.85"]
    return value


def source_trace(f):
    groups = length_groups(4)
    sampling = {
        "algo_version": 4,
        "enable_thinking": False,
        "context": sampling_context_for_campaign(f["bench"], f["engine"]),
        "request_interval_ms": 2,
        "max_tokens": 5120,
        "min_output_tokens": 3000,
        "length_groups": groups,
    }
    requests = [
        {
            "id": f"hf-{i}",
            "prompt": "x" * group["max_tokens"],
            "input_tokens": group["max_tokens"],
            "input_length_group": group["name"],
            "input_ids_sha256": token_ids_sha256([1] * group["max_tokens"]),
            "arrival_offset_ms": i * 2,
            "max_tokens": 5120,
            "sampling": {"temperature": 0.0, "top_p": 1.0},
        }
        for i, group in enumerate(groups)
    ]
    return {"schema_version": 1, "meta": {"sampling": sampling}, "requests": requests}


def test_duplicate_scalar_resolution_preserves_repeatable_args_and_last_values():
    args = [
        "--tp-size",
        "2",
        "--json-model-override-args",
        "{}",
        "--mem-fraction-static=0.85",
        "--context-length",
        "262151",
        "--mem-fraction-static",
        "0.6",
        "--tp=1",
        "--json-model-override-args",
        '{"a":1}',
    ]
    resolved = PROBE["normalize_scalars"](args)
    assert PROBE["scalar_args"](resolved) == {
        "tp_size": "1",
        "mem_fraction_static": "0.6",
        "context_length": "262151",
    }
    assert resolved.count("--mem-fraction-static") == 1
    assert resolved.count("--json-model-override-args") == 2
    assert args[0:2] == ["--tp-size", "2"]


@pytest.mark.parametrize("bad", [0, 1, float("nan"), float("inf"), -0.1])
def test_fraction_rejected_before_gpu(bad):
    with pytest.raises(ValueError, match="fractions"):
        PROBE["prepare_fields"](fields(), 0.8, bad)


def test_overrides_leave_source_and_thresholds_unchanged():
    source = fields()
    before = copy.deepcopy(source)
    prepared = PROBE["prepare_fields"](source, 0.8, 0.6)
    assert source == before
    assert (
        prepared["bench"]["correctness"]["thresholds"]
        == source["bench"]["correctness"]["thresholds"]
    )
    assert (
        PROBE["scalar_args"](prepared["bench"]["serve_args"])["mem_fraction_static"]
        == "0.8"
    )
    assert (
        PROBE["scalar_args"](prepared["bench"]["correctness"]["serve_args"])[
            "mem_fraction_static"
        ]
        == "0.6"
    )


@pytest.mark.parametrize("target", ["serve_args", "correctness"])
def test_tp2_generation_or_scorer_is_rejected(target):
    f = fields()
    args = (
        f["bench"]["serve_args"]
        if target == "serve_args"
        else f["bench"]["correctness"]["serve_args"]
    )
    args.extend(["--tp-size", "2"])
    with pytest.raises(ValueError, match="TP1/PP1"):
        PROBE["prepare_fields"](f, 0.8, 0.6)


def test_context_override_cannot_undo_scorer_headroom():
    f = fields()
    f["bench"]["correctness"]["serve_args"] += ["--context-length", "262144"]
    with pytest.raises(ValueError, match="undo scorer headroom"):
        PROBE["prepare_fields"](f, 0.8, 0.6)


def test_longest_inputs_use_explicit_shorter_fallback_without_padding():
    f = fields()
    source = source_trace(f)
    for capacity in (False, True):
        result = PROBE["longest_trace"](
            source, f, count=1, prefixes="distinct", capacity=capacity
        )
        assert result["requests"][0]["input_tokens"] == 16384
        assert result["requests"][0]["max_tokens"] == 5120
        assert result["requests"][0]["sampling"]["ignore_eos"] is capacity
        assert "sampling" not in result["meta"]  # diagnostic, not a qualified v4 trace
    fallback = PROBE["longest_trace"](
        source, f, count=2, prefixes="distinct", capacity=True
    )
    assert [r["input_tokens"] for r in fallback["requests"]] == [16384, 8192]
    assert fallback["meta"]["shorter_fallbacks"] == [
        {"request_id": "probe-001", "source_request_id": "hf-2", "input_tokens": 8192}
    ]
    assert fallback["requests"][1]["prompt"] == source["requests"][2]["prompt"]
    with pytest.raises(ValueError, match="not enough distinct"):
        PROBE["longest_trace"](source, f, count=5, prefixes="distinct", capacity=True)
    result = PROBE["longest_trace"](
        source, f, count=32, prefixes="repeated", capacity=True
    )
    assert len({r["id"] for r in result["requests"]}) == 32
    assert len({r["input_ids_sha256"] for r in result["requests"]}) == 1
    assert source == source_trace(f)


def test_over_limit_envelope_rejected_before_gpu():
    f = fields()
    source = source_trace(f)
    f["bench"]["model"]["max_model_len"] = 21504
    with pytest.raises(ValueError, match="engine reservation exceed context"):
        PROBE["longest_trace"](source, f, count=1, prefixes="distinct", capacity=True)


def test_request_uses_production_scorer_headroom_and_identical_baseline(tmp_path):
    f = PROBE["prepare_fields"](fields(), 0.8, 0.6)
    trace = PROBE["longest_trace"](
        source_trace(f), f, count=1, prefixes="distinct", capacity=True
    )
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    req = PROBE["prepare_request"](f, path)
    parsed = PROBE["validate_bench_request_dict"](req)
    starts = harness.plan_round_starts(
        parsed.engines, correctness_serve_args=parsed.correctness.serve_args
    )
    assert [s.role for s in starts] == [
        "baseline",
        "baseline-drift",
        "candidate-0",
        "scorer",
    ]
    scorer_args = PROBE["normalize_scalars"](starts[-1].spec.serve_args)
    assert PROBE["scalar_args"](scorer_args)["context_length"] == "262151"
    assert PROBE["scalar_args"](scorer_args)["mem_fraction_static"] == "0.6"
    assert req["engines"]["baseline"] == req["engines"]["candidates"][0]
    assert req["correctness"]["num_prompts"] == 1
    assert req["sla_bench"]["repetitions"] == 3
    assert req["hardware"]["gpu_count"] == 1


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"context_length": 21504},
        {"context_length": 262151, "mem_fraction_static": 0.85},
    ],
)
def test_runtime_settings_must_be_confirmed(info):
    with pytest.raises(EngineError, match="runtime"):
        PROBE["validate_runtime"](
            info, ["--context-length", "262151", "--mem-fraction-static", "0.6"]
        )


def report(**overrides):
    values = {
        "verdict": "pass",
        "num_prompts": 1,
        "num_positions_scored": 5120,
        "mean_logprob": -1.0,
        "min_logprob": -5.0,
        "quantile_logprob": -3.0,
        "coverage_ratio": 1.0,
        "evidence": "scores.jsonl",
    }
    return CorrectnessReport(**{**values, **overrides})


@pytest.mark.parametrize(
    "bad",
    [
        {"verdict": "infra_failed"},
        {"num_prompts": 0},
        {"coverage_ratio": 0.9},
        {"num_positions_scored": 0},
    ],
)
def test_incomplete_scoring_never_passes(bad):
    with pytest.raises(EngineError, match="incomplete or failed"):
        PROBE["require_reports"]({BASELINE_INDEX: report(), 0: report(**bad)}, 1)


@pytest.mark.parametrize("span", [5120, 5119])
def test_real_grade_hook_repeats_and_rejects_truncation(tmp_path, monkeypatch, span):
    f = fields()
    trace = PROBE["longest_trace"](
        source_trace(f), f, count=1, prefixes="distinct", capacity=True
    )
    pending = [
        PendingCorrectness(i, [CapturedOutput("probe-000", "p", "text", 5120, True)])
        for i in (BASELINE_INDEX, 0)
    ]
    calls = []
    globals_ = PROBE["diagnostic_hooks"].__wrapped__.__globals__

    def grade(url, items, **kwargs):
        calls.append(kwargs)
        for name in ("baseline", "candidate_0"):
            (kwargs["evidence_dir"] / f"{name}.jsonl").write_text(
                json.dumps(
                    {
                        "request_id": "probe-000",
                        "span_positions": span,
                        "scored_positions": span,
                    }
                )
                + "\n"
            )
        return {BASELINE_INDEX: report(), 0: report()}

    monkeypatch.setitem(globals_, "grade_all", grade)
    original = harness.grade_all
    with PROBE["diagnostic_hooks"](
        tmp_path, f, trace, capacity=True, timeout=1800, scorer_repetitions=3
    ) as state:
        if span != 5120:
            with pytest.raises(EngineError, match="full 5120-token"):
                harness.grade_all(
                    "local",
                    pending,
                    cfg="unchanged",
                    evidence_dir=tmp_path / "standard",
                )
        else:
            result = harness.grade_all(
                "local", pending, cfg="unchanged", evidence_dir=tmp_path / "standard"
            )
            assert result[0].verdict == "pass"
            assert state["scorer_repetitions"] == 3
            assert len(calls) == 3
            assert (tmp_path / "standard/candidate_0.jsonl").exists()
            assert all(call["cfg"] == "unchanged" for call in calls)
    assert harness.grade_all is original
    assert (tmp_path / "captured_outputs.json").exists()


def test_prepare_only_never_touches_gpu_and_preserves_failure_evidence(
    tmp_path, monkeypatch, capsys
):
    f = fields()
    source = source_trace(f)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    PROBE["save"](source_dir / "workload_trace.json", source)
    receipt = {
        "sampled_trace_sha256": PROBE["sha256_file"](
            source_dir / "workload_trace.json"
        ),
        "dataset": f["sampling_rule"]["dataset"],
        "revision": f["sampling_rule"]["revision"],
        "chat_template": {
            "model_repo": f["bench"]["model"]["hf_repo"],
            "model_revision": f["bench"]["model"]["hf_revision"],
        },
    }
    PROBE["save"](source_dir / "sampling_receipt.json", receipt)
    fixture_path = tmp_path / "fields.json"
    PROBE["save"](fixture_path, f)
    globals_ = PROBE["main"].__globals__
    monkeypatch.setitem(
        globals_,
        "build_prompt_formatter",
        lambda *a, **kw: SimpleNamespace(
            receipt={"chat_template": receipt["chat_template"]},
            encode=lambda prompt: [1] * len(prompt),
        ),
    )
    monkeypatch.setitem(
        globals_,
        "verify_hardware",
        lambda *a: pytest.fail("GPU touched during preparation"),
    )
    output = tmp_path / "out"
    command = [
        "--campaign-fields",
        str(fixture_path),
        "--source-preview",
        str(source_dir),
        "--output-dir",
        str(output),
        "--prompt-count",
        "1",
        "--generation-memory-fraction",
        "0.8",
        "--scorer-memory-fraction",
        "0.6",
        "--prepare-only",
    ]
    assert PROBE["main"](command) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "prepared_only"
    assert summary["exact_21504_boundary_exercised"] is False
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == f"Probe status: prepared_only; summary: {output / 'summary.json'}"
    # Token-count forgery in a source trace is rejected before any GPU work.
    source["requests"][-1]["prompt"] = "short"
    PROBE["save"](source_dir / "workload_trace.json", source)
    receipt["sampled_trace_sha256"] = PROBE["sha256_file"](
        source_dir / "workload_trace.json"
    )
    PROBE["save"](source_dir / "sampling_receipt.json", receipt)
    command[command.index(str(output))] = str(tmp_path / "bad")
    assert PROBE["main"](command) == 1
    assert (
        "CPU verification"
        in json.loads((tmp_path / "bad/summary.json").read_text())["error"]
    )


def test_runtime_mismatch_tears_down_started_container(tmp_path, monkeypatch):
    from bench.lifecycle import BenchNetwork, EngineContainer, EngineHandle
    from bench.schemas import EngineSpec

    f = fields()
    trace = PROBE["longest_trace"](
        source_trace(f), f, count=1, prefixes="distinct", capacity=True
    )
    globals_ = PROBE["diagnostic_hooks"].__wrapped__.__globals__
    handle = EngineHandle("http://local", "cid", "name", "sha256:" + "a" * 64, "image")
    cleaned = []
    monkeypatch.setattr(EngineContainer, "__enter__", lambda self: handle)
    monkeypatch.setattr(
        EngineContainer, "__exit__", lambda self, *args: cleaned.append(self.role)
    )
    monkeypatch.setitem(globals_, "get_json", lambda *a, **kw: {"context_length": 8192})
    original = harness.EngineContainer
    with PROBE["diagnostic_hooks"](
        tmp_path, f, trace, capacity=True, timeout=1800, scorer_repetitions=3
    ):
        container = harness.EngineContainer(
            spec=EngineSpec(
                "image", ["--context-length", "262151"], {}, "/cache", "sglang"
            ),
            network=BenchNetwork(),
            role="scorer",
            gpu_count=1,
        )
        with pytest.raises(EngineError, match="runtime context_length"):
            container.__enter__()
    assert cleaned == ["scorer"]
    assert harness.EngineContainer is original
    assert (tmp_path / "runtime/scorer/server_info.json").exists()


def test_gpu_sampling_failure_cannot_be_reported_as_success(tmp_path, monkeypatch):
    globals_ = PROBE["memory_samples"].__wrapped__.__globals__
    sampled = threading.Event()

    def fail(*a, **kw):
        sampled.set()
        raise OSError("nvidia-smi unavailable")

    monkeypatch.setattr(globals_["subprocess"], "run", fail)
    with (
        pytest.raises(EngineError, match="sampling failed"),
        PROBE["memory_samples"](tmp_path),
    ):
        assert sampled.wait(2)
    assert json.loads((tmp_path / "telemetry.json").read_text())["errors"]


def test_source_preview_fills_seven_missing_16k_slots_without_balanced_quotas(
    tmp_path, monkeypatch
):
    f = fields()
    f["sampling_rule"]["n_rows"] = 35
    sizes = list(range(16000, 15975, -1)) + list(range(14000, 13993, -1))
    texts = ["x" * size for size in sizes] + ["x" * 16000, "x" * 16385, ""]
    formatter = SimpleNamespace(
        render=lambda messages: messages[1]["content"],
        encode=lambda text: [1] * len(text),
        receipt={
            "chat_template": {
                "model_repo": f["bench"]["model"]["hf_repo"],
                "model_revision": f["bench"]["model"]["hf_revision"],
            }
        },
    )
    calls = []

    def fetch(rule, index):
        calls.append(index)
        return {
            "messages": [
                {"role": "user", "content": "write"},
                {"role": "assistant", "content": texts[index]},
            ]
        }

    globals_ = PROBE["build_source_preview"].__globals__
    monkeypatch.setitem(globals_, "build_prompt_formatter", lambda *a, **kw: formatter)
    monkeypatch.setitem(globals_, "fetch_hf_row", fetch)
    root = tmp_path / "source"
    PROBE["build_source_preview"](f, root, 32)
    assert calls == list(range(35))
    receipt = json.loads((root / "sampling_receipt.json").read_text())
    assert receipt["eligible_16k_prompts"] == 25
    assert receipt["eligible_distinct_prompts"] == 32
    assert receipt["diagnostic_only"] is True
    assert receipt["sampled_trace_sha256"] == PROBE["sha256_file"](
        root / "workload_trace.json"
    )
    source = json.loads((root / "workload_trace.json").read_text())
    trace = PROBE["longest_trace"](
        source, f, count=32, prefixes="distinct", capacity=True
    )
    assert [r["input_tokens"] for r in trace["requests"]] == sizes
    assert len(trace["meta"]["shorter_fallbacks"]) == 7
    assert len({r["input_ids_sha256"] for r in trace["requests"]}) == 32
    assert all(r["max_tokens"] == 5120 for r in trace["requests"])
    assert all(r["sampling"]["ignore_eos"] for r in trace["requests"])
    with pytest.raises(ValueError, match="even with shorter fallback"):
        PROBE["build_source_preview"](f, tmp_path / "insufficient", 33)


def test_model_volume_copies_verifies_reuses_and_rewrites(tmp_path, monkeypatch):
    source = tmp_path / "weights"
    source.mkdir()
    (source / "config.json").write_text("{}")
    volume = PROBE["DockerModelVolume"](tmp_path)
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
    volume = PROBE["DockerModelVolume"](tmp_path)
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


def test_diagnostic_metrics_accept_no_gaps_without_claiming_goodput():
    from bench.sla_bench import aggregate_rep_metrics

    rows = [
        {
            "request_id": "one",
            "completion_tokens": 5,
            "ttft_ms": 1,
            "e2e_ms": 2,
            "itl_ms": [],
        }
    ]
    kwargs = {"wall_s": 1, "p99_ttft_ms": 100, "p99_itl_ms": 100}
    with pytest.raises(EngineError, match="no inter-token"):
        aggregate_rep_metrics(rows, **kwargs)
    result = aggregate_rep_metrics(rows, **kwargs, require_token_timing=False)
    assert result["sla_goodput_ratio"] == 0
    assert result["output_tokens_per_s"] == 5


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "CUDA error: device-side assert",
        "out of memory",
        "Traceback (most recent call last):",
    ],
)
@pytest.mark.parametrize("processor", ["mimo_audio", "mimo_v2", "other_processor"])
@pytest.mark.parametrize("context_label", ["User-specified", "Target model's"])
def test_log_review_known_warnings_preserves_real_failures(
    tmp_path, suffix, processor, context_label
):
    logs = tmp_path / "round/evidence/correctness/engine_logs"
    logs.mkdir(parents=True)
    content = """Warning: User-specified context_length (262151) is greater than the derived context_length (262144). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.
Ignore import error when loading sglang.srt.multimodal.processors.mimo_audio: Could not load libtorchcodec.
[start of libtorchcodec loading traceback]
Traceback (most recent call last):
OSError: libavutil.so.60 missing
[end of libtorchcodec loading traceback]
"""
    (logs / "scorer.log").write_text(
        content.replace("mimo_audio", processor).replace(
            "User-specified", context_label
        )
        + suffix
    )
    if suffix:
        with pytest.raises(EngineError, match="scorer.log:7"):
            PROBE["review_engine_logs"](tmp_path)
    else:
        result = PROBE["review_engine_logs"](tmp_path)
        assert result["status"] == "passed"
        assert len(result["warnings"]) == 2
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("context_label", ["User-specified", "Target model's"])
def test_log_review_accepts_kimi_scorer_headroom_warning(tmp_path, context_label):
    logs = tmp_path / "round/evidence/correctness/engine_logs"
    logs.mkdir(parents=True)
    (logs / "scorer.log").write_text(
        f"[2026-10-10 01:00:00 TP0] Warning: {context_label} context_length (1048583) is greater than the derived context_length (1048576). This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config.\n"
    )
    result = PROBE["review_engine_logs"](tmp_path, 1048576)
    assert result["status"] == "passed"
    assert [w["kind"] for w in result["warnings"]] == [
        "scorer_context_headroom_warning"
    ]
    # Any other requested context is not the scorer's headroom.
    with pytest.raises(EngineError, match="scorer.log:1"):
        PROBE["review_engine_logs"](tmp_path)


def test_log_review_does_not_ignore_incomplete_optional_traceback(tmp_path):
    logs = tmp_path / "round/evidence/correctness/engine_logs"
    logs.mkdir(parents=True)
    (logs / "scorer.log").write_text(
        "Ignore import error when loading sglang.srt.multimodal.processors.mimo_audio: Could not load libtorchcodec.\n"
        "[start of libtorchcodec loading traceback]\nTraceback (most recent call last):\n"
    )
    with pytest.raises(EngineError):
        PROBE["review_engine_logs"](tmp_path)


@pytest.mark.parametrize("message", ["CUDA error: device-side assert", "out of memory"])
def test_log_review_rejects_runtime_error_inside_optional_block(tmp_path, message):
    logs = tmp_path / "round/evidence/correctness/engine_logs"
    logs.mkdir(parents=True)
    (logs / "scorer.log").write_text(
        "Ignore import error when loading sglang.srt.multimodal.processors.mimo_v2: Could not load libtorchcodec.\n"
        "[start of libtorchcodec loading traceback]\n"
        + message
        + "\n[end of libtorchcodec loading traceback]"
    )
    with pytest.raises(EngineError):
        PROBE["review_engine_logs"](tmp_path)


KIMI_SERVE_ARGS = [
    "--served-model-name", "Kimi-K3", "--tp", "8", "--mem-fraction-static", "0.88",
    "--max-running-requests", "64", "--enable-cache-report", "--enable-metrics",
    "--trust-remote-code", "--tool-call-parser", "kimi_k3", "--dcp-size", "8",
    "--max-mamba-cache-size", "320", "--speculative-algorithm", "DSPARK",
    "--speculative-draft-model-path", "/draft",
    "--speculative-dspark-block-size", "3", "--enable-linear-replayssm-spec",
    "--watchdog-timeout", "3600", "--reasoning-parser", "kimi_k3",
    "--cuda-graph-backend-prefill", "breakable", "--cuda-graph-max-bs-prefill", "4608",
]  # fmt: skip


def kimi_fields():
    value = fields()
    value["gpu_skus"] = ["B300"]
    value["bench"]["gpu_count"] = 8
    value["bench"]["model"].update(
        hf_repo="moonshotai/Kimi-K3",
        hf_revision="f831ab66814297da540d832a5235f8e904f29d06",
        quantization=None,
        max_model_len=1048576,
    )
    value["bench"]["serve_args"] = list(KIMI_SERVE_ARGS)
    value["bench"]["correctness"]["serve_args"] = ["--mem-fraction-static", "0.80"]
    return value


def test_kimi_k3_profile_keeps_tp8_and_dspark_draft_mount(tmp_path):
    f = PROBE["prepare_fields"](kimi_fields(), 0.88, 0.80)
    scorer = PROBE["scalar_args"](
        f["bench"]["serve_args"] + f["bench"]["correctness"]["serve_args"]
    )
    assert scorer["tp_size"] == "8"
    assert scorer["mem_fraction_static"] == "0.8"
    trace = PROBE["longest_trace"](
        source_trace(f), f, count=1, prefixes="distinct", capacity=True
    )
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    req = PROBE["prepare_request"](f, path)
    assert req["hardware"]["gpu_count"] == 8
    assert "--speculative-draft-model-path" in req["engines"]["baseline"]["serve_args"]


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f["bench"]["serve_args"].extend(["--tp", "4"]),
        lambda f: f["bench"]["serve_args"].extend(
            ["--speculative-draft-model-path", "RadixArk/Kimi-K3-DSpark"]
        ),
        lambda f: f["bench"].update(gpu_count=4),
        lambda f: f.update(gpu_skus=["H200"]),
    ],
)
def test_kimi_k3_profile_rejects_other_shapes(change):
    f = kimi_fields()
    change(f)
    with pytest.raises(ValueError, match="TP8|draft model path|GPU|reviewed profile"):
        PROBE["prepare_fields"](f, 0.88, 0.80)


def test_kimi_k3_draft_is_staged_and_mounted_read_only(tmp_path, monkeypatch):
    from bench.lifecycle import BenchNetwork, EngineContainer
    from bench.schemas import EngineSpec

    f = PROBE["prepare_fields"](kimi_fields(), 0.88, 0.80)
    trace = PROBE["longest_trace"](
        source_trace(f), f, count=1, prefixes="distinct", capacity=True
    )
    draft = tmp_path / "draft"
    draft.mkdir()
    globals_ = PROBE["diagnostic_hooks"].__wrapped__.__globals__
    staged = []
    monkeypatch.setitem(
        globals_,
        "stage_weights",
        lambda model: (
            staged.append(model)
            or SimpleNamespace(path=draft, weights_sha256="sha256:" + "b" * 64)
        ),
    )
    commands = []

    def enter(self):
        self.runner(["docker", "run", "-d", "image"])
        self.runner(["docker", "run", "-v", "/staged:/draft:ro", "image"])
        raise EngineError("stop after launch command")

    monkeypatch.setattr(EngineContainer, "__enter__", enter)
    monkeypatch.setattr(EngineContainer, "__exit__", lambda self, *args: None)
    with PROBE["diagnostic_hooks"](
        tmp_path, f, trace, capacity=True, timeout=1800, scorer_repetitions=3
    ):
        container = harness.EngineContainer(
            spec=EngineSpec("image", [], {}, "/cache", "sglang"),
            network=BenchNetwork(),
            role="baseline",
            gpu_count=8,
            runner=lambda cmd, **kw: commands.append(cmd),
        )
        with pytest.raises(EngineError, match="stop after launch"):
            container.__enter__()
    assert [(m.hf_repo, m.hf_revision) for m in staged] == [
        ("RadixArk/Kimi-K3-DSpark", "3c5bac301d9cf392706189d82ed947feca6c2f0f")
    ]
    assert commands == [
        [
            "docker",
            "run",
            "-v",
            f"{draft}:/draft:ro",
            "-d",
            "image",
        ],
        ["docker", "run", "-v", "/staged:/draft:ro", "image"],
    ]
    with pytest.raises(ValueError, match="draft model"):
        with PROBE["diagnostic_hooks"](
            tmp_path / "volume",
            f,
            trace,
            capacity=True,
            timeout=1800,
            scorer_repetitions=3,
            docker_model_volume=True,
        ):
            pass


def test_hardware_check_matches_profile_gpu_count(tmp_path, monkeypatch):
    globals_ = PROBE["verify_hardware"].__globals__
    rows = "\n".join(
        f"{i}, GPU-{i}, NVIDIA B300 SXM6 AC, 275040, 580" for i in range(8)
    )
    monkeypatch.setattr(
        globals_["subprocess"], "run", lambda *a, **kw: SimpleNamespace(stdout=rows)
    )
    profile = PROBE["PROFILES"][("B300", "moonshotai/Kimi-K3")]
    PROBE["verify_hardware"](tmp_path, profile)
    with pytest.raises(ValueError, match="exactly 1 RTX PRO 6000"):
        PROBE["verify_hardware"](
            tmp_path, PROBE["PROFILES"][("RTXPRO6000", "Qwen/Qwen3.8-27B-FP8")]
        )
