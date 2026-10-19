"""CPU-only tests for the scorer-only smoke."""

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from bench.lifecycle import EngineError
from bench.sampler import PromptFormatter

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "scorer_smoke", ROOT / "ops/scorer_smoke.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
pytestmark = pytest.mark.unit

FIELDS = ROOT / "fixtures/campaigns/sglang_kimi_k3_b300_exgentic/campaign-fields.json"
WORDS = PromptFormatter(lambda m, **kw: "", {}, lambda text: text.split())


@pytest.fixture(autouse=True)
def restore_health_timeout(monkeypatch):
    monkeypatch.setattr(config, "BENCH_HEALTH_TIMEOUT_S", config.BENCH_HEALTH_TIMEOUT_S)


def prepared_fields(tmp_path, n_rows=64):
    fields = json.loads(FIELDS.read_text())
    fields["sampling_rule"]["n_rows"] = n_rows
    path = tmp_path / "campaign-fields.json"
    path.write_text(json.dumps(fields))
    return fields, path


def candidate(index, session, tokens, tier="128k"):
    return {
        "row_index": index,
        "prompt": f"prompt {index}",
        "input_tokens": tokens,
        "input_ids_sha256": f"sha256:{index}",
        "input_length_group": tier,
        "session_id": session,
    }


def test_continuation_has_exactly_the_requested_tokens():
    text = smoke.continuation(WORDS, "one two three", 10)
    assert len(text.split()) == 10


def test_selection_keeps_the_longest_top_tier_prompt_per_session(monkeypatch, tmp_path):
    fields, _ = prepared_fields(tmp_path)
    rows = [
        candidate(0, "a", 120000),
        candidate(1, "a", 130000),  # a later, longer call of session a
        candidate(2, "b", 125000),
        candidate(3, "c", 60000, tier="64k"),
        None,
        candidate(5, "d", 118000),
    ] + [None] * 58
    monkeypatch.setattr(smoke, "candidate_for_row", lambda i, *a: rows[i])
    tier, sequences = smoke.select_sequences(
        fields, 2, formatter=WORDS, row_fetcher=lambda i: {}
    )
    assert tier == "128k"
    assert [(s["row_index"], s["session_id"]) for s in sequences] == [
        (1, "a"),
        (2, "b"),
    ]
    assert sequences[0]["total_tokens"] == 130000 + 5120
    assert len(sequences[0]["output_text"].split()) == 5120
    with pytest.raises(smoke.CheckFailed, match="only 3 distinct 128k prompts"):
        smoke.select_sequences(fields, 4, formatter=WORDS, row_fetcher=lambda i: {})


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    fields, path = prepared_fields(tmp_path)
    sequences = [
        {
            "id": f"smoke-{i}",
            "input_tokens": 130000 - i,
            "continuation_tokens": 5120,
            "total_tokens": 135120 - i,
            "prompt": "p",
            "output_text": "o",
        }
        for i in range(2)
    ]
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "sequences.json").write_text(
        json.dumps(
            {
                "campaign_fields_sha256": smoke.sha256_file(path),
                "tier": "128k",
                "sequences": sequences,
            }
        )
    )
    h = SimpleNamespace(
        path=path,
        prepared=prepared,
        info={"max_req_input_len": 200000, "max_total_num_tokens": 120000},
        scored=[],
        span=5100,
        starts=[],
        gpus_checked=False,
    )

    def check_gpus(sku, count):
        h.gpus_checked = True
        return [sku] * count

    monkeypatch.setattr(smoke, "check_gpus", check_gpus)
    scorer = SimpleNamespace(spec=SimpleNamespace(serve_args=["--tp", "8"]))
    req = SimpleNamespace(
        model=SimpleNamespace(hf_repo="m"), hf_token_env="HF_TOKEN", draft_model=None
    )
    monkeypatch.setattr(smoke, "scorer_request", lambda f, root: (req, scorer))
    monkeypatch.setattr(
        smoke, "stage_weights", lambda *a, **kw: SimpleNamespace(path=tmp_path)
    )

    class Provider:
        def __init__(self, **kwargs):
            self.draft_dir = None

        @contextmanager
        def start(self, start, *, phase):
            h.starts.append(start)
            yield "http://scorer"

    monkeypatch.setattr(smoke, "_EngineProvider", Provider)
    monkeypatch.setattr(smoke, "get_json", lambda url, path, timeout: h.info)
    monkeypatch.setattr(smoke, "probe_logprob_capability", lambda url, timeout: {})

    def score(url, captured, *, request_timeout_s, engine_name):
        assert request_timeout_s == 300.0 and engine_name == "sglang"
        if h.span is None:
            raise EngineError("scorer died")
        h.scored.append(captured.request_id)
        logprob = SimpleNamespace(logprob=-1.0)
        return [logprob] * h.span, max(h.span, 5100), ""

    monkeypatch.setattr(smoke, "score_captured_output", score)

    @contextmanager
    def no_samples(path):
        yield {"gpu0": {"peak_used_mib": 250000.0}}

    monkeypatch.setattr(smoke, "memory_samples", no_samples)
    return h


def run(h, tmp_path, *extra):
    out = tmp_path / "out"
    code = smoke.main(
        [
            "--campaign-fields",
            str(h.path),
            "--output-dir",
            str(out),
            "--prepared",
            str(h.prepared),
            *extra,
        ]
    )
    return code, json.loads((out / "summary.json").read_text())


def test_scores_every_sequence_on_one_scorer_start(gpu, tmp_path, capsys):
    code, summary = run(gpu, tmp_path)
    assert code == 0, summary
    assert summary["status"] == "passed"
    assert len(gpu.starts) == 1
    assert gpu.scored == ["smoke-0", "smoke-1"] * 2
    assert summary["capacity"]["longest_sequence_tokens"] == 135120
    assert summary["capacity"]["headroom_tokens"] == 200000 - 135120
    assert (
        capsys.readouterr()
        .out.splitlines()[-1]
        .startswith("Scorer smoke status: passed; summary: ")
    )


def test_a_sequence_beyond_the_scorer_pool_fails_before_scoring(gpu, tmp_path):
    gpu.info["max_req_input_len"] = 135120
    code, summary = run(gpu, tmp_path)
    assert code == 1
    assert "longest sequence is 135120" in summary["failures"][0]
    assert gpu.scored == []


def test_a_partly_scored_continuation_fails(gpu, tmp_path):
    gpu.span = 4000
    code, summary = run(gpu, tmp_path)
    assert code == 1
    assert "scored 4000 of 5100 positions" in summary["failures"][0]


def test_a_scorer_failure_is_an_engine_error(gpu, tmp_path):
    gpu.span = None
    code, summary = run(gpu, tmp_path)
    assert code == 3
    assert summary["error"] == "EngineError: scorer died"


def test_prepare_only_needs_no_gpu(gpu, tmp_path):
    code, summary = run(gpu, tmp_path, "--prepare-only")
    assert (code, summary["status"]) == (0, "prepared_only")
    assert not gpu.gpus_checked and not gpu.starts


def test_prepared_sequences_must_match_the_campaign_fields(gpu, tmp_path):
    fields = json.loads(gpu.path.read_text())
    fields["bench"]["correctness"]["serve_args"][-1] = "8"
    gpu.path.write_text(json.dumps(fields))
    code, summary = run(gpu, tmp_path)
    assert code == 2
    assert "other campaign fields" in summary["error"]
