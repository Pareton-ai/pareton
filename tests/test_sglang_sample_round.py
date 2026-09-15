"""Standalone NVFP4 launch settings match a round with the same model. Offline."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models

from bench.main import scorer_engine_spec
from bench.sampler import build_prompt_formatter
from bench.schemas import EngineSpec
from campaign.models import SLA
from worker.round_job import build_round_request

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("embedded", [False, True])
def test_sample_request_matches_production_launch_and_scorer(
    tmp_path, monkeypatch, embedded
):
    cache_root = tmp_path / "hf-cache"
    model_cache = (
        cache_root
        / "RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead"
        / "009632fef96dd349150baa780c984e62e70e91fe"
    )
    model_cache.mkdir(parents=True)
    template = (
        "{% for m in messages %}{{ m.content }}{% endfor %}"
        "{% if enable_thinking %}<think>{% endif %}\r\n"
    )
    tokenizer_config = {"model": "nvfp4"}
    if embedded:
        tokenizer_config["chat_template"] = template
    else:
        (model_cache / "chat_template.jinja").write_bytes(template.encode("utf-8"))
    (model_cache / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    tokenizer_json = Tokenizer(
        models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
    ).to_str()
    (model_cache / "tokenizer.json").write_text(tokenizer_json)
    monkeypatch.setattr("config.BENCH_HF_CACHE_DIR", cache_root)

    def check_cached_tokenizer(rule, **kwargs):
        assert kwargs["model_repo"] == "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"
        assert kwargs["model_revision"] == "009632fef96dd349150baa780c984e62e70e91fe"
        assert kwargs["config_loader"]() == {
            "model": "nvfp4",
            "chat_template": template,
        }
        assert kwargs["tokenizer_loader"]() == tokenizer_json
        formatter = build_prompt_formatter(rule, **kwargs)
        assert "Explain binary search." in formatter.render("Explain binary search.")
        assert formatter.receipt["chat_template"]["sha256"] == (
            "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()
        )
        return formatter

    trace_path = tmp_path / "workload_trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "meta": {"name": "offline-sample-test"},
                "requests": [
                    {
                        "id": f"p{i}",
                        "prompt": "Explain binary search.",
                        "arrival_offset_ms": i * 2,
                        "max_tokens": 8,
                        "sampling": {"temperature": 0, "top_p": 1},
                    }
                    for i in range(32)
                ],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["prepare.py", str(tmp_path)])
    monkeypatch.setattr("bench.sampler.build_prompt_formatter", check_cached_tokenizer)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: "sha256:" + "a" * 64
    )
    runpy.run_path(
        str(ROOT / "ops/sglang-sample-round/prepare.py"), run_name="__main__"
    )
    request = json.loads((tmp_path / "bench_request.json").read_text())
    fields = json.loads(
        (ROOT / "fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json").read_text()
    )
    assert request["model"] == {
        "hf_repo": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
        "hf_revision": "009632fef96dd349150baa780c984e62e70e91fe",
        "dtype": "bfloat16",
        "quantization": "modelopt_mixed",
        "max_model_len": 262144,
    }
    assert request["model"] == fields["bench"]["model"]
    production = build_round_request(
        {
            "gpu_sku": "RTX5090",
            "sampled_trace_sha256": request["workload_trace"]["sha256"],
            "scoring_rule": fields["scoring_rule"],
        },
        SimpleNamespace(
            bench=fields["bench"], engine=fields["engine"], sla=SLA(**fields["sla"])
        ),
        [
            {"role": role, "engine_image_ref": "sha256:" + "a" * 64}
            for role in ("baseline", "candidate")
        ],
        task_id=request["task_id"],
        trace_path=str(trace_path),
    )
    assert request["engines"] == production["engines"]
    for engine in [request["engines"]["baseline"], *request["engines"]["candidates"]]:
        args = engine["serve_args"]
        for flag, value in (
            ("--model-path", "/model"),
            ("--context-length", "262144"),
            ("--dtype", "bfloat16"),
            ("--kv-cache-dtype", "bfloat16"),
        ):
            assert args[args.index(flag) + 1] == value
        assert args[args.index("--quantization") + 1] == "modelopt_mixed"
    scorer = scorer_engine_spec(EngineSpec.from_dict(request["engines"]["baseline"]))
    assert (
        scorer.serve_args[scorer.serve_args.index("--kv-cache-dtype") + 1] == "bfloat16"
    )
    assert (
        scorer.serve_args[scorer.serve_args.index("--quantization") + 1]
        == "modelopt_mixed"
    )
    assert (
        scorer.serve_args[scorer.serve_args.index("--context-length") + 1] == "262151"
    )
    assert scorer.env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"
    assert (
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
        not in request["engines"]["baseline"]["env"]
    )
