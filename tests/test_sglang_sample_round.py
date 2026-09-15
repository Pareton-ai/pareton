"""Standalone NVFP4 launch settings match a round with the same model. Offline."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.main import scorer_engine_spec
from bench.schemas import EngineSpec
from campaign.models import SLA
from worker.round_job import build_round_request

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_sample_request_matches_production_launch_and_scorer(tmp_path, monkeypatch):
    cache_root = tmp_path / "hf-cache"
    model_cache = (
        cache_root
        / "nvidia--Qwen3.8-27B-NVFP4"
        / "dbb8f445b3145f8a4c18ddc769f032d57d32867c"
    )
    model_cache.mkdir(parents=True)
    (model_cache / "tokenizer_config.json").write_text('{"model": "nvfp4"}')
    (model_cache / "tokenizer.json").write_text("cached-nvfp4-tokenizer")
    monkeypatch.setattr("config.BENCH_HF_CACHE_DIR", cache_root)

    def check_cached_tokenizer(rule, **kwargs):
        assert kwargs["model_repo"] == "nvidia/Qwen3.8-27B-NVFP4"
        assert kwargs["model_revision"] == "dbb8f445b3145f8a4c18ddc769f032d57d32867c"
        assert kwargs["config_loader"]() == {"model": "nvfp4"}
        assert kwargs["tokenizer_loader"]() == "cached-nvfp4-tokenizer"

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
        "hf_repo": "nvidia/Qwen3.8-27B-NVFP4",
        "hf_revision": "dbb8f445b3145f8a4c18ddc769f032d57d32867c",
        "dtype": "bfloat16",
        "quantization": None,
        "max_model_len": 262144,
    }
    fields["bench"]["model"] = request["model"]
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
        ):
            assert args[args.index(flag) + 1] == value
        assert "--quantization" not in args
    scorer = scorer_engine_spec(EngineSpec.from_dict(request["engines"]["baseline"]))
    assert "--quantization" not in scorer.serve_args
    assert (
        scorer.serve_args[scorer.serve_args.index("--context-length") + 1] == "262151"
    )
    assert scorer.env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"
    assert (
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
        not in request["engines"]["baseline"]["env"]
    )
