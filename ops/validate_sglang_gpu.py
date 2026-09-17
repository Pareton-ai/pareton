"""Run native-patch and full-context probes on an already provisioned GPU host.

This does not provision hardware or create a campaign. Use the normal full-round
harness afterwards to validate FP8 correctness, timing and reproducibility.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from bench.http import post_json
from bench.lifecycle import BenchNetwork, EngineContainer
from bench.main import scorer_engine_spec
from bench.validate import load_bench_request
from bench.weights import stage_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--probe-image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    req, _ = load_bench_request(args.request)
    if req.engines.baseline.name != "sglang":
        raise ValueError("SGLang request required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--gpus",
            "all",
            "--ipc=host",
            "--entrypoint",
            "python",
            args.probe_image,
            "-m",
            "sglang.pareton_build_probe",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    (args.output_dir / "native-probe.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    print("CUDA, JIT and Rust mutation probes passed", flush=True)

    weights = stage_weights(req.model, token_env=req.hf_token_env)
    scorer = scorer_engine_spec(req.engines.baseline)
    with (
        BenchNetwork() as network,
        EngineContainer(
            scorer,
            network,
            role="context-boundary-probe",
            gpu_count=req.hardware.gpu_count,
            weights_dir=weights.path,
            pull=False,
            logs_dir=args.output_dir,
        ) as engine,
    ):
        # Exact input IDs exercise the reported full-window failure independently
        # of tokenizer compression or an early EOS during sampled generation.
        ids = [42] * req.model.max_model_len
        response = post_json(
            engine.base_url,
            "/generate",
            {
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": 0,
                "return_text_in_logprobs": True,
            },
            timeout=180,
        )
        rows = response["meta_info"]["input_token_logprobs"]
        assert len(rows) == len(ids), (len(rows), len(ids))
        assert [row[1] for row in rows] == ids
        assert all(math.isfinite(row[0]) for row in rows[1:])
        assert response["meta_info"]["completion_tokens"] == 1
        evidence = {
            "model": req.model.hf_repo,
            "model_revision": req.model.hf_revision,
            "engine_image": scorer.image,
            "replay_context": req.model.max_model_len,
            "scorer_serve_args": scorer.serve_args,
            "context_override": scorer.env.get(
                "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
            ),
            "input_positions": len(ids),
            "scored_positions": len(rows) - 1,
            "last_token_logprob": rows[-1][0],
            "output_tokens": response["meta_info"]["completion_tokens"],
            "verdict": "pass",
        }
        (args.output_dir / "context-boundary.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
        print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
