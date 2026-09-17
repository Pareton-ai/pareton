"""Create inspectable LongWriter inputs using the campaign round sampler, on CPU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from uuid import UUID

from bench.longform import source_messages
from bench.sampler import (
    LONGFORM_ALGO_VERSION,
    SamplerError,
    build_prompt_formatter,
    compute_sample_seed,
    fetch_hf_row,
    generate_trace,
    parse_sampling_rule,
    sampling_context_for_rule,
)


def preview(*, fields, output_dir, campaign_id, seed_block, block_hash):
    rule = parse_sampling_rule(fields["sampling_rule"])
    if rule["algo_version"] != LONGFORM_ALGO_VERSION:
        raise SamplerError("preview_longform requires algo_version 4")
    model = fields["bench"]["model"]
    formatter = build_prompt_formatter(
        rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
    )
    kwargs = {
        "rule": rule,
        "seed_hex": compute_sample_seed(block_hash=block_hash, campaign_id=campaign_id),
        "row_fetcher": lambda i: fetch_hf_row(rule, i),
        "prompt_formatter": formatter,
        "sample_seed_block": seed_block,
        "sample_seed_block_hash": block_hash,
        "sampling_context": sampling_context_for_rule(
            rule, fields["bench"], fields["engine"]
        ),
    }
    sampled = generate_trace(**kwargs)
    rebuilt = generate_trace(**kwargs, sampling_receipt=sampled.receipt)
    if rebuilt.body != sampled.body:
        raise SamplerError("preview receipt replay changed trace bytes")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "workload_trace.json").write_bytes(sampled.body)
    (root / "sampling_receipt.json").write_text(
        json.dumps(sampled.receipt, indent=2) + "\n"
    )
    trace = json.loads(sampled.body)
    with (root / "index.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            (
                "request_id",
                "dataset_row",
                "input_tier",
                "input_tokens",
                "history_answer_tokens",
                "max_tokens",
            )
        )
        for request, source in zip(
            trace["requests"], sampled.receipt["requests"], strict=True
        ):
            name = request["id"]
            writer.writerow(
                (
                    name,
                    source["row_index"],
                    request["input_length_group"],
                    request["input_tokens"],
                    source["history_answer_tokens"],
                    request["max_tokens"],
                )
            )
            (root / f"{name}.prompt.txt").write_text(request["prompt"])
            messages = source_messages(fetch_hf_row(rule, source["row_index"]), rule)
            (root / f"{name}.messages.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2) + "\n"
            )
    return sampled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-fields",
        type=Path,
        default=Path("fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json"),
    )
    parser.add_argument("--sampling-rule", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign-id", type=UUID, default=UUID("00000000-0000-0000-0000-000000000001")
    )
    parser.add_argument("--seed-block", type=int, default=1)
    parser.add_argument("--block-hash", default="c" * 64)
    args = parser.parse_args(argv)
    fields = json.loads(args.campaign_fields.read_text())
    if args.sampling_rule:
        fields["sampling_rule"] = json.loads(args.sampling_rule.read_text())
    preview(
        fields=fields,
        output_dir=args.output_dir,
        campaign_id=args.campaign_id,
        seed_block=args.seed_block,
        block_hash=args.block_hash,
    )
    print(
        f"CPU preview and exact receipt replay complete: {args.output_dir / 'index.tsv'}"
    )
    print("This does not qualify natural output length or GPU performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
