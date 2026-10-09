"""Narrow a successful four-tier PRO6000 qualification without new GPU requests.

This is an operator evidence conversion, not a fresh GPU qualification. Only the
32-to-16 prompt / four-to-two tier change and its stricter exclusion allowance
are allowed; all execution and generation pins must match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from bench.longform import (
    candidate_for_row,
    digest,
    qualification_contract,
    require_qualification,
    sampling_context_for_campaign,
)
from bench.qualify_longform import evaluate_response
from bench.sampler import (
    SamplerError,
    build_prompt_formatter,
    fetch_hf_row,
    parse_sampling_rule,
)


SOURCE_FILES = ("sampling_rule.json", "qualification.jsonl", "summary.json")


def require(condition, message):
    if not condition:
        raise SamplerError(f"qualification reuse refused: {message}")


def sha256(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def reuse(*, source_dir, output_dir, fields, formatter=None, row_fetcher=None):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    require(not output_dir.exists(), "use a fresh output directory")
    source = {name: (source_dir / name).read_bytes() for name in SOURCE_FILES}
    old = parse_sampling_rule(json.loads(source["sampling_rule.json"]))
    summary = json.loads(source["summary.json"])
    require(
        summary.get("evidence_sha256") == sha256(source["qualification.jsonl"]),
        "source evidence checksum mismatch",
    )
    records = [json.loads(line) for line in source["qualification.jsonl"].splitlines()]
    require(
        bool(records) and records[0].get("type") == "contract",
        "missing source contract",
    )
    header = records[0]
    require(
        all(r.get("type") == "response" for r in records[1:]),
        "unexpected evidence records",
    )
    require(
        summary.get("sampling_rule_sha256") == digest(old),
        "source rule checksum mismatch",
    )
    require_qualification(old, header["bench"], header["engine"])
    unqualified = {
        k: v
        for k, v in old.items()
        if k not in ("qualification", "eligible_row_indices")
    }
    require(
        parse_sampling_rule(header["sampling_rule"]) == unqualified,
        "source header differs from qualified rule",
    )

    new = parse_sampling_rule(fields["sampling_rule"])
    require(
        new.get("input_tiers") == ["8k", "16k"]
        and new["n_prompts"] == 16
        and new.get("max_baseline_prompt_drops") == 4,
        "target must be the 16-prompt 8k/16k contract with four exclusions",
    )
    require(
        "qualification" not in new and "eligible_row_indices" not in new,
        "target must be the unqualified campaign fixture",
    )
    require(
        old["n_prompts"] == 32
        and old.get("input_tiers", ["2k", "4k", "8k", "16k"])
        == ["2k", "4k", "8k", "16k"]
        and old.get("max_baseline_prompt_drops", 8) == 8,
        "source must be the original four-tier 32-prompt contract",
    )
    expected = {
        **unqualified,
        "n_prompts": 16,
        "input_tiers": ["8k", "16k"],
        "max_baseline_prompt_drops": 4,
    }
    require(
        new == expected
        and new["algo_version"] == 5
        and new["request_concurrency"] == 4
        and new["request_timeout_s"] == 600,
        "generation, dataset or replay pins changed",
    )
    old_bench = json.loads(json.dumps(fields["bench"]))
    require(
        old_bench["correctness"].get("num_prompts") == 16,
        "target correctness must use 16 prompts",
    )
    old_bench["correctness"]["num_prompts"] = 32
    require(
        header["bench"] == old_bench and header["engine"] == fields["engine"],
        "model, image, serving, hardware or correctness pins changed",
    )
    require(
        header["baseline_identity"].get("engine_ref")
        == fields["base_image_digest"]
        == fields["bench"]["baseline_engine_image_digest"],
        "baseline image identity mismatch",
    )
    repetitions = old["qualification"]["repetitions"]
    require(
        repetitions == 2 and summary.get("repetitions") == 2,
        "source must have two qualification repetitions",
    )
    require(
        header.get("concurrency") == summary.get("concurrency") == 4,
        "source qualification must use four workers",
    )
    accepted = old["eligible_row_indices"]
    require(
        len(accepted) == summary.get("qualified_rows") == 64,
        "source must contain 64 qualified rows",
    )
    by_row = defaultdict(list)
    for record in records[1:]:
        by_row[record["row_index"]].append(record)
    groups = {}
    for index in accepted:
        rows = sorted(by_row[index], key=lambda r: r["repetition"])
        require(
            [r["repetition"] for r in rows] == [0, 1],
            f"row {index}: missing or duplicate repetitions",
        )
        first = rows[0]
        for rep, record in enumerate(rows):
            require(
                all(
                    record[k] == first[k]
                    for k in ("input_length_group", "input_tokens", "input_ids_sha256")
                ),
                f"row {index}: inconsistent input identity",
            )
            require(
                record["sampling"]
                == {
                    "temperature": new["temperature_range"][rep],
                    "top_p": 1.0,
                    "seed": 0,
                },
                f"row {index}: generation settings changed",
            )
            observed = evaluate_response(
                {
                    "choices": [
                        {
                            "text": record["text"],
                            "finish_reason": record["finish_reason"],
                        }
                    ],
                    "usage": {
                        "completion_tokens": record["completion_tokens"],
                        "prompt_tokens": record["input_tokens"],
                    },
                },
                new,
                record["input_tokens"],
            )
            require(
                record.get("rejection") is None and observed["rejection"] is None,
                f"row {index}: output no longer passes qualification",
            )
        groups[index] = first["input_length_group"]
    counts = Counter(groups.values())
    require(
        counts == {"2k": 16, "4k": 16, "8k": 16, "16k": 16}
        and dict(counts) == summary.get("qualified_rows_by_input_tier"),
        "source tier quotas do not match its summary",
    )
    selected = [i for i in accepted if groups[i] in new["input_tiers"]]
    print(
        "Source evidence verified; rebuilding the 32 retained prompts with the pinned tokenizer...",
        flush=True,
    )
    model = fields["bench"]["model"]
    formatter = formatter or build_prompt_formatter(
        new, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
    )
    require(
        formatter.receipt == header["formatter"],
        "pinned tokenizer/template differs from source qualification",
    )
    row_fetcher = row_fetcher or (lambda i: fetch_hf_row(new, i))
    context = sampling_context_for_campaign(fields["bench"], fields["engine"])
    seen = set()
    for index in selected:
        candidate = candidate_for_row(
            index, row_fetcher(index), formatter, new, context
        )
        require(
            candidate is not None, f"row {index}: no longer in a selected input tier"
        )
        reference = by_row[index][0]
        require(
            all(
                candidate[k] == reference[k]
                for k in ("input_length_group", "input_tokens", "input_ids_sha256")
            ),
            f"row {index}: source prompt/tokenization changed",
        )
        require(candidate["input_ids_sha256"] not in seen, "duplicate selected prompts")
        seen.add(candidate["input_ids_sha256"])
    provenance = {
        "type": "pro6000_four_to_two_tier_qualification_reuse",
        "gpu_requests_made": 0,
        "source_contract_sha256": old["qualification"]["contract_sha256"],
        "source_files": {name: sha256(data) for name, data in source.items()},
        "retained_rows": selected,
        "scope": "Reuses successful source qualification only; a fresh 16-prompt C4 shadow round is required.",
    }
    new_header = {
        **header,
        "bench": fields["bench"],
        "sampling_rule": dict(new),
        "derivation": provenance,
    }
    evidence = (
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [
                new_header,
                *[r for r in records[1:] if r["row_index"] in selected],
            ]
        )
        + "\n"
    ).encode()
    new["eligible_row_indices"] = selected
    new["qualification"] = {
        "contract_sha256": qualification_contract(
            new, fields["bench"], fields["engine"]
        ),
        "repetitions": repetitions,
    }
    require_qualification(new, fields["bench"], fields["engine"])
    new_summary = {
        **summary,
        "evidence_sha256": sha256(evidence),
        "sampling_rule_sha256": digest(new),
        "qualified_rows": len(selected),
        "qualified_rows_by_input_tier": dict(Counter(groups[i] for i in selected)),
        "scope": provenance["scope"],
        "derivation": provenance,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source").mkdir()
    for name, data in source.items():
        (output_dir / "source" / name).write_bytes(data)
    (output_dir / "qualification.jsonl").write_bytes(evidence)
    for name, obj in (
        ("summary.json", new_summary),
        ("reuse.json", provenance),
        ("sampling_rule.json", new),
    ):
        (output_dir / name).write_text(json.dumps(obj, indent=2) + "\n")
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign-fields", type=Path, required=True)
    args = parser.parse_args()
    reuse(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        fields=json.loads(args.campaign_fields.read_text()),
    )
    print(
        "Reused 32 qualified rows (16 per selected tier); no GPU requests made. Run a fresh C4 shadow round.",
        flush=True,
    )


if __name__ == "__main__":
    main()
