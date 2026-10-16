"""Move a qualified Kimi K3 C16 rule to the 32-prompt C32 contract without GPU work.

Qualification is a per-row property: each pooled row produced enough clean output
at both temperature endpoints. It does not depend on how many rows a round draws
or how many run at once. This keeps the qualified pool and repetitions unchanged
and only rewrites n_prompts, request_concurrency, max_baseline_prompt_drops and
the correctness prompt count, then re-hashes the contract against the fixture.

It first proves the source rule was qualified against exactly the current K3
fixture apart from those fields. A pool of 32 rows (8 per tier) means every
round replays all of them in a seed-dependent order with new temperatures.
Run a C32 shadow round before seeding a campaign from the output.

  PYTHONPATH=. python ops/restamp_k3_c32.py OLD_RULE.json NEW_RULE.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from bench.longform import qualification_contract, require_qualification
from bench.sampler import SamplerError, parse_sampling_rule

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "fixtures/campaigns/sglang_kimi_k3_b300/campaign-fields.json"
SOURCE_PINS = {
    "n_prompts": 16,
    "request_concurrency": 16,
    "max_baseline_prompt_drops": 4,
}
SOURCE_CORRECTNESS_PROMPTS = 16


def require(condition, message):
    if not condition:
        raise SamplerError(f"restamp refused: {message}")


def restamp(old_rule, fields):
    old = parse_sampling_rule(old_rule)
    require("qualification" in old, "source rule is not qualified")
    new = parse_sampling_rule(fields["sampling_rule"])
    require(
        "qualification" not in new and "eligible_row_indices" not in new,
        "target must be the unqualified campaign fixture",
    )
    bench, engine = fields["bench"], fields["engine"]
    source_bench = copy.deepcopy(bench)
    source_bench["correctness"]["num_prompts"] = SOURCE_CORRECTNESS_PROMPTS
    source_rule = {**new, **SOURCE_PINS}
    unqualified = {
        k: v
        for k, v in old.items()
        if k not in ("qualification", "eligible_row_indices")
    }
    require(
        unqualified == source_rule,
        "source rule differs from the C16 K3 contract beyond the C32 fields",
    )
    # The stored hash must match the fixture as it was, so nothing else drifted.
    require_qualification(old, source_bench, engine)
    pool = old["eligible_row_indices"]
    tiers = len(new["input_tiers"])
    require(
        len(pool) >= new["n_prompts"] and len(pool) % tiers == 0,
        f"pool of {len(pool)} rows cannot fill {new['n_prompts']} prompts",
    )
    new["eligible_row_indices"] = list(pool)
    new["qualification"] = {
        **old["qualification"],
        "contract_sha256": qualification_contract(new, bench, engine),
    }
    require_qualification(new, bench, engine)
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output file already exists")
    new = restamp(json.loads(args.source.read_text()), json.loads(FIELDS.read_text()))
    args.output.write_text(json.dumps(new, indent=2) + "\n")
    print(
        f"Restamped {len(new['eligible_row_indices'])} qualified rows for "
        f"{new['n_prompts']} prompts at C{new['request_concurrency']}; "
        f"contract {new['qualification']['contract_sha256']}. "
        "No GPU requests made. Run a C32 shadow round before seeding.",
        flush=True,
    )


if __name__ == "__main__":
    main()
