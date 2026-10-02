# Affine corpus workload

Sampler version 5 (`type: affine_corpus`) replays agent turns from Affine's
public corpus. It is a workload source for new campaigns, starting with the
Affine leader on H100. Versions 1 through 4 and their receipts are unchanged.

## Source contract

Affine publishes schema 3 at `https://data.affine.io`: a mutable pointer
(`corpus/manifest.json`), immutable manifest revisions
(`corpus/manifests/{sha256}.json`), a Parquet turn index and gzip JSONL chunks
of `duel_turns@v4` view records. A rule pins one immutable revision:

| Field | Meaning |
| --- | --- |
| `base_url` | https origin of the public bucket |
| `manifest_sha256` | SHA-256 of the manifest revision's bytes |
| `index_sha256`, `n_turns` | Index object and row count named by that manifest |
| `corpus_epoch` | Epoch recorded in that manifest |
| `max_prefix_chars` | Index-level history size cap, applied before tokenization |
| `sources`, `action_kinds` | Optional sorted allowlists; absent means all |
| `n_prompts`, `max_tokens`, `request_interval_ms`, `enable_thinking` | Round shape |
| `temperature` or `temperature_range` | Optional, same semantics as version 4 |

The sampler never reads the pointer, so a newly published corpus cannot change
an existing campaign. Each object is verified before use: the manifest by the
hash of its bytes, the index by the hash of its Parquet bytes, and each chunk by
the hash of its **uncompressed** JSONL, which is the hash the manifest
publishes. Only active `view_v4` chunks are accepted, and every index row must
point at one. Downloads are size-capped (128 MiB compressed, 1 GiB inflated per
chunk) and stay on https.

Verified objects are cached by content hash under `PARETON_AFFINE_CACHE_DIR`
(default `~/.cache/pareton/affine-corpus`) and re-verified on every load.
Chunks stay gzip on disk. Epoch 103 is about 1.4 GiB compressed and 5 GiB
inflated; a round touches only the chunks of the turns it inspects.

`fixtures/workloads/affine_corpus_e103/sampling_rule.json` pins epoch 103,
published 2026-10-01: 300,209 indexed turns in 183 chunks.

`cpu-qualification-summary.json` beside it records a CPU qualification of that
corpus with the Qwen/Qwen3.8-27B tokenizer at a 32,768-token context: 198,609
turns pass the index filters; of 2,048 scanned, 1,785 render inside the tiers
(514 / 297 / 505 / 469 for 2k / 4k / 8k / 16k) and 263 exceed 16,384 tokens.
The scan took about 3.5 minutes on a warm cache, and one round draw with
replay about 45 seconds. It is evidence for the source contract, not a launch
rule for any campaign.

## Turns

An index row names a reply node in one view record. Its prompt is the
root-to-parent path through the record's message graph, rendered with the
campaign's pinned chat template and tokenizer. The reply is Affine's reference
and is never sent; its hash is kept in the receipt. Sibling branches, later
nodes and other roots are not on the path. Tool calls and tool results are
already baked into message text. Nothing in the corpus is executed.

A turn is rejected when its record is not `duel_turns@v4`, disagrees with its
index row (trajectory, turn meta, node, or prefix length), has a broken or
cyclic graph, an unknown role, non-text content, a target that is not an
assistant reply, or a history that does not end on a user message. Turn IDs
that appear more than once in the index are ambiguous and excluded.

## Sampling and replay

The round seed orders every eligible turn by `sha256(seed:turn_id)`. Turns are
rendered in that order until four contiguous input tiers are full:

| Tier | Input tokens | Requests per 32 |
| --- | --- | --- |
| 2k | 1-2048 | 8 |
| 4k | 2049-4096 | 8 |
| 8k | 4097-8192 | 8 |
| 16k | 8193-16384 | 8 |

Input tokens count the full rendered history and generation prefix. Turns
outside the tiers or the model context are skipped, as are prompts identical
to one already drawn. There is no shorter fallback. Requests use natural EOS
and the rule's `max_tokens` ceiling; Pareton's correctness and scoring apply.
Affine's duel score is not imported.

The receipt records the rule, the template and tokenizer pins, the context,
the selected turn IDs, and per request the chunk key and hash, line, node,
source, action kind, phase, history length, input token hash and reference
hash. Worker replay rebuilds the rule from the receipt, re-fetches only the
selected turns and fails unless the trace and receipt reproduce exactly.

## Preview and qualify on CPU

Both commands take campaign fields with `bench`, `engine` and `sampling_rule`,
the same shape as `fixtures/campaigns/*/campaign-fields.json`.

```bash
python -m bench.qualify_affine preview \
  --campaign-fields campaign-fields.json --output-dir /workspace/affine-preview

python -m bench.qualify_affine qualify \
  --campaign-fields campaign-fields.json --output-dir /workspace/affine-qualification \
  --scan-turns 2048
```

`preview` draws one round with the campaign sampler, verifies exact receipt
replay, and writes the trace, receipt, `index.tsv`, and per request the exact
prompt and its messages.

`qualify` scans a fixed, seed-ordered slice of the eligible pool with the
campaign tokenizer. `qualification.jsonl` records the contract and, per turn,
its source, action kind, phase, input tokens, tier and rejection reason.
`summary.json` records corpus pins, pool composition, token distributions by
tier, rejection counts, and a demo draw with its selected turn IDs and exact
replay. Each tier must have at least `n_prompts` eligible turns in the scan,
otherwise evidence is kept and no rule is written. The written rule binds
`qualification.contract_sha256` to the rule, `bench` and `engine`; campaign
seeding rejects a missing or stale contract for every status.

Neither command runs inference. Output length, correctness under load, GPU
memory and performance on the target H100 are separate launch gates.

## Rollout

No database migration is needed. Existing campaigns, rules and receipts are
untouched. Workers that replay an Affine round need https access to
`data.affine.io` and cache space. To move to a newer corpus, pin a new
manifest revision, requalify, and seed a new campaign; do not edit an open
campaign's rule.
