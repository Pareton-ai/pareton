# LongWriter campaign workload

The Qwen SGLang launch helper uses `zai-org/LongWriter-6k` at revision
`0db15c0624f19d63e2efe1021595af933cc5b6cc` (6000 rows). Sampler version 4
renders the original `user` and `assistant` messages as conversation history,
then appends the pinned `followup_prompt` as a new user turn. The follow-up asks
for a new complete work of approximately 4,000-6,000 words. Thinking is disabled.
The source answer is context, not a reference target for the new response.

Each 32-request round contains eight inputs in each tier. Length includes the
entire rendered conversation and generation prefix, measured by the pinned
model tokenizer:

| Tier | Input tokens | Requests |
| --- | --- | --- |
| 2k | 1844-2048 | 8 |
| 4k | 3687-4096 | 8 |
| 8k | 7373-8192 | 8 |
| 16k | 14746-16384 | 8 |

Rows outside these bands are skipped. Source messages are never padded or
truncated. There is no 32k tier or fallback when a tier cannot be filled.
Versions 1 through 3 keep their existing behavior and receipt formats.

The model, engine, hardware, serving arguments, 32-request workload, 2 ms
arrivals, fees and emissions remain as configured in the seed helper.
`max_tokens=5120` remains a ceiling. Requests respect EOS and never set a
minimum generation length. A long source answer does not establish that the
model will produce a long new response.

## Qualify the source pool

Run qualification on the Linux Docker host serving the trusted baseline, using
the exact model, published image and serving arguments from
`ops/seed-sglang-qwen38-27b.sh`. Publish the container port on loopback, for example
`-p 127.0.0.1:8000:8000`, and pass that container's name or ID. The endpoint must
expose `/v1/completions`, `/tokenize` and `/server_info`. Use a dedicated endpoint:
qualification submits real inference work. From the repository root:

```bash
python -m bench.qualify_longform \
  --base-url http://127.0.0.1:8000 \
  --container "$BASELINE_CONTAINER" \
  --engine-ref "$NATIVE_ENGINE_REF" \
  --output-dir /workspace/longwriter-qualification \
  --pool-size 64 --repetitions 2 --concurrency 4
```

The qualifier inspects the running container through the local Docker socket
(`/var/run/docker.sock`). It verifies that the loopback endpoint matches a
published TCP port and that the container's image ID has the requested immutable
repository digest. The recorded image reference comes from inspected image
metadata. Missing digests, mismatches, stopped containers, host networking,
remote endpoints and HTTP proxy routing fail qualification. Container identity,
start time and image are checked again before the qualified rule is written.
Run as a user with access to that socket; remote Docker contexts and SSH tunnels
are not supported by this verification path.

The evidence records the inspected container and image identity. This trusts the
local Docker host and does not provide hardware attestation. The operator must
still use the campaign's pinned model and serving settings; the qualifier checks
input tokenization and capacity against the server. It indexes source rows in a
fixed hash order, skips malformed rows and inputs outside the tier bands, and
rejects duplicate rendered prompts. It qualifies tiers in descending order
(16k, 8k, 4k, 2k), using batches of at most `--concurrency` rows (default 4).
Repetitions for each row remain sequential. After each batch it stops if the
remaining rows cannot fill that tier. Per-response logs report token counts and
rejection reasons; evidence writes are serialized. Use `--concurrency 1` for
serial qualification. Changing concurrency does not alter the source prompts
or relax the output checks. The default pool contains 64 rows, with 16
qualified rows per tier. An explicit pool size must be a multiple of four and
at least 32. Extra rows in one tier cannot replace missing rows in another.
Each accepted row must produce at least 3000 tokens
in every repetition, with nonempty text that passes the harness's repetition
checks. Generation stops normally or reaches its 5120-token ceiling. Reaching
the ceiling without suppressing EOS qualifies the row, but does not establish
where it would naturally end with a larger allowance.

`qualification.jsonl` records the contract and generated responses for review.
`sampling_rule.json` pins the accepted row indices and hashes the campaign settings.
`summary.json` records the scope of the run and an evidence hash for manual audit.
Downstream checks detect stale settings; they do not authenticate the qualification
or verify the evidence file. Use a rule produced by the trusted operator's qualifier.
Older rules containing `qualification.evidence_sha256` remain accepted; that field
is diagnostic only. There is no automatic semantic-quality judge in this step;
review the saved outputs and run
the full correctness benchmark before launch. If fewer than the requested pool
size qualify, evidence is retained but no launch rule is written. Use a fresh
directory for another attempt; failures never silently fall back to short rows.

This is bounded-concurrency output qualification. It does not validate the
full campaign arrival pattern, latency, correctness, or candidate performance. Stop the qualification
baseline container to release the GPUs, then run the standalone sample with
the qualified rule and a fresh directory:

```bash
bash ops/sglang-sample-round/run.sh \
  /workspace/pareton-sample-round-longwriter \
  /workspace/longwriter-qualification/sampling_rule.json
```

The standalone runner uses the fixture's published baseline image. Qualify that
same image when using this runner. Inspect `bench_report.json` and its evidence.
Every measured baseline and drift repetition must still meet the output-length
floor. If one stops short under load, the round fails as a baseline workload
error. It does not force continuation, drop the request or penalize a miner.
Candidate outputs continue through the existing length, correctness and timing
checks.

## Open the campaign

After qualification and full GPU validation, run on the configured controller:

```bash
bash ops/seed-sglang-qwen38-27b.sh "$NATIVE_ENGINE_REF" "$INITIAL_FEE_TAO" \
  /workspace/longwriter-qualification/sampling_rule.json
```

All three arguments are required, including the qualified rule file. The fixture
rule remains a source template for qualification and CPU previews. Every v4
campaign status, including draft, rejects missing or stale qualification before
inserting a profile or campaign. A different model, image, serving configuration
or sampling rule requires requalification.

Rounds sample distinct rows from the frozen eligible pool using the chain seed.
Receipts pin selected rows, the follow-up, history answer hashes, rendered input token hashes,
tokenizer/template metadata and the trace hash. Worker replay fetches only the
selected rows and fails if the source or rendering contract changes. No database
migration is required, and existing campaigns are not rewritten.

## Natural-output repetition enforcement

For every retained correctness prompt, every measured natural-output repetition
must pass the absolute repetition checks on its full reconstructed text, including
text beyond the baseline's output length. This also applies when generation hits
the 5120-token ceiling. A looping non-median repetition fails correctness even
when the latency-median response is clean. The policy applies to all normal-EOS
workloads, not only LongWriter; it is deliberately stricter than median-only
grading. Logprob checks and existing candidate length checks still use the
latency-median response. Repetition checks inspect only the newly generated
response, excluding the rendered user/assistant history and follow-up. The same
output policy applies across the 2k, 4k, 8k and 16k input tiers; input length does
not change the repetition thresholds or grant a forced-tail exemption.

Correctness evidence records per-repetition outcomes in `repetition_degeneracy`;
full texts remain in the corresponding SLA `rep_N/requests.jsonl` evidence. The
existing character n-gram and repeated-span thresholds, including the thinking /
answer split, are unchanged. These are heuristic repetition checks, not a
semantic-quality guarantee or a detector for every short repeated passage.

The existing baseline instability policy is unchanged: if any measured natural
baseline repetition is degenerate, that correctness prompt is excluded for all
engines with an audited reason. More than four exclusions fail the round; an
empty retained set cannot pass. Qualification already rejects degenerate outputs,
but qualification alone does not establish behavior under concurrent round load.

Speculative decoding receives the same text checks. Stream deltas are joined
before grading; token counts and chunk boundaries cannot truncate the inspected
text. Existing timing validation rejects streams with fewer inter-chunk gaps than
reported completion tokens require, so some coalesced speculative streams remain
incompatible with SLA timing. Forced-tail diagnostic exemptions require both a
forced baseline probe reference and the original request's `ignore_eos=true`;
normal-EOS requests cannot inherit them.

## Inspect inputs on a CPU VM

From an installed repository checkout, run:

```bash
python -m bench.preview_longform --output-dir /workspace/longwriter-preview
column -t -s "$(printf '\t')" /workspace/longwriter-preview/index.tsv
less /workspace/longwriter-preview/hf-001.messages.json
less /workspace/longwriter-preview/hf-031.prompt.txt
```

The command uses the same seed calculation, formatter and sampler as campaign
rounds, then verifies exact receipt replay. The default seed is a deterministic
demo. To reproduce a particular round, pass its `--campaign-id`, `--seed-block`,
`--block-hash`, campaign fields and qualified `--sampling-rule` file.
The output directory must be new. This CPU command downloads only source data
and tokenizer files, not model weights, and performs no inference or DB writes.

`index.tsv` lists input tier, total input tokens, and `history_answer_tokens`.
The latter counts the source assistant message now included in the input; it
is not a generated output length. Each request has both a `.messages.json` file
with its three roles and a `.prompt.txt` file with the exact rendered model input.
`workload_trace.json` and `sampling_receipt.json` retain the full replay contract.

A CPU audit of all 6000 source rows with the pinned Qwen tokenizer and this
follow-up found 185 eligible 2k, 506 eligible 4k, 216 eligible 8k and 25 eligible
16k inputs. A 32-request trace filled every tier and replayed exactly. These
counts establish input availability only. The 16k pool has limited diversity;
GPU qualification must still establish how many rows generate long responses.
