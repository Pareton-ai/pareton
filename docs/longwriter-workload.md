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

New Qwen campaign fixtures pin `temperature_range=[0.1, 1.01]`. The sampler
reproducibly derives one temperature per prompt from the round seed and request
index, uniformly over the range to six decimal places. Each prompt keeps that
temperature across warmups and measured repetitions; different rounds derive
new temperatures.

New campaigns allow a maximum mean-logprob drop of `2.5` below the opening
baseline, through the same trusted scorer. Absolute likelihood floors and the
per-prompt `0.10` distinct-character-16-gram drop limit remain unchanged. This
wider likelihood tolerance is an operator-selected setting, not a measured
false-positive guarantee. Existing campaigns retain their pinned threshold.

Generation always uses `seed=0`, including qualification, warmups, opening and
second baselines, and candidates. Repetitions repeat the same sampling settings
to measure timing stability. They do not deliberately vary sampled continuations.
The round seed determines prompt selection and temperatures, not the generation
seed. Fixed sampling settings do not promise bitwise-identical engine outputs.
SLA evidence records the temperature and actual generation seed, including failed
requests. `top_p=1` and prefix-cache reuse are unchanged. The teacher-forced
correctness scorer still uses its existing scoring settings.

Version 4 rules without generation fields retain temperature zero and seed zero
and reproduce their original trace bytes and qualification hashes. Fixed
`temperature` is still supported, but cannot coexist with `temperature_range`.
Range endpoints must be increasing finite numbers between 0 and 2. Generation
policy is recorded in the rule, receipt and trace metadata; each request's settings
must match the derivation. Changing the temperature policy invalidates prior
qualification. Requalify the source pool and create
a new campaign rather than overriding an open campaign's trace at runtime. Higher
temperature can change output lengths, logprob distributions and timing variance;
run the full concurrent baseline validation before launch.

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

For a temperature range, the first two qualification repetitions test the lower
and upper endpoints. Additional repetitions use the row's derived temperature.
Every qualification repetition uses `seed=0`. Evidence records the actual
settings. Endpoint qualification does not establish
stability at every intermediate temperature or under full concurrent round load.

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
Both baseline replays run before the candidates. If any measured natural
baseline repetition is short or repetitive, exclude that prompt from correctness
and performance scoring for every candidate. Up to eight unique prompts may be
excluded across both runs; a ninth exclusion or an empty retained set fails the
round as a baseline workload error. Candidate outputs continue through the
existing length, correctness and timing checks on retained prompts.

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

For each retained correctness prompt, check every measured natural response for
absolute repetition. A looping sibling disqualifies the candidate even when a
clean response is the latency median. Logprob grading and baseline-relative checks
still use the latency-median response. The policy applies to normal-EOS workloads
across all four input tiers. Inspect only newly generated text, including text
beyond the baseline's output length; conversation history is never graded.

Correctness evidence identifies `output_selection=latency_median` for logprob and
relative grading, and records every absolute repetition result in
`repetition_degeneracy`. All generated texts remain in SLA `rep_N/requests.jsonl`
evidence. Character n-gram and repeated-span thresholds, including the
thinking/answer split, are unchanged.
An additional baseline-relative check rejects a selected response whose distinct
character-16-gram ratio is more than 0.10 below the lowest ratio from the opening
baseline's valid measured responses for that prompt. Exactly 0.10 is allowed.
The baseline and candidate metrics use the same thinking/answer split. Outputs
shorter than 64 characters retain the existing exemption. Evidence records the
reference ratio, observed drop and allowed drop, including when the scorer fails
after a known text failure. The second baseline still checks stability and shared
exclusions; it does not change the opening reference's ratios.

This whole-response heuristic detects repeated sentence templates that can clear
the absolute bars. It does not measure factual accuracy or instruction following,
and different response lengths and styles can change its value. Validate its
false-positive rate on independent baseline outputs, including code and lists,
before deploying it. Forced-tail relative findings remain diagnostic under the
existing forced-generation policy.

Baseline validation inspects all measured natural repetitions in both baseline
runs, before candidates start. Repetitive outputs and v4 outputs below 3000 tokens
exclude the prompt for every candidate, including performance scoring and the
baseline comparison. The exclusion limit is eight unique prompts across both
runs, not eight per run. A ninth exclusion or an empty retained set fails the
round. `evidence/correctness/baseline_exclusions.json` records the shared reasons;
scored entries also retain them in `score_report.excluded_prompts`. Candidate
outputs cannot trigger exclusions. Qualification alone does not establish
stability under concurrent round load.

The second baseline retains the `baseline-drift` role and `baseline_drift` report
field for compatibility. Because it now precedes candidates, the comparison
measures initial baseline repeatability, not hardware drift across the candidate
runs. The existing `PARETON_BASELINE_DRIFT_CEILING` (default `0.05`) still voids
a round when the absolute comparison exceeds the ceiling. The config name and
`baseline_drift` void reason remain compatibility names; neither implies that
hardware conditions were measured during or after candidates.
Plan version 2 in progress metadata lets the dashboard retain historical order
for old rounds and show both baselines first for new rounds.

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


## Dashboard report compatibility

`GET /v1/rounds/{round_id}/entries/{entry_id}/report` retains its existing
fields and score arithmetic. New reports add:

- `workload.temperature` or `workload.temperature_range` when explicitly pinned
  in the sampling receipt.
- `sla.sampling`: measured request ID, repetition, temperature, top-p, actual
  integer seed and EOS policy, including failed requests.
- `correctness.prompt_checks`: text-free repetition diagnostics keyed by request
  ID, including candidate and opening-baseline distinct ratios, their difference,
  the allowed drop, exclusions and the applied repetition verdict.

The last two are stored in existing report JSON. No DB migration or evidence
bundle fetch is needed. Old reports have no diagnostics; consumers must treat
missing data as unknown. Disqualified entries can have prompt checks even when
`prompts` contains no score contributions. Forced-tail diagnostics retain their
exemptions and must not be displayed as enforced full-output failures.

## Rollout on an active validator

Merge the baseline-stability parent and this change in order. Use the installed
release coordinator described in `ops/runbook.md`; let it drain the active job
before replacing the checkout and restarting services. Do not pull a new
checkout under a running round or force-stop its GPU job.

The existing campaign retains its frozen trace, temperature and seed policy.
The relative repetition guard applies to newly executed rounds after deployment,
including rounds in that campaign, so announce the stricter correctness policy
before resuming. Completed reports, scores and submission events are unchanged.
The additive API and frontend can deploy in either order; historical entries
cannot gain diagnostics without their original stored data.

To activate the temperature range, qualify a fresh pool using the new rule and
the campaign's pinned baseline image, model, GPU and serving arguments. Then
validate full concurrent baseline rounds at the new settings before seeding a
new campaign. Do not edit the ongoing campaign's rule, receipts or qualification
hash. Compare repeated candidate and baseline runs on held-out traffic before
claiming a serving speedup. Local unit tests do not perform GPU qualification.
