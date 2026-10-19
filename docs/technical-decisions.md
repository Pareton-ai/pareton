# Technical decisions

## 2026-09-21: Allow four baseline-relative repetition flags per candidate

Keep the per-prompt 0.10 maximum drop in distinct character-16-gram ratio, but
allow up to four distinct retained prompt IDs to exceed it in one candidate's
round evaluation. The fifth flag disqualifies the candidate. Count prompts,
not repetitions; baseline-excluded prompts and diagnostic forced tails do not
consume the allowance. This is a fixed count, including when baseline exclusions
reduce the usual 32-prompt cohort. Keep all retained prompts in likelihood and
performance scoring. Absolute repetition, empty-output, likelihood, coverage
and timing-stability checks retain their existing behavior.

Retain every per-prompt flag and add the failure count, IDs and allowance to the
correctness report. A fifth known flag survives a later scorer error; tolerated
flags alone do not convert infrastructure failure into disqualification.

Linear tools were unavailable. This policy follows the user's explicit request.
Historical repairs must verify complete saved evidence and all remaining gates,
recompute performance and leadership, and append corrective submission events.


## 2026-09-22: Allow four repeated-span failures per candidate

Keep the longest repeated-span ceiling at 0.25, but tolerate breaches on up to
four distinct retained prompt IDs per candidate per round. The fifth affected
prompt disqualifies the candidate. Count a prompt once across all measured
repetitions, the scored prefix and the full output. Preserve every diagnostic
and publish the count, IDs and allowance in the correctness report.

This allowance is separate from the four baseline-relative repetition flags.
The distinct character-16-gram floor of 0.15 and empty outputs remain immediate
failures, including in later repetitions or the answer after a thinking span
breach. Baseline exclusions remain strict. Excluded prompts and exempt forced
tails do not consume the span allowance; non-exempt natural prefixes do.
Likelihood, coverage, timing stability and scoring use their existing rules.
Tolerated prompts remain in correctness and performance scoring.

A known fifth span failure survives a subsequent scorer error. Tolerated span
flags alone cannot turn an incomplete scorer run into a correctness failure.
Linear tools were unavailable; this policy follows the user's explicit request.


## 2026-09-23: Archive campaigns out of the default dashboard list

Add `archived` to the campaigns.status vocabulary. Archiving preserves the
campaign's history and consumed payment references while excluding it from the
default dashboard list: `GET /v1/campaigns` now defaults to `status=open`, and
the public filter accepts only `open`, `closed`, or `archived`. Explicit
`closed` and `archived` filters remain available, and direct lookup by
campaign ID still returns an archived campaign. Internal `draft` remains a
valid stored status but is not a public filter value.


## 2026-10-01: Campaign-controlled patch disclosure

Campaigns choose `patch_visibility.mode`: `private` (the default for new and
existing campaigns) or `public_after_reveal`, with `reveal_delay_s` defaulting
to 172800. The policy is stored and exposed in the manifest but excluded from
`manifest_hash`, so updates preserve commitments and customer signoffs.

The reveal clock starts at the first complete round with a scored or disqualified
entry, even for submissions without an old enrollment event. Live/void rounds,
infrastructure failures, and operator bans do not start it. Closing or archiving
a campaign does not release private patches. Explicit patch availability and
download requests publish eligible objects lazily using checksum-verified S3 copies;
ordinary list/detail reads withhold links and never perform S3 I/O;
private policy always suppresses publication and locations.

The policy applies retrospectively to existing submissions. Changing policy
cannot revoke files already published or downloaded. See
[configuration and rollout](patch-visibility.md).

## 2026-09-23: Patches stay private permanently (superseded)

Miner patches never become public. The API has no patch download routes and
never exposes retrieval locations: no `retrieval_url`, `patch_reveal_at`, or
`patch_download_url` on any submission response. Patch hashes remain the public
identifiers for submissions, on listings, details, rounds, and leader rows.

Signed upload still returns the private S3 locator to the submitting miner,
and the on-chain commitment stores that locator. It identifies the private
object; it does not grant access. The watcher, worker, and other validator
services keep credentialed S3 access for fetch and build.

Old append-only `submission_events` rows are untouched: nothing rewrites past
`committed` details. Objects already copied to public storage under the former
reveal policy are outside this code change.


## 2026-10-18: Exgentic v6 workload for a new Kimi K3 campaign

Use the pinned Exgentic agent-llm-traces-v2 corpus as independent request
snapshots. Sampler v6 normalizes successful chat calls, preserves system/tool
inputs, renders with the campaign tokenizer, and selects eight distinct prompts
per tier. Each tier uses 90-100% of its named token count.
Allow at most one snapshot from a source session in a round and in the
qualified pool. Append the campaign-pinned long-form follow-up before rendering
and measuring input tiers. Never pad, truncate, execute tools or feed candidate
output into another request. Reject unsupported modalities.

Keep the pinned Kimi serving configuration,
C32 schedule, warmed-cache measurement, likelihood gates, and baseline-relative
completion accounting. Preserve Kimi's `[0.1, 1.01]` temperature range,
natural EOS, 5120-token ceiling and 1500-token output floor. The raw completion
endpoint retains generated tool-call syntax as
text for scoring. Qualification does not establish tool execution correctness.

Prepare the exact normalized snapshot count and CPU coverage before running
the baseline qualification command. Fail without a launch rule if any tier's
qualified pool is short. Require a new qualification contract and new campaign;
preserve historical LongWriter rule/receipt behavior. No old qualification can
be restamped into this workload.


## 2026-10-18: Preserve Kimi policy when replacing the input corpus

Review against PR #193 and the subsequent C32 change in PR #203 found that the
initial Exgentic commit incorrectly selected greedy generation, a one-token
floor and a C4 qualification default. Restore the existing generation policy
and default qualification to the campaign's request concurrency. Qualify v6 in
mixed-tier batches so the 16-per-tier pool quota does not cap C32 at C16.
Keep legacy qualification scheduling unchanged. Insufficient outputs at either
temperature extreme block launch; the dataset change does not relax that gate.
GPU scorer, SLA and full-round validation remain required before opening the
new campaign. No production settings or historical evidence change here.


## 2026-10-18: Append a broad long-form task to Exgentic snapshots

The user's whole-corpus review found many long inputs but short natural agent
outputs. Append one explicit, campaign-pinned user follow-up that invites a
complete software implementation, mathematical exposition, technical tutorial,
analysis or narrative. Keep source history and tool definitions as context,
without replaying the source answer or executing tools. Ask for substantial,
self-contained output without placeholders or repetitive padding. This is a
long-context generation workload, not a replay of the original agent task.

Count the full rendered history, tool definitions and follow-up with Kimi K3's
pinned tokenizer before assigning the existing 90-100% input bands. Source-model
counts in wider power-of-two bands are inventory evidence, not proof of these
target-token quotas. Preparation and qualification scan the whole normalized
split by default; selection uses deterministic hash ordering across that split,
not a benchmark-grouped prefix. Retain one source session per round and C32.
The prompt is part of the sampling receipt and qualification contract. Missing
follow-ups fail parsing; changed follow-ups invalidate prior qualification.
Re-run the full CPU audit and baseline qualification before launch. Preserve
temperature, output-length, correctness and scoring policy.


## 2026-10-19: Size the Kimi Exgentic tiers to the DCP KV pool

The workload exists to punish engines that drop `--dcp-size 8` and so hold
about a sixth of the KV cache. At C32 with eight prompts per tier, 16k-128k
tiers put about 1.9M prompt tokens in flight, which is more than the DCP-8
baseline holds (estimated at 1.5M tokens from a C64 run at 39% KV use), so the
baseline would queue too. Use 8k/16k/32k/64k with equal weights: about 1.05M
tokens, inside the baseline pool and about four times a no-DCP pool. Sampler v6
accepts any ordered tier set, so the fixture alone carries this choice.

One session's later calls extend its earlier prompts, so two calls of a session
in different tiers would share a cached prefix. Keep one call per session per
round and per qualified pool. Different sessions of one harness can still share
system and tool prefixes, so the CPU audit reports how many prompt tokens of a
preview round a prefix cache must hold.

`campaign.carryover` enters the latest N submissions of a closed campaign into
its replacement. It copies each patch under the new campaign's prefix, keeps
the source commit block so on-chain order still decides cohorts and ties, and
reuses no payment reference, image or score. Patches that never ran, duplicate
an earlier carried patch, or belong to an excluded hotkey are skipped. A dry run
prints the plan and its hash; `--apply` requires that hash.
