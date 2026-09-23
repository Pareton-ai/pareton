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


## 2026-09-23: Patches stay private permanently

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
