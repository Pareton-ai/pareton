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


## 2026-09-22: Keep every patch private permanently

Supersede all delayed-publication and legacy-public exemptions at the user's
request. Public API reads never publish patches, disclose download links, or
schedule a reveal. Remove the public-copy implementation and timer logic.
Retain PARETON_PATCH_REVEAL_DELAY_S with a ten-year default only for rollback
compatibility; it has no effect on public access in this version.

Failure reasons (round_entries.disqualify_reason and report reason strings)
are null on public routes unless the entry's status is scored, because worker
exception strings can contain Python tracebacks quoting lines from the miner's
patched source. Blocking the patch download alone does not prevent that
disclosure. The filter is read-time only; stored evidence is unchanged. Raw
build/engine logs stay private wherever they surface; the remaining miner
diagnostics (event details, evidence references, job errors) stay public with
patch URLs masked inside them. Serving diagnostics behind authentication is a
separate follow-up.

The privacy rollout is not complete at the API: existing public S3 objects
remain downloadable directly until the bucket policy deploys, and candidate
container images carry patched source. Deploy authenticated legacy readers
before restricting S3 access, verify anonymous reads fail for patch and
evidence objects and candidate images, purge cached public artifacts where
applicable, and confirm internal workers still fetch patches and pull images
afterward. Previously downloaded copies cannot be recalled.
