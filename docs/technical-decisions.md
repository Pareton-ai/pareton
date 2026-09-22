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

## 2026-09-22: Keep every patch private permanently

Supersede all delayed-publication and legacy-public exemptions at the user's
request. Public API reads never publish patches, disclose download links, or
schedule a reveal. Remove the public-copy implementation and timer logic.
Retain PARETON_PATCH_REVEAL_DELAY_S with a ten-year default only for rollback
compatibility; it has no effect on public access in this version.

Keep legacy objects and on-chain locators unchanged. Use authenticated S3 reads
for every accepted patch locator, then restrict patch and evidence objects to
production's AWS account with a scoped bucket-policy deny. Preserve traces and
other public campaign artifacts. Deploy authenticated readers before applying
the policy; bucket policy and registry visibility are operator-managed, not
side effects of starting the API. No schema or audit-event mutation is needed.

Withhold build logs, raw submission event details, event evidence references,
and job errors through the public API because they can contain patch source.
Preserve public states, hashes, and score metrics. Linear PAR-99's evidence
boundary was reviewed; its older proposed report subset does not override the
current scoring UI or this explicit permanent-privacy request. Existing private
GHCR access remains the enforcement boundary for candidate image contents.
See docs/patch-privacy.md for storage rollout order and verification.
