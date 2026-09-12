# Stage-2 isolated acceptance evidence — 2026-09-12

Environment: macOS host, Docker Desktop, image `pareton-systemd-verify`
(Ubuntu 24.04, systemd, python3.12, Vector 0.57.0), container
`--privileged --cgroupns=host`. Source: stage2 worktree (branch
`marcus/stage2-release-safety`) staged without `.git`, re-initialized as a
fresh repo inside the container. Script:
`ops/isolated-acceptance/stage2-acceptance.sh` (unmodified run).

Result (first run, pre-CR commit 2887328): 40/40 with an os.chdir-side effect
that a later commit removed — invalid as evidence for the committed tree (see
the 2026-09-12 CR, P0-2).

**Re-run after the CR fixes (same day, no drop-ins, committed units):**
40/40 assertions PASS, final line `ALL-STAGE2-ISOLATED-ACCEPTANCE-PASSED`,
exit 0 — including the deploy unit's own WorkingDirectory/EnvironmentFile
driving the DB probe, finite TimeoutStartSec=4h, full-tree release-scope
classification, stopped-state semantics, request lifecycle (refusals end as
failed; busy rollback continues), started_at at tick entry, verify-from-idle
entering verifying, and maintenance-timer restore on log failure. Key observed behaviors:

- S1 bootstrap: `request reset` + one deploy run created state under hold,
  drained, applied, re-exec'd to the installed entrypoint, started all five
  resident units (stand-ins running the real probe loop), and the first
  verify failed exactly on `notification-acceptance-required` (rc=1) with
  business healthy.
- S2 drill: corrupt-state deploy failed (rc=1) → OnFailure notifier ran →
  structured `deploy_failure_notified` event reached the sink →
  `record-notification-acceptance` accepted it (Axiom evidence verified) →
  `request verify` completed (rc=0), state idle, hold kept.
- S3: `request unpause` cleared hold; the A→B release (requirements change)
  ran drain→quiesce→apply→re-exec→verify end to end (rc=0): recovery copy
  kept, checkout at B, `.deploy-done` alias == verified commit,
  `maintenance_finished` shipped, all 7 per-unit probes received.
- S4 gate matrix: `applying` → ExecCondition exit 1 skips the start without
  failing (inactive/success); verifying+started allows startup; corrupt
  state → exit 255 → unit failed with `release_gate_error` in the journal.
- S5: a live holder of the shared activity lock (real
  `worker/coordination.py` claim guard) made the tick record
  `last_step=active-work`, return 0, and write nothing.
- S6: valid one-shot GPU request consumed → warning + probe, real reap
  skipped; consumed request → real reap ran (marker file).
- S7: mock Axiom dropping one source → checker exit 1 naming the unit.

Local unit suites the same day: `pytest -q -m "not docker and not e2e"` →
1378 passed / 0 failed / 44 deselected. Not covered here (production-only
per spec §9): B1 real bench, B12 real rollback on the VPS, B14 real Discord
reception, B20 production bootstrap, §8.1 VM reboot drills (T1–T6).

## S8 rollback round (same day, third run)

Reviewer follow-up: the 40/40 did not exercise `restore_recovery_venv` or a
real recovery copy. Added S8 (rollback after a verified A->B release, with a
B-era marker planted in the live venv). Building it exposed and fixed two
real gaps:

- A rollback after a SUCCESSFUL release targeted `verified_commit` — which
  by then equals the current commit — so it would have restored the
  previous release's venv onto unchanged code (old deps, new code). The
  state now records `recovery_commit` (the commit the copy captures) and
  rollback targets it.
- `from_commit` at release start was taken from HEAD, which tooling or a
  crashed attempt may already have moved; it is now the last verified
  baseline (the original deploy.sh's `DEPLOYED` semantics).

Result: **47/47 assertions PASS** (40 + S8's 7), exit 0, no drop-ins. S8
observed: rollback registered and completed rc=0; state idle with
verified_commit back at A and hold kept; checkout and `.deploy-done` back
at A; the B-era venv marker gone (recovery copy restored, not pip);
B-era file gone from the checkout. Unit suite same day: 1396 passed /
0 failed (new: rollback-target and missing-venv regressions).

## Fourth-round follow-up (same day, fourth run)

Reviewer observations 1-4 (all non-blocking) implemented:

- hold.baseline_commit now anchors to the rollback target (read from the
  recovery copy's own recovery-meta.json when recovery_commit is absent —
  the pre-field state shape), so `status` can no longer read as a no-op;
  observed in-run: hold.baseline == verified == A after the rollback.
- S8 gained positive evidence: an A-era marker planted before the A->B
  release rides into the recovery copy and must survive the rollback
  (proving the venv contents came from the copy, not "nothing changed").
- Runbook and spec 6.3 document the anchor semantics, including that a
  vector-only fast path does not refresh the recovery anchor (a rollback
  after it returns to the previous FULL release's baseline; undoing a TOML
  change means publishing the reverse edit or vector-repair).

Result: **48/48 assertions PASS**, exit 0, no drop-ins. Unit suite same
day: 1397 passed / 0 failed (new: hold-anchor + meta-fallback regression).
