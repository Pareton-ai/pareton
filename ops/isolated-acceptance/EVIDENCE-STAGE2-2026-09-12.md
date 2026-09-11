# Stage-2 isolated acceptance evidence — 2026-09-12

Environment: macOS host, Docker Desktop, image `pareton-systemd-verify`
(Ubuntu 24.04, systemd, python3.12, Vector 0.57.0), container
`--privileged --cgroupns=host`. Source: stage2 worktree (branch
`marcus/stage2-release-safety`) staged without `.git`, re-initialized as a
fresh repo inside the container. Script:
`ops/isolated-acceptance/stage2-acceptance.sh` (unmodified run).

Result: **40/40 assertions PASS**, final line `ALL-STAGE2-ISOLATED-ACCEPTANCE-PASSED`,
exit 0. Key observed behaviors:

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
