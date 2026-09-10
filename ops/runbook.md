# Stage-1 runbook: config sync bootstrap, alert acceptance, hotfixes

Implements section 8 of `docs/第一阶段配置与部署告警-spec.md`. Read-only checks
are safe any time; everything under "Bootstrap" and "Acceptance" happens in an
authorized maintenance window with the deploy timer stopped. Nothing here
touches business data, rounds, or GPU state.

## 0. Facts this runbook assumes

- Target host: the authorized validator VPS (`188.166.18.9`, seen as
  `pareton-prod-02`); owner-confirmed to be the only production host, or each
  additional host gets its own recorded run of this runbook.
- The webhook for deploy alerts is `PARETON_DISCORD_DEPLOY_WEBHOOK` in
  `/opt/pareton/.env` (owner-placed). The Axiom token `PARETON_AXIOM_TOKEN`
  in the same file was verified by the owner via a direct ingest test.
- State lives under `/var/lib/pareton-deploy/`: `last-run.env` (deploy
  progress), `alert-state.json` (alert dedup), `sync-pending.json` (owed
  reloads/restarts), `sync-backup/<ts>/` (pre-install copies for rollback).

## 1. Read-only checks (no window needed)

```sh
/usr/local/lib/pareton-ops/sync-config.py check --repo /opt/pareton
/usr/local/lib/pareton-ops/notify-deploy-failure.py validate-local
```

`check` exit codes: 0 clean · 1 drift · 2 incomplete · 3 blocked (unknown
file/mask/credential prerequisite). It never writes, reloads, restarts, or
sends anything. From the SSH audit account the two credential checks report
`unverifiable` (a non-root caller cannot read the 0600 env file) — treat that
as "not verified here", not as a fault; re-run as root on the box.

## 2. Bootstrap (maintenance window, operator with write access)

1. Confirm stage-1 PRs are merged and `origin/main` is the commit to install.
2. `systemctl stop pareton-deploy.timer`, then wait for any running
   `pareton-deploy.service` to finish (`systemctl status`); stopping the timer
   does not stop an in-flight deploy.
3. Drain or pause task intake per the current service list; note worker
   pending flags. Keep unit enablement states as recorded — do not "clean up".
4. Tighten credentials (owner-approved migration): `/opt/pareton/.env` →
   `root:root 0600`; record before/after. Keep `PARETON_AXIOM_TOKEN` working —
   Vector reads it via its unit's `EnvironmentFile`.
5. Install the self-updating ops set by hand, helpers first, main entry last:

   ```sh
   install -d -m 0755 /usr/local/lib/pareton-ops
   for f in ops_common.py sync-config.py notify-deploy-failure.py; do
     install -m 0755 /opt/pareton/ops/$f /usr/local/lib/pareton-ops/$f
   done
   install -m 0755 /opt/pareton/ops/deploy.sh /usr/local/bin/pareton-deploy
   ```

6. Install units and Vector config through the sync itself (this also swaps
   the inline token for the env reference and sets the TOML to `root:root 0600`):

   ```sh
   /usr/local/lib/pareton-ops/sync-config.py apply --repo /opt/pareton
   ```

   `apply` validates candidates first (`systemd-analyze verify`, `vector
   validate` with the real env), installs atomically, runs one `daemon-reload`,
   restarts Vector when its files changed, and re-checks. Rollback copies are
   under `/var/lib/pareton-deploy/sync-backup/`. The old inline-token TOML is
   the pre-install backup — keep it in the restricted location.
7. Verify the OnFailure chain landed (content equality covers it):

   ```sh
   grep -A1 OnFailure /etc/systemd/system/pareton-deploy.service
   systemctl cat pareton-deploy-failed.service
   ```

8. Probe the alert channel (explicit test-send; does not touch dedup state)
   and have the designated receiver confirm receipt, including the message ID:

   ```sh
   /usr/local/lib/pareton-ops/notify-deploy-failure.py test-send --note bootstrap
   ```

9. Full check must return 0, Vector active, and worker pending flags recorded:

   ```sh
   /usr/local/lib/pareton-ops/sync-config.py check --repo /opt/pareton
   systemctl is-active vector
   ls /opt/pareton/.deploy-pending /opt/pareton/.deploy-rounds-pending 2>/dev/null
   ```

## 3. Acceptance: controlled deploy failure (still in the window)

1. Add a `/run`-only test drop-in that fails before any real work:

   ```sh
   printf '[Service]\nExecStartPre=/bin/false\n' \
     > /run/systemd/system/pareton-deploy.service.d/test-failure.conf
   systemctl daemon-reload
   ```

2. Trigger the real service: `systemctl start pareton-deploy.service`. It
   fails at the ExecStartPre — before fetch, config sync, and any restart —
   so nothing production-side moves.
3. The team channel must receive the alert within ~a minute; the receiver
   confirms it names this failure (exit status, step `unknown`/pre-script,
   current commits from `last-run.env` of the previous tick are acceptable
   only if labeled as such — the notifier marks unmatched runs `unknown`).
4. Remove the injection and reload; a manual `check` during the window would
   have reported the file as an unknown override — that is expected:

   ```sh
   rm /run/systemd/system/pareton-deploy.service.d/test-failure.conf
   rmdir /run/systemd/system/pareton-deploy.service.d 2>/dev/null || true
   systemctl daemon-reload
   ```

   If removal fails, keep the timer stopped and resolve before resuming — a
   leftover test drop-in fails every deploy until removed.

## 4. Closing the window

1. Run one normal deploy tick by hand: `systemctl start pareton-deploy.service`;
   it must succeed end-to-end (config sync → restarts → pending handling) and
   clear the fault state (`record-success`).
2. `systemctl start pareton-deploy.timer` (the owner decides the resume time;
   unfinished drain checks keep it stopped).
3. Record acceptance evidence: target commit, full `check` output, scan list,
   Vector delivery sample (journald vs `pareton-prod`), the controlled-failure
   service record + Discord message ID + receiver confirmation, timer state.
4. Afterward, verify new events arrive in `pareton-prod` (weights events and
   deploy logs are the natural probes). Do not force a chain weights submit.

## 5. Hotfix procedure once auto-sync is live

Live edits to managed files are reverted by the next tick. To hotfix:

1. `systemctl stop pareton-deploy.timer`; wait out any running deploy.
2. Fix the live file; keep a copy of the change.
3. Merge the fix to `main` — a pushed branch or open PR is **not** enough;
   resuming the timer before the merge reverts the fix as drift.
4. Confirm `git -C /opt/pareton fetch origin main && git rev-parse origin/main`
   contains the fix, and the live file matches the repo copy
   (`sync-config.py check` returns 0).
5. `systemctl start pareton-deploy.timer`.

## 6. Failure notes

- A failed deploy pages once per fault key; identical faults repeat at most
  every 30 minutes. A full successful tick clears the fault.
- A missing/broken webhook fails the deploy itself (visible in journald) and
  the alert for that failure cannot be delivered — check
  `journalctl -u pareton-deploy-failed.service` when alerts seem missing.
- Config faults block code deploys (fail-closed) but never stop running
  services. Roll back with the newest `sync-backup/<ts>/` copies and
  `systemctl daemon-reload` if a sync install ever misbehaves.
- `daemon-reload`/Vector-restart debts survive ticks until completed
  (`sync-pending.json`); do not delete that file to "clear" them.
