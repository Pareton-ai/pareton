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
- Every managed live file was captured from the 2026-09-10 read-only audit
  and matches the repo copy (worker main unit + both drop-ins recorded
  verbatim; `vector.service`, round-worker, gpu-reap, api, watcher, weights,
  deploy service/timer, builder-cleanup all byte-identical). The only
  intentional live diffs at bootstrap: the two new drop-ins, the deploy unit's
  `OnFailure=` line, the new `pareton-deploy-failed.service`, the ops
  programs, and the Vector TOML (env-ref token + two added include_units).
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

   `apply` validates candidates first — `systemd-analyze verify` per staged
   unit (resolving `ExecStart` against the live filesystem, which is why step 5
   installs the ops programs before this step) and `vector validate` with the
   real env — then installs atomically, runs one `daemon-reload`, restarts
   Vector when its files changed, and re-checks. Rollback copies are under
   `/var/lib/pareton-deploy/sync-backup/`. The old inline-token TOML is the
   pre-install backup — keep it in the restricted location.
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

---

# Stage-2 runbook: release coordination, rollback, recovery

Implements section 10 of `docs/第二阶段发布安全-spec.md`. The stage-2 entry
`ops/deploy.sh` is a thin wrapper around `/usr/local/lib/pareton-ops/release.py
tick`; the release state machine owns both coordination locks
(`/run/pareton-deploy.lock` deploy mutex, `/run/pareton-activity.lock` worker
activity lock) and every deployment write path. State lives in
`/var/lib/pareton-deploy/release-state.json` (schema and gate matrix: spec
section 4.5); `.deploy-done` is a compat alias rewritten from state.

## S0. Read-only checks (no window needed)

```sh
/usr/local/lib/pareton-ops/release.py status
systemctl show pareton-deploy.service -p TimeoutStartUSec   # infinity after stage-2
journalctl -u pareton-deploy -n 50 --no-pager
```

`status` prints the release state, any registered request, the drill record,
and the current probe file. Exit 2 means the state file is corrupt/missing.

## S1. Requests (spec 6.3)

Requests other than `hold` are registered under the deploy mutex and executed
by a `pareton-deploy.service` run (start it manually, or wait for the timer):

```sh
R=/usr/local/lib/pareton-ops/release.py
$R request hold   --reason "investigating" --operator NAME   # immediate + timer disable --now
$R request unpause --main-commit <origin/main SHA> --operator NAME
$R request reset  --baseline-commit <verified SHA> --confirm-evidence "<what was checked>" \
                  [--recovery-copy /var/lib/pareton-deploy/recovery/<ts>] --operator NAME
$R request resume --operator NAME
$R request cancel --reason "wrong target" --operator NAME
$R request verify --operator NAME
$R request rollback --reason "target broken" --operator NAME
$R request vector-repair --target <COMMIT> --operator NAME
systemctl start pareton-deploy.service      # executes the registered request
```

Rules that matter operationally:

- `hold` takes effect immediately (short state lock); a running install is
  never killed and finishes first. It also runs
  `systemctl disable --now pareton-deploy.timer` as a second layer.
- `verify` never clears hold; `rollback`/`cancel` set hold themselves.
  Only `unpause` clears it, and only when the phase is idle and the given
  `--main-commit` still equals `origin/main`.
- One request at a time; a failed request stays on disk as failed until the
  next registration replaces it.
- `reset` archives the corrupt state as `release-state.corrupt.<ts>` (a
  `.missing` marker when absent) and requires operator evidence plus a
  baseline commit; it rebuilds under hold and re-verifies like a release.

## S2. Residual submission records (spec 5.1)

A deploy tick exits non-zero with step `submission-record-unresolved` (or
`round-record-unresolved` / `db-probe-error`) when the activity lock is free
but database records remain. Rounds void automatically via the watcher's
stale reaper once it runs; submissions have no reaper and need this CLI:

```sh
/opt/pareton/.venv/bin/python /opt/pareton/ops/recover_submission.py inspect --job <ID>
/opt/pareton/.venv/bin/python /opt/pareton/ops/recover_submission.py recover \
    --job <ID> --attempt <N> --outcome requeue --operator NAME --reason "..."
```

`recover` requires the deploy mutex, a closed gate (draining/quiescing) and
the exclusive activity lock — run the deploy tick first so it reports the
record. Outcomes: `requeue` (no durable evidence; job returns to pending and
may rebuild) or `settle` (terminal reject evidence → failed; bench_queued/
scored → done). Contradictory evidence refuses. The lock being free does
not prove Docker builds or remote GPU resources ended — confirm externally.

## S3. Failure drill and acceptance record (spec 7.4)

After the first stage-2 install (and after any change to
`ops/notify-deploy-failure.py`, `ops/deploy.sh`, `ops/release.py`,
`ops/ops_common.py`, the deploy/deploy-failed units, `vector.service`, or
non-exempt `vector.toml` fields), run a real failure drill in an authorized
window with the timer disabled:

1. Induce a real deploy failure (for example a corrupt state file — the B21
   path — restored immediately afterwards).
2. Confirm the Discord channel received the alert; note the message id.
3. Register the evidence (queries Axiom for the deploy-failed event of that
   invocation; refuses without it):

   ```sh
   $R record-notification-acceptance --invocation <INVOCATION_ID> \
       --message-id <DISCORD_MESSAGE_ID> --confirmed-by NAME
   ```

4. `request verify` + one deploy run completes the release. Pure
   `include_units` member additions (keeping `pareton-deploy-failed.service`)
   are the single exemption and do not need a new drill.

## S4. Manual escalation for a stuck stop (spec 5.2)

API/watcher/weights have `TimeoutStopSec=infinity`; a genuinely hung stop is
an authorized human decision, never a timer:

1. `$R request hold` and `systemctl disable --now pareton-deploy.timer`;
   confirm no in-flight install.
2. Record the unit, `MainPID`, cgroup (`systemctl status <unit>`), recent
   logs, and any associated tasks/resources.
3. After confirming there is no in-flight work to protect — or the owner
   explicitly accepts interrupting it — kill exactly that one unit:

   ```sh
   systemctl kill --signal=SIGKILL pareton-<unit>.service
   ```

   Never use wildcards; never kill the deploy/notifier chain to bypass state
   checks.
4. Handle residual records per S2, then restore services through the
   coordination entries above.

## S5. What looks like failure but is not

- `drain-wait` non-zero ticks after 30 minutes: the deploy unit enters
  failed on purpose while a long bench drains; the notifier rate-limits.
  The state stays in draining and the next tick continues waiting.
- ExecCondition skips during applying/quiescing: the unit is inactive and
  NOT failed; services do not auto-start after a skip — the release starts
  them explicitly at verify.
- Heartbeat-absent pages during a maintenance window are real: the worker
  parks (keeps heartbeating while only the claim gate closes, but the
  process is stopped during applying/verifying). `maintenance_started`/
  `maintenance_finished` events in Axiom correlate the window.
