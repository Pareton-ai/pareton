# Independent submission and round workers

Run two execution services on the validator VPS:

| Service | Command | Claims |
| --- | --- | --- |
| `pareton-worker` | `python -m worker.main --queue submissions` | Submission gates and local image builds |
| `pareton-round-worker` | `python -m worker.main --queue rounds` | Pending rounds, GPU evaluation and scoring |

The watcher still creates rounds. Both execution services read the same database
and campaign pins. No schema migration, campaign reseed or image rebuild is needed.
Each service processes one item at a time and drains its current item on SIGTERM.

The default `--queue all` remains available for local commands and older units. It
checks rounds first. This cannot isolate a round that arrives after the same
process enters a long build, so production must install both dedicated services.

## Why a separate process is required

On 2026-09-08 a manual SGLang baseline build held the shared builder storage lock.
The worker claimed vLLM submission job 120, then blocked acquiring that lock.
Round 32 was created later, while the only execution process was already blocked.
Changing claim order would not have helped that timeline.

The round worker never claims submission jobs and never enters the hermetic
builder. It evaluates images already published by digest. The builder lock,
ccache mounts and cleanup policy keep their existing behavior. vLLM builds still
wait behind the manual SGLang build; pending rounds with built images can proceed.
Both processes still share VPS CPU and memory, so retain conservative compilation
parallelism on the validator host.

Round claims already use `FOR UPDATE SKIP LOCKED`. A round can have only one
claimant, and the database allows at most one pending or running round per
campaign. Neither queue isolation nor deployment requires changing round rows
with SQL. A round already completed by an operator must not be reset or replayed.

## Roll out on the existing VPS

Run these commands as root after this change is merged to `main`. This change has
no new Python dependencies. The checkout is `/opt/pareton`, and both units use its
existing `.venv` and `.env`.

The drop-in changes only the existing worker's command, preserving its live
restart policy, environment and timeout overrides. The SIGTERM targets only the
Python worker and requests its existing graceful drain. It leaves Docker and any
manual baseline build running. Once the current submission or round finishes,
that process exits without claiming another item; its next start uses
`--queue submissions`.

```bash
set -euo pipefail
systemctl stop pareton-deploy.timer
flock /run/pareton-deploy.lock bash <<'SH'
set -euo pipefail
cd /opt/pareton
git pull --ff-only origin main
install -m 0755 ops/deploy.sh /usr/local/bin/pareton-deploy
install -m 0644 ops/systemd/pareton-round-worker.service /etc/systemd/system/
install -d /etc/systemd/system/pareton-worker.service.d
cat > /etc/systemd/system/pareton-worker.service.d/queue.conf <<'UNIT'
[Service]
ExecStart=
ExecStart=/opt/pareton/.venv/bin/python -m worker.main --queue submissions
UNIT
systemctl daemon-reload
touch .deploy-pending
if systemctl is-active --quiet pareton-worker; then
    systemctl kill --kill-who=main --signal=SIGTERM pareton-worker
fi
systemctl enable --now pareton-round-worker
SH
systemctl start pareton-deploy.timer
systemctl start pareton-deploy.service
```

The deploy lock here coordinates only deployment; it is separate from the builder
storage lock. Sending a signal with [`systemctl kill`](https://github.com/systemd/systemd/blob/main/man/systemctl.xml)
does not submit a service stop job. This lets the current build finish without
starting `TimeoutStopSec`'s stop countdown. Do not replace this drain step with
`systemctl restart pareton-worker` while a build is blocked. Keep the deploy timer
stopped if an installation command fails, resolve that failure, then resume it.

The old combined worker can finish a round it already owns during this
transition. If another campaign has a pending round, the new service can evaluate
it concurrently until the old process drains. To preserve strictly one GPU round
across all campaigns during migration, wait for the old worker's current round to
finish before enabling the new round service. This does not require waiting for
an in-flight submission build.

Subsequent deploys keep `.deploy-pending` and `.deploy-rounds-pending` independently.
An active round defers the round worker's restart; a busy build does not. The
existing worker's probe checks both queues to protect any legacy combined process.
A database probe failure defers restart. An uninstalled round unit is skipped,
with its pending restart retained until installation.

## Verify and monitor

```bash
systemctl is-active pareton-round-worker
systemctl show pareton-worker pareton-round-worker -p MainPID -p ExecStart
journalctl -u pareton-round-worker -n 80 --no-pager
journalctl -u pareton-worker -n 40 --no-pager
```

Expect `starting worker queue=rounds` and round processing in the new unit's log.
An old worker blocked on storage can remain active until the build finishes;
`signal 15 received; finishing current job before exit` confirms its drain request.
The unit's configured `ExecStart` changes before that old process exits, so also
check for `starting worker queue=submissions` after its replacement starts.

Heartbeats carry `queue=rounds` or `queue=submissions`. `queue_depth` counts pending
items for that role; round depth includes rounds in capacity backoff. Combined
mode reports the sum. Monitor each role separately so a healthy build worker
cannot conceal a missing round worker. A heartbeat proves the process is alive;
round progress still comes from round phase and heartbeat events.

Add `pareton-round-worker` to the live Vector configuration's
`sources.journald.include_units`, preserving its existing sink credentials and
other settings. The repo configuration includes it. The live config has known
credential drift, so do not overwrite it wholesale. Validate the edited file
using the existing Vector service environment, then restart Vector to ship the
new unit's logs. Journald verification above works before that logging change.
