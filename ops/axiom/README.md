# Axiom observability

This page explains the logging setup and how to operate it. Read the first two
sections once. After that, jump to the task you need.

## What this is

Pareton services on the VPS write logs to journald. A fourth service,
Vector, reads journald and ships the logs to Axiom, a hosted log database.
Axiom watches the incoming events and emails you when something breaks.

```text
pareton-worker  --+
pareton-api      +--> journald --> Vector --> Axiom dataset "pareton-prod" --> monitors --> email
pareton-gpu-reap-+   (on VPS)    (on VPS)    (cloud, 30-day retention)
```

Postgres stays the source of truth for job state. Axiom is for search,
debugging, and alerts. Nothing is installed on GPU pods.

## The moving parts

| Piece                  | Where it lives                                                                           | What it does                                                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Lifecycle events       | `observability/events.py`                                                                | Code emits one JSON object per event (`heartbeat`, `job_failed`, ...) through the logger `pareton.lifecycle`.                                |
| journald               | VPS, automatic                                                                           | Systemd log store. `journalctl -u pareton-worker` reads it.                                                                                  |
| Vector                 | VPS, systemd unit `vector`                                                               | Ships journald lines to Axiom. Config: `/etc/vector/vector.toml` (copy in repo: `ops/vector/vector.toml`). Buffers to disk if Axiom is down. |
| Dataset `pareton-prod` | Axiom console                                                                            | Where logs land. Keeps 30 days.                                                                                                              |
| Ingest token           | `/opt/pareton/.env` as `PARETON_AXIOM_TOKEN`, and hardcoded in `/etc/vector/vector.toml` | Password that lets Vector write to the dataset.                                                                                              |
| Monitors               | Axiom console                                                                            | Alert rules. See "When an alert email arrives".                                                                                              |
| Axiom MCP              | Cursor settings                                                                          | Lets an agent query logs without SSH. Server `https://mcp.axiom.co/mcp`, browser OAuth sign-in.                                              |

## Search logs

No SSH needed. Open the Axiom console, go to Query, pick `pareton-prod`. Two
useful queries:

```apl
['pareton-prod'] | where _time > ago(1h) | sort by _time desc | limit 50
['pareton-prod'] | where event == "job_failed" | where _time > ago(24h)
```

Or ask a Cursor agent; the Axiom MCP server can run these queries for you.

## Check that shipping works

On the VPS (`ssh root@162.243.21.87`):

```bash
journalctl -u vector --since '10 minutes ago'
```

- No `ERROR` lines: shipping works.
- `Unauthorized`: the token in `/etc/vector/vector.toml` is wrong or revoked.
  See "Task: rotate the Axiom token".

Then confirm events arrive: run the first query in "Task: search the logs".

## Deploy code changes to the VPS

```bash
ssh root@162.243.21.87
cd /opt/pareton && git pull
.venv/bin/pip install -r requirements.txt
systemctl restart pareton-worker pareton-api
```

You do not need to restart `vector` for code deploys. Deploys need Xavier's
approval.

## Change the Vector config

1. Edit `ops/vector/vector.toml` in the repo and merge the change.
2. On the VPS, copy it to `/etc/vector/vector.toml`.
3. Run `vector validate /etc/vector/vector.toml`. Do not skip this.
4. Run `systemctl restart vector`.

The token line is the exception: keep the real token value in
`/etc/vector/vector.toml` on the VPS and keep `${PARETON_AXIOM_TOKEN}` in the
repo copy. The repo must not contain the token.

## Rotate the Axiom token

1. Axiom console: Settings, API tokens, create a new ingest token for
   `pareton-prod`.
2. On the VPS: put it in `/opt/pareton/.env` as `PARETON_AXIOM_TOKEN=xaat-...`
   and on the `token = "xaat-..."` line in `/etc/vector/vector.toml`.
3. Run `vector validate /etc/vector/vector.toml`, then `systemctl restart vector`.
4. Revoke the old token in the Axiom console.

Why the token is hardcoded: Vector did not expand `${PARETON_AXIOM_TOKEN}`
under systemd and got 401 Unauthorized from Axiom even with the variable set.
Hardcoding plus `chmod 600 /etc/vector/vector.toml` is the working setup.

## When an alert email arrives

Alerts go to bohdan@pareton.ai and xavier@pareton.ai.

| Monitor                   | Fires when                                                                     | First check                                                                                                       | Usual fix                                                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `worker-heartbeat-absent` | No `heartbeat` event for 15 min. The worker is dead or stuck. A green heartbeat does not mean the chain is being read: the heartbeat runs on its own thread and keeps beating through a 6-hour build. Use `chain-scan-stalled` for chain reads. | `systemctl status pareton-worker` on the VPS.                                                                     | `systemctl restart pareton-worker`. Read `journalctl -u pareton-worker -n 100` for the cause.                                                                  |
| `chain-scan-stalled`      | Fewer than 1 `chain_scanned` event in 10 min. We stopped reading the chain.     | Axiom: `['pareton-prod'] \| where event == "chain_scanned" \| sort by _time desc` for the last scan and its block. Then `journalctl -u pareton-worker -n 100` for `chain scan failed`. | A dead subtensor websocket or an RPC outage. Restart the scanning service. New submissions stay on chain, so they are ingested on the next good scan.          |
| `lifecycle-failures`      | A `destroy_failed`, `pod_ttl_exceeded`, or `provider_balance_low` event.       | Search Axiom for the event; it carries `pod`, `provider`, and `error`.                                            | `destroy_failed`: a GPU pod may still be running and billing; destroy it by hand in the provider console. `provider_balance_low`: top up the provider balance. |
| `job-failure-spike`       | More than 5 `job_failed` in 1 hour. Systemic breakage, not one bad submission. | Axiom: `['pareton-prod'] \| where event == "job_failed" \| summarize count() by stage` to find the failing stage. | Usually a bad deploy or a provider outage. Roll back or wait, then watch the monitor resolve.                                                                  |

### Is the worker alive, or is it working?

Two events answer two different questions. Read both before you restart anything.

| Event           | Cadence            | What it proves                                                                    |
| --------------- | ------------------ | --------------------------------------------------------------------------------- |
| `heartbeat`     | Every 5 min        | The worker process is alive. It carries `queue_depth`, the number of jobs waiting. |
| `chain_scanned` | Every 30 s         | We read the chain. It carries `block`, `commitments_seen`, and `ingested`.         |

`chain_scanned` fires on every successful scan, including a scan that finds
nothing. That is why its absence is an alert. `submission_ingested` cannot do
this job, because a quiet chain and a broken scanner both emit nothing.

A rising `queue_depth` is not an alert on its own. The worker builds one
submission at a time, and a build can take 6 hours, so a real queue is normal.

## First-time setup (already done 2026-08-05)

Kept for reference if you rebuild the VPS.

1. Axiom console: create dataset `pareton-prod` and an ingest API token.
2. VPS: install Vector from https://vector.dev/docs/setup/installation/
3. Copy `ops/vector/vector.toml` to `/etc/vector/vector.toml` and
   `ops/vector/vector.service` to `/etc/systemd/system/vector.service`.
4. Put the token in `/etc/vector/vector.toml` (see "Task: rotate the Axiom
   token") and run `chmod 600 /etc/vector/vector.toml`.
5. Run `sudo mkdir -p /var/lib/vector` (disk buffer).
6. Run `vector validate /etc/vector/vector.toml`, then
   `sudo systemctl daemon-reload && sudo systemctl enable --now vector`.
7. Run `systemctl restart pareton-worker` and confirm a `heartbeat` event shows
   in Axiom within a minute.
