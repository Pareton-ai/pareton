# Axiom observability (WS-A7)

Postgres remains the source of truth for operational state. Axiom is for search,
debugging, and alerting. No agents on GPU pods.

## Pre-flight

1. Dataset `pareton-prod` in the Pareton Axiom org.
2. Ingest-only API token as `PARETON_AXIOM_TOKEN` in `/opt/pareton/.env` on the droplet.
3. Cursor agents query logs via Axiom MCP (`https://mcp.axiom.co/mcp`, OAuth).

## Step 1 (code)

Structured lifecycle events emit through logger `pareton.lifecycle` as single-line JSON.
See `observability/events.py`.

## Step 2 (deploy, Xavier-approved)

1. Install Vector on the droplet: https://vector.dev/docs/setup/installation/
2. Copy `ops/vector/vector.toml` to `/etc/vector/vector.toml`.
3. Copy `ops/vector/vector.service` to `/etc/systemd/system/vector.service`.
4. `sudo mkdir -p /var/lib/vector` (disk buffer data_dir).
5. Validate before enabling: `vector validate /etc/vector/vector.toml`.
6. `sudo systemctl daemon-reload && sudo systemctl enable --now vector` and
   verify `systemctl restart vector`.

## Step 3: monitors and notifier (created 2026-08-05)

Email notifier `pareton-ops-email` → `bohdan@pareton.ai` (ID `LcdCHLqROCgv8il1a8`).

| Monitor | ID | Type |
| --- | --- | --- |
| worker-heartbeat-absent | b3ToVKWjImwQD4hpXi | Threshold, Below 1 / 15m |
| lifecycle-failures | wpbFNgAs8oVKXJtg9X | MatchEvent |
| job-failure-spike | WwNtNXLgbDmTCCfF1Q | Threshold, Above 5 / 60m |

Create a Discord or email notifier in the Axiom console if you need to change the destination.

### worker-heartbeat-absent

Threshold: zero `heartbeat` events in 15 minutes.

```apl
['pareton-prod'] | where event == "heartbeat" | summarize count()
```

- Type: Threshold
- Operator: Below
- Threshold: 1
- Range: 15 minutes
- Interval: 5 minutes

### lifecycle-failures

Match on `destroy_failed`, `pod_ttl_exceeded`, or `provider_balance_low`.

```apl
['pareton-prod'] | where event in ("destroy_failed", "pod_ttl_exceeded", "provider_balance_low")
```

- Type: MatchEvent
- Range: 5 minutes
- Interval: 5 minutes

### job-failure-spike

More than 5 `job_failed` events in 1 hour.

```apl
['pareton-prod'] | where event == "job_failed" | summarize count()
```

- Type: Threshold
- Operator: Above
- Threshold: 5
- Range: 60 minutes
- Interval: 15 minutes

## Synthetic test events (after monitors exist)

```bash
set -a && source /opt/pareton/.env && set +a
curl -X POST "https://api.axiom.co/v1/datasets/pareton-prod/ingest" \
  -H "Authorization: Bearer ${PARETON_AXIOM_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '[{"event":"destroy_failed","pod":"synthetic-test","provider":"lium","error":"monitor-test"}]'
```

## Step 5 verification (post-deploy)

1. Validate config before enabling: `vector validate /etc/vector/vector.toml`.
2. `sudo systemctl restart vector`.
3. Ingest a synthetic heartbeat (curl above with `"event":"heartbeat"`), then
   confirm via Axiom MCP that `['pareton-prod'] | where event == "heartbeat" | take 5`
   returns it with `event` as a top-level field.
4. `sudo systemctl restart pareton-worker`; the heartbeat thread emits at
   start, then every 5 minutes (including during long jobs). Confirm a real
   heartbeat is searchable within a minute.
5. Stop `pareton-worker`; `worker-heartbeat-absent` fires after its 15-minute
   window; restart resolves.
6. Synthetic `destroy_failed` triggers `lifecycle-failures`.
7. `bench_completed` events include `evidence_s3_url` and `evidence_sha256` on real runs.
