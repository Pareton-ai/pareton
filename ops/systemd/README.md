# Worker systemd units

`pareton-worker.service` is the live (non-mock) worker unit for the production
VPS (`pareton-prod-01`, repo at `/opt/pareton`). It exists so the PAR-17 live
acceptance run is a copy-and-restart, not a hand edit on the host.

## PAR-17 deploy and run runbook

### 1. Preconditions

- PAR-13 (provider fallback) is merged to main.
- On the VPS: `cd /opt/pareton && git pull`.
- `/opt/pareton/.env` sets `PARETON_GPU_PROVIDER=lium` and
  `PARETON_GPU_PROVIDER_FALLBACKS=shadeform` (or relies on the defaults in
  `config.py`).
- Lium and Shadeform accounts are funded.

### 2. Install the live unit

1. Run `systemctl cat pareton-worker` on the VPS and diff against
   `ops/systemd/pareton-worker.service`. The repo unit is reconstructed from a
   partial observation. Keep any extra directives the live unit has.
2. Copy the reconciled unit to
   `/etc/systemd/system/pareton-worker.service`.
3. `systemctl daemon-reload && systemctl restart pareton-worker`.
4. Confirm with `journalctl -u pareton-worker -f` that the process starts
   without `--mock-bench`.

### 3. Drive one submission

- Submit one throwaway patch with `python -m miner.commit_patch` against the
  open campaign `c02a40b0-6eb3-4853-827e-22d4794b814e` (8x H200-SXM-141GB,
  GLM-5.2-FP8, trace `synthetic_v1`).
- Expect the build to take multiple hours. The bench then rents a real 8x H200
  pod (Lium first, Shadeform on capacity miss).

### 4. Verify

- `bench_reports` rows exist with a verdict per stage.
- `submission_events` shows the terminal state and is append-only.
- The evidence bundle is in S3 (`stage0/evidence/<submission_id>/`).
- `python -m gpu list` is empty after the run (pod and volume destroyed).
- Axiom monitors stay quiet during the run.

### 5. Rollback

Re-add `--mock-bench` to `ExecStart`, then `systemctl daemon-reload &&
systemctl restart pareton-worker`.

### 6. Safety

Run exactly one worker against the production database. Job claims use
`FOR UPDATE SKIP LOCKED`, but a second live worker still doubles GPU spend.
