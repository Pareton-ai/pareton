# ops/

Deployment artifacts for the production VPS. Files here are **verbatim copies of
what runs in production** — not templates. Change the copy here first, then
re-install it on the box, so the two never drift.

## Layout

| Path                              | Installed to                    | Notes                                                            |
| --------------------------------- | ------------------------------- | ---------------------------------------------------------------- |
| `systemd/pareton-api.service`     | `/etc/systemd/system/`          | Sandboxed dynamic user; uvicorn on `127.0.0.1:8000`              |
| `systemd/pareton-worker.service`  | `/etc/systemd/system/`          | Submission gates and builds (queue via drop-in)                  |
| `systemd/pareton-worker.service.d/queue.conf` | `/etc/systemd/system/pareton-worker.service.d/` | Clears `ExecStart`, re-sets it with `--queue submissions` |
| `systemd/pareton-worker.service.d/timeout.conf` | `/etc/systemd/system/pareton-worker.service.d/` | `TimeoutStopSec=4h`, overrides the unit's `8h` |
| `systemd/pareton-round-worker.service` | `/etc/systemd/system/`     | Round evaluation (`--queue rounds`)                              |
| `systemd/pareton-watcher.service` | `/etc/systemd/system/`          | Chain ingest, `python -m worker.watcher`                         |
| `systemd/pareton-weights.service` | `/etc/systemd/system/`          | Weight cadence, `python -m weights`. Holds the validator wallet. |
| `systemd/pareton-deploy.service`  | `/etc/systemd/system/`          | Oneshot, invoked by the timer; `OnFailure=` chains the alerter  |
| `systemd/pareton-deploy.timer`    | `/etc/systemd/system/`          | **Fires every 60s**                                              |
| `systemd/pareton-deploy-failed.service` | `/etc/systemd/system/`     | Started by `OnFailure`; sends the Discord deploy-failure alert  |
| `systemd/pareton-builder-cleanup.service` | `/etc/systemd/system/` | Docker image and BuildKit cleanup oneshot                         |
| `systemd/pareton-builder-cleanup.timer` | `/etc/systemd/system/` | Runs builder cleanup hourly                                      |
| `docker/daemon.json`                  | Merge into `/etc/docker/daemon.json` | Disables Docker's competing BuildKit GC without selecting an image store |
| `deploy.sh`                       | `/usr/local/bin/pareton-deploy` | The pull-deploy script itself                                    |
| `sync-config.py`                   | `/usr/local/lib/pareton-ops/`   | Stage-1 config check/apply; called by deploy.sh every tick       |
| `notify-deploy-failure.py`        | `/usr/local/lib/pareton-ops/`   | Deploy-failure notifier (4 modes); runs on `OnFailure`           |
| `ops_common.py`                    | `/usr/local/lib/pareton-ops/`   | Shared stdlib helpers for the two programs above                 |
| `gpu/pareton-gpu-reap.service`    | `/etc/systemd/system/`          | Oneshot GPU TTL reap                                             |
| `gpu/pareton-gpu-reap.timer`      | `/etc/systemd/system/`          | Fires every 10 min                                               |
| `vector/vector.service`           | `/etc/systemd/system/`          | Log shipping                                                     |
| `vector/vector.toml`              | `/etc/vector/`                  | Axiom sink, dataset `pareton-prod`, token via env ref           |
| `caddy/Caddyfile`                 | `/etc/caddy/`                   | TLS terminator, proxies to `127.0.0.1:8000`                      |

`deploy.sh` installs to `/usr/local/bin` rather than running from the repo
checkout so that a `git pull` cannot rewrite the script while it is executing.
Since stage 1 it **self-installs** that copy (plus the `/usr/local/lib/pareton-ops`
programs) from the just-pulled commit on every successful deploy, helpers first
and the main entry last. The one-time manual bootstrap that installs the first
self-updating copy is in [`runbook.md`](runbook.md).

## A merge to `main` is a production deploy

### RTX PRO 6000 Qwen3.8 FP8 campaign

Use a fresh **Ubuntu GPU VM with one RTX PRO 6000, NVIDIA drivers, Docker and
NVIDIA Container Toolkit preinstalled**. Run the VM commands as root in Bash;
port 30000 must be free. Allow disk space for the model cache plus a volume copy.

The [fixture](../fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json)
pins Qwen3.8-27B-FP8, TP1, 262144 context, v5 **16 requests at C4**
(eight each at 8k/16k, equal 0.5 weights, at most four baseline exclusions),
private patches and a **0.1 TAO** initial fee. Generation/scorer memory fractions are **0.80/0.60**,
following [#189](https://github.com/Pareton-ai/pareton/pull/189).

Use a new run directory and either fresh qualification or
[verified reuse of the original successful qualification](pro6000-qualification-reuse.md).
The old rule cannot be used directly. Both paths require a fresh 16-prompt shadow
round. Deploy this version of the worker before launch.

Every long stage runs under `nohup`. **Ctrl-C on `tail` stops only the viewer.**
Proceed only when that stage's `.exit-code` file contains `0`; missing/nonzero
status means inspect its log, not relaunch it. Never enable `set -e` in the
interactive shell. Keep the run directory and evidence.

#### 1. Checkout and install dependencies

```bash
set +e
umask 077
apt-get update && apt-get install -y python3 python3-venv python3-pip git curl jq
mkdir -p /workspace
cd /workspace
git clone --branch arpan/pro6000-fp8-campaign https://github.com/Pareton-ai/pareton.git
cd pareton
nvidia-smi --query-gpu=name,memory.total --format=csv
docker info >/dev/null

export PRO6000_RUN_DIR=$(mktemp -d /var/tmp/pareton-pro6000-XXXXXX)
export PRO6000_SETUP_DIR="$PRO6000_RUN_DIR/setup"
mkdir "$PRO6000_SETUP_DIR"
export PRO6000_FIELDS=fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json
export PRO6000_ENGINE_REF=$(jq -er '.base_image_digest' "$PRO6000_FIELDS")
export PRO6000_BASELINE_CONTAINER="$(basename "$PRO6000_RUN_DIR")-qualification"
export PRO6000_QUAL_NET="$(basename "$PRO6000_RUN_DIR")-network"
export PARETON_BENCH_HEALTH_TIMEOUT_S=3600
declare -p PRO6000_RUN_DIR PRO6000_SETUP_DIR PRO6000_FIELDS PRO6000_ENGINE_REF \
  PRO6000_BASELINE_CONTAINER PRO6000_QUAL_NET PARETON_BENCH_HEALTH_TIMEOUT_S \
  > "$PRO6000_RUN_DIR/env.sh"
printf 'Keep this run path: %s\n' "$PRO6000_RUN_DIR"
nohup bash ops/setup-pro6000.sh >> "$PRO6000_SETUP_DIR/setup.log" 2>&1 < /dev/null &
```

Setup creates `.venv`, installs `requirements.txt` (including pinned `tokenizers`),
runs `pip check` and checks host imports. Authenticate with `docker login ghcr.io`
if the image requires it. Watch setup, then check its result:

```bash
tail -f "$PRO6000_SETUP_DIR/setup.log" || true
cat "$PRO6000_SETUP_DIR/setup.exit-code"
```

#### 2. Qualify the baseline

After setup returns `0`, activate `.venv`. This job stages and verifies weights,
starts the baseline, runs `bench.qualify_longform` with pool 32, two repetitions,
concurrency 4 and timeout 600, then removes its container/network/volume on success.

```bash
source .venv/bin/activate
if docker network create "$PRO6000_QUAL_NET"; then
  nohup bash ops/qualify-pro6000.sh >> "$PRO6000_RUN_DIR/step2.log" 2>&1 < /dev/null &
fi
```

```bash
tail -f "$PRO6000_RUN_DIR/step2.log" || true
cat "$PRO6000_RUN_DIR/step2.exit-code"
```

After reconnecting: `cd /workspace/pareton`, source the saved run's `env.sh`, then
`source .venv/bin/activate`. Inspect the existing job; do not repeat setup.

#### 3. Run the C4 shadow benchmark

After step 2 returns `0`, generate the trace/request and benchmark the unchanged
baseline as candidate. The job preserves the normal v5 C4 workload and scorer.

```bash
nohup bash ops/shadow-pro6000.sh >> "$PRO6000_RUN_DIR/step3.log" 2>&1 < /dev/null &
```

```bash
tail -f "$PRO6000_RUN_DIR/step3.log" || true
cat "$PRO6000_RUN_DIR/step3.exit-code"
jq '{verdict, entries, error}' "$PRO6000_RUN_DIR/shadow/bench_report.json"
```

Require exit `0`, a scored candidate, passing correctness, both surviving
tiers and passing drift/repeatability gates. Review the retained evidence; CPU
checks and #189's diagnostic do not replace this GPU run. Before seeding, also
confirm offline miner-build/native-probe checks for the pinned engine and the
[deployment/privacy requirements](../docs/patch-visibility.md#rollout), including
#186/#187 backend rollout and compatible frontend support (#88/#89, plus
[pareton-frontend#91](https://github.com/Pareton-ai/pareton-frontend/pull/91) for the
8k/16k tier subset). Merge and deploy #91 before launch, then verify the campaign's
tier weights, entry score breakdowns and C4 scheduling labels in the dashboard.

#### 4. Seed once on the configured controller

Transfer `qualification/sampling_rule.json` and its evidence to protected storage
on the controller. Use the reviewed checkout and its prepared `.venv`/`.env`,
following the [campaign launch skill](../docs/campaign_launch_skill.md).
**This creates an open campaign.** Set the transferred file path below.

```bash
set +e
cd /opt/pareton
source .venv/bin/activate
set -a; source .env; set +a
umask 077
export PRO6000_QUALIFIED_RULE=/absolute/path/to/qualification/sampling_rule.json
export PRO6000_SEED_DIR=$(mktemp -d /var/tmp/pareton-pro6000-seed-XXXXXX)
nohup bash ops/seed-pro6000-job.sh >> "$PRO6000_SEED_DIR/seed.log" 2>&1 < /dev/null &
printf 'Keep this seed path: %s\n' "$PRO6000_SEED_DIR"
```

```bash
tail -f "$PRO6000_SEED_DIR/seed.log" || true
cat "$PRO6000_SEED_DIR/seed.exit-code"
```

After exit `0`, take the UUID from `seed.log` and verify the API result:

```bash
read -r -p 'Campaign UUID: ' PRO6000_CAMPAIGN_ID &&
curl -fsS "https://api.pareton.ai/v1/campaigns/$PRO6000_CAMPAIGN_ID" | jq .
```

Compare the returned pins/settings with the fixture: open status, private patches,
v5 C4 with 16 prompts in 8k/16k, 0.5/0.5 weights, four-exclusion limit, 0.1 TAO
fee, image/model revisions, memory fractions and emissions. Verify
private patch access through the linked privacy runbook. If seed completion is
uncertain, check the log and database/API before any retry; a new job directory
does not prevent duplicate campaigns.

### Correctness scorer memory

Pin scorer overrides in the campaign's `bench.correctness.serve_args`:

```json
["--mem-fraction-static", "0.4"]
```

The Qwen seed helper supplies these through repeatable
`--bench-correctness-serve-args` options. The round request carries them to the
remote harness, which appends them only to the trusted scorer's serving arguments.
The scorer inherits the campaign's TP and GPU allocation. The original NVFP4 Qwen stages use TP4
and four GPUs; timed baseline, candidate and drift stages retain memory fraction
`0.85`. No scorer TP environment setting or additional GPUs are required.

Deploy the harness before seeding a campaign with these arguments. Existing
campaigns need a pinned manifest update. Verify that the scorer's Docker launch
uses `--mem-fraction-static 0.4` and the campaign's TP and GPU count, and require
completed baseline and candidate correctness reports. To restore the baseline
memory setting, remove the correctness memory override and update the manifest.

### Temperature-extremes logprob check

Run this on the campaign's GPU host, with the exact fixture, before opening a new
campaign. The trusted baseline generates one sampled campaign prompt ten times at
each end of the sampling rule's `temperature_range` (0.1 and 1.01 in the v5
LongWriter rule), with natural EOS and seeds 0 to 9. The campaign's scorer then
grades every output separately against the campaign's absolute bars: mean
logprob, the token logprob at `min_token_quantile` (production applies
`min_token_logprob` there), and coverage. `results.json` also records each
output's raw minimum token logprob and whether it is below `min_token_logprob`.

```bash
export PARETON_BENCH_HEALTH_TIMEOUT_S=3600
PYTHONPATH=. nohup python -u ops/temperature_logprob_check.py \
  --campaign-fields fixtures/campaigns/sglang_kimi_k3_b300/campaign-fields.json \
  --output-dir "$(mktemp -u /var/tmp/pareton-temperature-check-XXXXXX)" \
  > temperature-check.log 2>&1 < /dev/null &
```

Exit `0` means all twenty outputs passed; `1` means at least one failed a bar
(listed in `summary.json` as `failed_samples`); `2` or `3` means the check could
not complete. A failure at 1.01 with a pass at 0.1 suggests the thresholds are
too strict for the campaign's own sampling range: revisit them with the
campaign owner before launch rather than narrowing the temperature range.
`--request-index` selects another prompt from the same fixed-seed trace.

### Deployment lifecycle

`pareton-deploy.timer` polls `origin/main` every 60 seconds. There is no
separate promote step. Every tick that holds the deploy lock first runs the
stage-1 config sync (`sync-config.py deploy-hook`): managed-file drift
converges to the Git content automatically, while unknown files, masks, or a
broken alert credential fail the deploy and page the team. Any merge then
restarts `pareton-api`, `pareton-watcher`, and `pareton-weights` within a
minute. Each execution worker has its own pending
restart. The round worker waits only for running rounds; the existing worker checks
both queues to protect legacy combined processes. A busy build does not defer an
idle round worker's update. A deploy that fails — including a failing worker
busy-probe — triggers `OnFailure=pareton-deploy-failed.service`, which sends a
rate-limited alert straight to the team Discord channel (independent of
Vector/Axiom).

**During a maintenance window, stop this timer first.** Stopping any other unit
while the timer is live means the timer may restart it underneath you. Stopping
the timer does not stop a deploy that is already running — wait for it, then
work. After a manual hotfix to a managed file, the fix must be **merged to
`main`** (a pushed branch or open PR is not enough) before the timer is
resumed, or the next tick reverts the live change as drift.

### Worker heartbeat alerts

After both services are shipping logs, filter the existing Axiom
`worker-heartbeat-absent` monitor to `pareton-worker.service`, then clone it as
`round-worker-heartbeat-absent` with the second query below. Keep the current
notifiers and evaluation frequency, use **Below 1 over 15 minutes**, and enable
**Alert on no data** for each. The existing `_SYSTEMD_UNIT` field identifies the
process, so the heartbeat payload does not need changing.

```apl
['pareton-prod']
| where event == "heartbeat" and _SYSTEMD_UNIT == "pareton-worker.service"
| summarize count()
```

```apl
['pareton-prod']
| where event == "heartbeat" and _SYSTEMD_UNIT == "pareton-round-worker.service"
| summarize count()
```

Use two fixed filters: a grouped query can lose a missing service's group while
the other continues reporting. These alerts detect absent processes or telemetry;
progress stalls still require round phase/heartbeat monitoring. Adding weights
to the allowlist resumes its telemetry on the next scheduled event, without
forcing a weight submission or recovering previously discarded logs.

**During a maintenance window, stop this timer first.** Stopping any other unit
while the timer is live means the timer may restart it underneath you.

## Known drift — needs a decision

Resolved on 2026-09-10 by the stage-1 capture (files here now mirror the live
box, verified against the read-only audit output of that day; made byte-exact
on 2026-09-11 per owner review — provenance lives in this README, never as
added comments inside the files, so the first sync sees zero diff on the
worker and owes it no restart):

1. ~~`pareton-worker.service` differs from the live unit~~ — the committed file
   is the byte-exact live version; the queue split lives in the
   `queue.conf` drop-in (byte-exact as well).
2. ~~`TimeoutStopSec` drop-in drift~~ — `timeout.conf` (4h) is committed next
   to the unit; the effective value stays 4h. Revisit the value itself later.
4. ~~`vector.toml` inline token~~ — the repo keeps the `${PARETON_AXIOM_TOKEN}`
   form; the owner verified the env token via a direct ingest test, so the
   live file migrates to the env reference at the stage-1 bootstrap and new
   events arriving in `pareton-prod` are the acceptance evidence
   (`vector validate` passing proves nothing about token validity).

Still open, each because it changes production behavior:

3. **`aws/pareton-api-iam-policy.json` overstates the live IAM policy.** It
   grants `s3:ListBucket` and `s3:DeleteObject`; the live `pareton-api` user
   has neither. Only `PutObject`/`GetObject` on `stage0/*` actually work.
   Private patch uploads, validator reads, and public copies require only
   `PutObject`/`GetObject`. The file should not be treated as an accurate record
   of live permissions.

5. **The box needs swap, and nothing here says so.** Hermetic builds compile
   vLLM's CUDA kernels; `cicc` peaks at 6–12 GB per job and will OOM a 16 GB
   box. The only record of this is a comment in `a2b-build.sh` telling you to
   `fallocate` a 64 G swapfile by hand. A host rebuilt from this directory
   silently gets no swap, and the first submission dies with `Killed` /
   `exit status 137` — which surfaces as `hermetic_build_failed` and rejects
   the miner's patch for an infrastructure fault. `pareton-prod-02` now has a
   64 G swapfile with an `/etc/fstab` entry; provisioning should create one.

## Reinstalling a unit

```sh
scp ops/systemd/pareton-deploy.timer root@<host>:/etc/systemd/system/
ssh root@<host> systemctl daemon-reload
ssh root@<host> systemctl restart pareton-deploy.timer
```

## Builder disk cleanup

The persistent build host keeps local retention tags for the build-base and
baseline engine images of every draft or open campaign. Published candidate
tags are removed after their digest-pinned reference is stored in Postgres.
The hourly fallback sweep removes leftover candidate tags and prunes ordinary
BuildKit records after Docker storage crosses 75% usage. It targets the same
explicit Buildx builder as miner builds and does not run daemon-wide image or
system prune commands.
BuildKit `exec.cachemount` records are excluded because they hold the warmed
baseline ccache used by later miner builds.

Docker Engine's background BuildKit GC is disabled on the dedicated builder
host. Pareton's filtered cleanup is the only BuildKit GC authority, so Docker
cannot independently reclaim `exec.cachemount`. Both the worker and cleanup
units fail their startup check unless `/etc/docker/daemon.json` has
`builder.gc.enabled=false`.

Baseline build and serving images for every draft or open campaign have local
retention tags. Candidate and leader images are durable in GHCR by digest.
Rounds read those digest-pinned references from Postgres and pull them on the
GPU host, so removing a builder-host candidate tag never causes a rebuild.
This cleanup never deletes registry artifacts.

The cleanup fails without deleting anything when Postgres is unavailable, and
skips a run when a build holds the shared storage lock. Preview it before
installing the timer:

```sh
cd /opt/pareton
set -a
. ./.env
set +a
.venv/bin/python -m builder.cleanup --dry-run --force
```

Install the Docker policy and both units during a maintenance window. The
committed file is a merge fragment, not a replacement daemon configuration.
It deliberately omits `features.containerd-snapshotter`. Merge it into the
host's current configuration so Docker keeps its active classic or containerd
image store and every unrelated daemon setting. Record the active storage
driver before the restart and require the same value afterward.

```sh
systemctl stop pareton-deploy.timer
systemctl stop pareton-worker
command -v jq
test -f /etc/docker/daemon.json
image_store_before=$(docker info --format '{{json .DriverStatus}}')
cp -a /etc/docker/daemon.json /etc/docker/daemon.json.pre-pareton-gc
daemon_merged=$(mktemp)
jq -s '.[0] * .[1]' \
  /etc/docker/daemon.json ops/docker/daemon.json > "$daemon_merged"
dockerd --validate --config-file="$daemon_merged"
install -m 0644 "$daemon_merged" /etc/docker/daemon.json
rm -f "$daemon_merged"
systemctl restart docker
image_store_after=$(docker info --format '{{json .DriverStatus}}')
test "$image_store_after" = "$image_store_before"
.venv/bin/python -m builder.gc_config
cp ops/systemd/pareton-worker.service /etc/systemd/system/
cp ops/systemd/pareton-builder-cleanup.service /etc/systemd/system/
cp ops/systemd/pareton-builder-cleanup.timer /etc/systemd/system/
systemctl daemon-reload
systemctl start pareton-worker
systemctl enable --now pareton-builder-cleanup.timer
systemctl start pareton-builder-cleanup.service
systemctl start pareton-deploy.timer
journalctl -u pareton-builder-cleanup.service -n 100 --no-pager
```

## Use a fixed GPU machine

Set the existing provider mode in the validator's `.env`, then restart
`pareton-round-worker` after its current round finishes:

```dotenv
PARETON_GPU_PROVIDERS=static_ssh
PARETON_GPU_STATIC_SSH=user@host:port
PARETON_GPU_SSH_KEY_PATH=/path/to/key
```

Use a dedicated Linux/NVIDIA node with SSH and passwordless sudo for non-root
users. One validator owns the node; its workers must share `PARETON_GPU_STATE_DIR`.
Bootstrap still recreates the remote Python environment each round.

### Campaigns and rental ownership

- Every campaign uses this host. Before switching to a 5090, close incompatible
  campaigns and finish their pending/running rounds on compatible hardware.
  **Closing a campaign does not drain its queue.** Provisioning rejects wrong or
  mixed GPU models and insufficient counts; oversized nodes log unused GPUs.
- Static mode never releases the rental, even at campaign closure. Release it
  manually. Prefer a 4-GPU node for a 4-GPU campaign.
- Use operator-managed rental and volume names. **Retire TTL management before
  reusing a `pt-<timestamp>-<ttl>h-*` rental:** the existing reaper still scans cloud
  providers for those names, even when `PARETON_GPU_PROVIDERS=static_ssh`.

### Housekeeping and recovery

Each round removes abandoned Pareton bench containers, anonymous volumes and
networks before image pulls, then removes candidate images and temporary outputs
after collecting results. Output deletion is limited to the current run and
requires every attempted download to succeed. Failed transfers or lost SSH
sessions leave reports on the GPU node for manual recovery; later rounds and the
periodic reaper preserve them. Retrieve these reports before removing them.
Only Pareton bench names and candidate references in
`/opt/pareton/.static-host-images.json` are reclaimed. Keep this ledger across
restarts; failed image deletions remain tracked for retry. Current baseline
images, model downloads and baseline compile caches are preserved. Inspect older,
untracked images separately. No separate daily image-pruning job is needed.

Local locking prevents overlapping workers. Remote locking protects active
harnesses from cleanup and prevents bootstrap over a surviving harness. Busy
hosts defer the round using `PARETON_PROVISION_RETRY_S`, preserving its cohort and
seed, including when the reaper takes the lock during bootstrap. SSH or
lock-system failures still fail the round. A remote harness has the
configured benchmark timeout plus a 30-second kill grace period; a new harness
waits up to 120 seconds for maintenance, then defers if the lock remains held.
The timeout and harness run in a separate session with output redirected to
`supervisor.log` in the remote run directory, so SSH hangup or closed pipes do
not remove the deadline. The SSH client allows 60 additional seconds for remote
termination and status delivery. The harness directly holds the lock, which is
released when it exits or is killed. Ordinary reaper contention during final
cleanup is skipped without a failure alert or changes to the lock owner's files.

The existing **`pareton-gpu-reap.timer` runs on the validator VPS every 10 minutes**
and SSHes to the configured node. Enable it and deploy updated code on both hosts
(normal bootstrap uploads GPU-side code). It skips active harnesses, removes
orphan Pareton containers/networks, then checks NVIDIA compute PIDs, allowing two
seconds for process exit. It leaves images and reports untouched while a worker
may be downloading results, and works even with a corrupt image ledger.

Remaining GPU processes, failed inspection or SSH failure emit
`static_host_cleanup_failed` and fail the reaper run. The timer requires a running
validator and reachable node. It cannot reset the GPU driver or kill unrelated
processes, and its check does not require zero reported VRAM. Validate crash
recovery on the actual node before relying on unattended cleanup.

### Alerts

There is no cloud fallback with only `static_ssh` configured. Failed rounds emit
`round_voided`. Hard cleanup failures (GPU processes still running, inspection
or SSH failure) emit `static_host_cleanup_failed` without discarding a valid
score. A candidate image still referenced by a container stays tracked and emits
`static_host_cleanup_deferred`. That retry is not a page.

The Axiom monitor uses the operations notifier, **Above 0 over 30 minutes**,
evaluated every 5 minutes, and **Alert on no data** off. Thirty minutes covers
the 10-minute reaper, so one stuck GPU stays one open alert. The `error !has`
clause ignores deferred image retries that older builds logged as
`static_host_cleanup_failed`. Keep the separate worker heartbeat alerts above.

```apl
['pareton-prod']
| where (event == "round_voided" and void_reason in
    ("pod_provision_failed", "pod_failed", "round_timeout", "heartbeat_stale"))
    or (event == "static_host_cleanup_failed"
        and error !has "candidate image cleanup needs retry")
| summarize failures=count()
```

Activate the monitor/notifier during rollout; code deployment does not create it.
Investigate failures in `pareton-round-worker` and `pareton-gpu-reap` journals.

### Builder disk headroom

`pareton-builder-cleanup.service` runs from an hourly timer. Each run emits
`builder_cleanup` with `usage_before_percent` and `usage_after_percent`. A run
that finishes at or above the hard watermark (default 90 percent) sets
`above_hard_watermark` to true and exits 2.

The Axiom monitor uses the operations notifier, **Above 0 over 2 hours**,
evaluated every 15 minutes, and **Alert on no data** off. Two hours covers the
hourly timer, so one full disk stays one open alert.

```apl
['pareton-prod']
| where event == "builder_cleanup" and above_hard_watermark == true
| summarize failures=count()
```

Activate the monitor/notifier during rollout; code deployment does not create it.
