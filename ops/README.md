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

### RTX PRO 6000 Qwen3.8 FP8 campaign (draft)

`ops/seed-sglang-qwen38-27b-pro6000.sh` prepares a campaign for one `RTXPRO6000`,
with `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, FP8
quantization, BF16 activation dtype, and SGLang commit
`4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc`. The model revision is a Hugging Face
pin, not an engine source commit.

**Draft dependency:** after [PR #187](https://github.com/Pareton-ai/pareton/pull/187)
merges, revise this campaign to sampler v5, `request_concurrency: 4`, the fixed
output budget, and `weighted_tier_completion_speedup` with explicit tier weights
and failure penalty. Until then, these fixtures retain main's 32-prompt v4
interval workload and `median_e2e_speedup` only for development validation.
That interval workload does not enforce sustained C32 or C4. Requalify the final
C4 contract and run its full shadow round before opening the campaign. Do not
launch this interim draft. Verify the backend and frontend activation dependencies
in #187 before launch.

The qualification fixture
`fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json` reuses both
image pins from the original Qwen campaign: the finished Pareton engine at
`ghcr.io/pareton-ai/pareton-baseline@sha256:43d5d33c2d3f61923d7ff96b8c69b77b8ddee28f749c10bb876ed538169fd431`.
This is `engine_image` in the original `image-pins.json`, not its bootstrap
`build_base_image`. It includes the offline miner installer and warmed build
cache. A GPU-specific rebuild is not required merely because TP or GPU count
changes. Matching the source commit alone does not establish compatible CUDA,
kernel binaries, drivers, or runtime behavior; qualify the existing digest on
RTX PRO 6000. If compatibility requires a rebuild, pin the new finished engine
in both fields and use it throughout qualification and seeding.

| vLLM configuration | Pinned SGLang serving argument |
| --- | --- |
| `--tensor-parallel-size 1` | `--tp 1` |
| `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder` | `--tool-call-parser qwen3_coder` (no separate auto-tool-choice switch) |
| `--reasoning-parser qwen3` | `--reasoning-parser qwen3` |
| `--max-model-len 262144` | `--context-length 262144` |
| `--max-num-seqs 32` | `--max-running-requests 32` |
| `--mm-encoder-tp-mode data` | `--mm-enable-dp-encoder` |
| `--speculative-config {"method":"mtp","num_speculative_tokens":3}` | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |

All these SGLang flags are stored in `bench.serve_args` and hashed into the
manifest. Engine capacity remains 32 even when benchmark concurrency becomes C4.
The speculative settings require runtime qualification on the target GPU.

Before any campaign launch, run `bench/qualify_longform.py` through its module
entry point on the Linux RTX PRO 6000 host, from the repository root with the
Pareton Python environment activated. First start the trusted baseline container
with the fixture's model revision and serving arguments, pinned weights mounted
at `/model`, and its API port published to `127.0.0.1:30000`. Use a Docker port
mapping, not host networking: the qualifier verifies the container's identity,
image digest and published endpoint. In the same shell, run:

```bash
PRO6000_ENGINE_REF=$(python -c 'import json; print(json.load(open("fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json"))["base_image_digest"])')
PRO6000_BASELINE_CONTAINER=pareton-pro6000-baseline  # actual running container name
PRO6000_QUALIFICATION_DIR=$(mktemp -d /var/tmp/pareton-pro6000-qualification-XXXXXX)
python -m bench.qualify_longform \
  --campaign-fields fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json \
  --base-url http://127.0.0.1:30000 \
  --container "$PRO6000_BASELINE_CONTAINER" \
  --engine-ref "$PRO6000_ENGINE_REF" \
  --output-dir "$PRO6000_QUALIFICATION_DIR" \
  --pool-size 64 --repetitions 2 --concurrency 4
```

The qualifier writes `sampling_rule.json` and evidence into the fresh output
directory. Its `--concurrency 4` controls pool qualification only; it does not
set the scored replay's concurrency. The command above works with the interim
v4 fixture; after the #187 migration, rerun it with the final v5 fixture and
follow #187's full shadow-round procedure. Earlier v4 receipts do not qualify v5.

Only after the C4 migration, fresh qualification, and full GPU round including
the trusted scorer succeed, configure `PARETON_DATABASE_URL` and the explicit
initial fee, then seed a new campaign using the qualified output:

```bash
bash -n ops/seed-sglang-qwen38-27b-pro6000.sh
ops/seed-sglang-qwen38-27b-pro6000.sh \
  "$PRO6000_ENGINE_REF" "$INITIAL_FEE_TAO" \
  "$PRO6000_QUALIFICATION_DIR/sampling_rule.json"
```

The helper preserves the supplied qualified rule unchanged. No TP4 memory
fractions or qualification receipts are copied. The scorer inherits this
campaign's flags with the harness's context headroom and logprob settings.
Validate any required memory adjustment in both the helper and fixture before
qualification. The 262144 context setting does not make LongWriter a full-context
or multimodal test. Image reuse does not reuse TP4 performance or correctness
qualification.

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
