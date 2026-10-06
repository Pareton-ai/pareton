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

This new campaign uses the merged #186 privacy policy and #187 v5 contract:

- `patch_visibility: {"mode": "private"}`: no public reveal, including after
  evaluation or campaign closure. Private mode has no reveal delay. This policy
  is operational and deliberately outside `manifest_hash`; never enable public
  disclosure for this campaign. Keep patch storage and candidate registries private.
- `algo_version: 5`, **32 requests at C4**, eight requests per 2k/4k/8k/16k tier
  before baseline exclusions. Each tier drains before the next starts; FIFO slot
  refill caps admitted requests at four. Exclusions/final drain can lower occupancy.
  Engine `--max-running-requests 32` remains capacity, not benchmark concurrency.
- Natural EOS, 5120-token ceiling, 3000-token baseline minimum, thinking disabled,
  and an absolute 600-second request timeout. Both baseline runs establish the
  union of exclusions (at most eight, with no empty tier). Candidates must emit
  at least 90% of the measured baseline's reported tokens per eligible request.
- `weighted_tier_completion_speedup`, equal 0.25 tier weights, failure penalty 0.1.
  Tier completion includes client queueing; any scoreable failure caps speed
  credit at zero before the failure deduction. There is no fixed-output mode.
- Initial fee **0.1 TAO**, supplied by the seed helper; emissions start at 0.20
  and decline to zero over 201600 leader-held blocks.

The fixture pins one `RTXPRO6000`,
`Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, BF16 activation
dtype, FP8 quantization, 262144 context, and SGLang
`4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc`. Its serving arguments retain TP1,
`qwen3_coder` tool parsing, `qwen3` reasoning parsing, data-parallel multimodal
encoding, and EAGLE with three steps, top-k one and four draft tokens.

The operator reports a successful run of the [#189 diagnostic](https://github.com/Pareton-ai/pareton/pull/189)
with generation memory fraction `0.80` and scorer fraction `0.60`. These are now
pinned as `--mem-fraction-static` in `bench.serve_args` and
`bench.correctness.serve_args`, respectively. The diagnostic does not replace
fresh qualification and a complete shadow round for this v5 C4 contract.

Both image fields reuse the original campaign's finished Pareton engine:
`ghcr.io/pareton-ai/pareton-baseline@sha256:43d5d33c2d3f61923d7ff96b8c69b77b8ddee28f749c10bb876ed538169fd431`.
This is `engine_image`, not the bootstrap `build_base_image`. No target-GPU build
is required solely for TP1, but runtime compatibility and the full workload must
be qualified on RTX PRO 6000. A replacement engine requires updating both fixture
image fields and fresh qualification. TP4, v4, C32 or other-model receipts do not
qualify this contract. The context limit does not make this a full-context or
multimodal benchmark.

#### 1. Deployment and host prerequisites

Before launching the campaign, complete target-GPU qualification and a full
shadow round. These are operator instructions, not evidence that deployment or launch has
occurred. Use the updated [campaign launch skill](../docs/campaign_launch_skill.md).
Verify the deployed API/workers contain #186, #187 and this PR, the fee-history
and `20261001_campaign_patch_visibility.sql` migrations are applied, and compatible
frontend support from frontend PRs #88 and #89 is deployed. Follow the existing
[privacy migration/runbook](../docs/patch-visibility.md#rollout) and release
coordinator; do not replace a live worker checkout with this branch. Confirm the
private object-store/registry access controls and private patch API behavior.

Host Python dependencies must be installed **before step 1 creates a GPU run**.
The engine container does not supply the host qualifier's packages. Use Python
3.10+ with `venv` support (the reported host uses Python 3.12). On Ubuntu/Debian,
if needed, install the host tools with
`sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip git curl jq`.
Docker, NVIDIA drivers and NVIDIA Container Toolkit must already work on this
host; these Python setup commands do not install or modify them.

From the root of the latest PR checkout, prepare its `.venv` as one background
job. `requirements.txt` includes the API requirements and pins
`tokenizers==0.22.2` and `jinja2==3.1.6`; do not install an arbitrary tokenizer
version or rely on packages inside the SGLang image. No HF token is required for
this public model; an existing `HF_TOKEN` can improve Hub rate limits.

```bash
set +e
umask 077
export PRO6000_SETUP_DIR=$(mktemp -d /var/tmp/pareton-pro6000-setup-XXXXXX)
nohup bash ops/setup-pro6000.sh >> "$PRO6000_SETUP_DIR/setup.log" 2>&1 < /dev/null &
printf 'Launched PID %s; setup evidence: %s\n' "$!" "$PRO6000_SETUP_DIR"
```

Watch separately; Ctrl-C stops only the viewer:

```bash
tail -n 60 -f "$PRO6000_SETUP_DIR/setup.log" || true
```

Do not proceed until `setup.exit-code` exists and contains `0`. If setup fails,
inspect `setup.log`; do not create a baseline container. After successful setup,
activate the installed environment in the interactive shell:

```bash
if [[ -f "$PRO6000_SETUP_DIR/setup.exit-code" ]] && \
   [[ "$(cat "$PRO6000_SETUP_DIR/setup.exit-code")" == 0 ]]; then
  source .venv/bin/activate
else
  printf 'Setup has not succeeded; inspect %s/setup.log before continuing.\n' "$PRO6000_SETUP_DIR"
fi
```

The controllers repeat `ops.pro6000_preflight` with their actual Python
interpreter. This checks required host imports, including the lazy tokenizer,
dataset and worker imports, and the pinned tokenizer/template versions before
weight staging or other expensive work. It does not establish GPU qualification.

On an idle, dedicated Linux host with one RTX PRO 6000, working NVIDIA drivers,
NVIDIA Container Toolkit, Docker and repo Python dependencies, use a separate
checkout of the reviewed PR commit. Authenticate to GHCR through the existing
credential mechanism if needed. Run the following in one Bash shell with the
Pareton virtualenv active; `curl` and `jq` are also required:

```bash
# Keep error handling inside job scripts, not the interactive SSH shell.
set +e
umask 077
export PRO6000_FIELDS=fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json
export PRO6000_RUN_DIR=$(mktemp -d /var/tmp/pareton-pro6000-XXXXXX)
export PRO6000_ENGINE_REF=$(python -c 'import json,os; print(json.load(open(os.environ["PRO6000_FIELDS"]))["base_image_digest"])')
export PRO6000_BASELINE_CONTAINER="$(basename "$PRO6000_RUN_DIR")-qualification"
export PRO6000_QUAL_NET="$(basename "$PRO6000_RUN_DIR")-network"
export PARETON_BENCH_HEALTH_TIMEOUT_S=3600
nvidia-smi --query-gpu=name,memory.total --format=csv
# Verify this is the intended idle RTX PRO 6000 before proceeding.
docker pull "$PRO6000_ENGINE_REF"
docker network create "$PRO6000_QUAL_NET"
```

Retain `PRO6000_RUN_DIR` and its evidence. On interruption, stop/remove only the
named qualification container, recorded model volume and network from this run;
remove containers before their volume. Do not prune shared Docker or build caches.

#### 2. Start the exact baseline and qualify a fresh source pool

This stages the immutable weights locally, copies them through Docker's API into
a named volume, verifies every copied file's SHA-256, then mounts the volume
read-only. This avoids client-path bind mounts on containerized GPU hosts whose
Docker daemon has a different filesystem. It needs disk for one additional model
copy; copying and hashing can take several minutes. The implementation follows
#189's volume workaround without its diagnostic workload changes.

Use a checkout containing `ops/pro6000_model_volume.py`. The qualifier still
requires the Docker daemon's published port to be reachable at
`http://127.0.0.1:30000` from this shell; a remote daemon without local forwarding
is unsupported. Ensure port 30000 is free. The model-volume fix addresses storage
visibility, not network namespace reachability.

Use an ordinary bridge network for this loopback-published qualification endpoint:
Docker's internal-network port-publishing limitation prevents this procedure from
reaching the server (see [moby#36174](https://github.com/moby/moby/issues/36174)).
The port remains bound to `127.0.0.1`. Staged weights and `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1` avoid model downloads; they do not block outbound traffic.
This qualification setup does not enforce network egress isolation. The shadow
round harness retains its internal network and direct container-IP connection.

The whole step runs as **one background job**: startup, health checks,
qualification and success cleanup. Its Python children inherit `nohup`'s hangup
handling. There is no foreground `wait` or coordinator to interrupt. Pull the
latest PR branch first; `ops/qualify-pro6000.sh` and `ops/pro6000_model_volume.py`
must both exist, and the virtualenv must be active.

Launch once per fresh run directory:

```bash
# Undo errexit in the interactive shell if an older runbook enabled it.
set +e
nohup bash ops/qualify-pro6000.sh >> "$PRO6000_RUN_DIR/step2.log" 2>&1 < /dev/null &
printf 'Launched PID %s; log: %s/step2.log\n' "$!" "$PRO6000_RUN_DIR"
```

Watch separately. **Ctrl-C here stops only the viewer**, and `|| true` also
prevents an interactive shell with errexit enabled from closing:

```bash
tail -n 60 -f "$PRO6000_RUN_DIR/step2.log" || true
```

After reconnecting, restore the virtualenv and environment using the same run
directory. Read `step2.pid` and `step2.log`; do not launch another job. The script
claims `step2.lock` to reject duplicate runs and records its final status in
`step2.exit-code`. Missing status means unfinished or interrupted unexpectedly,
not success. Proceed to step 3 only after exit code `0` **and** review of the
qualification artifacts. Nonzero status stops dependent work and success cleanup;
inspect the logs before retrying. A failed run needs a fresh directory, after
confirming its processes/containers are no longer using the GPU.

```bash
if [[ -f "$PRO6000_RUN_DIR/step2.exit-code" ]]; then
  printf 'Step 2 exit code: '
  cat "$PRO6000_RUN_DIR/step2.exit-code"
else
  printf 'No completion status yet; inspect %s/step2.log and step2.pid.\n' "$PRO6000_RUN_DIR"
fi
```

`qualification/sampling_rule.json` is the qualified output to use below. The
qualifier's `--concurrency 4` controls source screening; the fixture's separate
`request_concurrency: 4` controls scored replay. Do not use an old pool or modify
its qualified settings. Failed qualification must be investigated and rerun into
a fresh directory, without relaxing correctness or privacy implicitly.

#### Recover qualification after a host dependency failure

For the reported missing-`tokenizers` failure, the baseline is already running
and no qualification evidence was written. Keep that container and its volume.
Install dependencies using the setup procedure above, then activate `.venv`.
Do not rerun step 1: retain `PRO6000_BASELINE_CONTAINER`, `PRO6000_QUAL_NET`,
`PRO6000_ENGINE_REF` and `PRO6000_FIELDS` from the failed run. Confirm that its
controller has exited before continuing.

Use a fresh output directory while retaining the old evidence. Copy only the
verified volume receipt; the retry checks that the running container actually
mounts that recorded volume read-only and verifies its image and loopback binding.
It then reuses the baseline, requalifies from source and performs normal cleanup
only after success. It does not loosen any v5/C4 settings.

```bash
export PRO6000_PREVIOUS_RUN_DIR="$PRO6000_RUN_DIR"
export PRO6000_RUN_DIR=$(mktemp -d /var/tmp/pareton-pro6000-retry-XXXXXX)
if cp "$PRO6000_PREVIOUS_RUN_DIR/model_volume.json" "$PRO6000_RUN_DIR/model_volume.json"; then
  nohup bash ops/qualify-pro6000.sh --reuse-baseline >> "$PRO6000_RUN_DIR/step2.log" 2>&1 < /dev/null &
  printf 'Launched recovery PID %s; log: %s/step2.log\n' "$!" "$PRO6000_RUN_DIR"
fi
```

```bash
tail -n 60 -f "$PRO6000_RUN_DIR/step2.log" || true
```

Require `step2.exit-code` to contain `0` and review the qualification artifacts,
then use this **new** `PRO6000_RUN_DIR` for step 3. Missing/nonzero completion
status is not permission to proceed. Do not manually write a success marker.

#### 3. Generate the C4 trace and a worker-derived shadow request

Use an unchanged baseline as the candidate to validate the complete harness
before opening. The stock `ops/sglang-sample-round/run.sh` is TP4/NVFP4-specific
and must not be used for this campaign. CPU preview below reproduces the sampled
trace from its receipt, but does not replace GPU validation.

The opt-in `ops.pro6000_model_volume` runner below executes the normal v5 harness
and only replaces its model bind mount with a verified Docker volume. It does not
change the request, source pool, C4 scheduling, natural EOS, thresholds or scorer.
A separate volume is prepared for the shadow round and removed on exit; evidence
is retained in `shadow/model_volume.json`. Unset the optional engine-cache bind
path for this portable run so it cannot cause the same filesystem mismatch.
Existing cache files are not removed. After a hard kill, inspect that run's named
containers before manually removing its recorded volume; never prune shared data.

The entire preview → request → shadow sequence runs in one `nohup` job. Pull
the latest PR branch so `ops/shadow-pro6000.sh` is present. It requires
`step2.exit-code` to contain `0`, stops at the first failed stage, rejects a
second attempt in the same run directory, and records `step3.pid` plus
`step3.exit-code`. Its output streams to `step3.log` and the individual stage logs.
No foreground `wait` is needed, and it never seeds a campaign automatically.

```bash
set +e
nohup bash ops/shadow-pro6000.sh >> "$PRO6000_RUN_DIR/step3.log" 2>&1 < /dev/null &
printf 'Launched PID %s; log: %s/step3.log\n' "$!" "$PRO6000_RUN_DIR"
```

Watch separately; Ctrl-C stops only the viewer:

```bash
tail -n 60 -f "$PRO6000_RUN_DIR/step3.log" || true
```

After reconnecting, restore the existing run's variables and read its PID/logs;
do not relaunch. Only a recorded exit code `0` permits report review below. A
missing status is not success: the job may still be running or may have failed
before recording status. On failure, inspect the logs before attempting recovery.

```bash
if [[ -f "$PRO6000_RUN_DIR/step3.exit-code" ]] && \
   [[ "$(cat "$PRO6000_RUN_DIR/step3.exit-code")" == 0 ]]; then
  jq '{verdict, entries, error}' "$PRO6000_RUN_DIR/shadow/bench_report.json"
else
  printf 'Step 3 has not succeeded; inspect %s/step3.log and step3.exit-code.\n' "$PRO6000_RUN_DIR"
fi
```

Inspect the complete report and retained evidence, not just process exit status:
unchanged candidate must be scored, baseline/candidate correctness must pass,
all four tiers must survive exclusions, and drift/repeatability gates must pass.
Review `evidence/correctness/baseline_exclusions.json`, queue-inclusive tier times,
the 0.25 weights, failure deduction, natural output lengths and observed C4
occupancy (allowing exclusions and final drain). Both baseline runs establish
exclusions before the candidate; they are not extra qualification starts.
Verify GPU count, peak VRAM, scorer headroom and cleanup. Confirm the existing
nonempty offline miner-build/native-probe checks for the reused engine; baseline
self-comparison alone does not prove that changed CUDA/Rust kernels execute.
If serving or memory settings change, update both helper and fixture and restart
qualification. Preserve all evidence under this run directory.

#### 4. Seed once on the configured controller, then verify

Only after deployment checks, qualification and shadow validation pass, transfer
`qualification/sampling_rule.json` and its evidence to owner-only storage on the
controller. Load its existing protected environment and activate its virtualenv.
Set the path below to the transferred qualified file, and use the reviewed
checkout whose fixture matches the GPU run. **This command creates an open row**;
do not seed a draft first or rerun it to change a fee.

Start the reviewed seed operation as one background job; the wrapper records
`seed.pid`, `seed.log` and `seed.exit-code` under a protected directory. Keep that
directory path across reconnects. Its lock prevents reuse of the same directory,
but is **not database idempotency**: never create a fresh directory just to retry
an uncertain seed result. Inspect database/API state first.

```bash
set +e
cd /opt/pareton
source .venv/bin/activate
set -a
source .env
set +a
umask 077
export PRO6000_QUALIFIED_RULE=/absolute/path/to/pro6000-qualification/sampling_rule.json
export PRO6000_SEED_DIR=$(mktemp -d /var/tmp/pareton-pro6000-seed-XXXXXX)
nohup bash ops/seed-pro6000-job.sh >> "$PRO6000_SEED_DIR/seed.log" 2>&1 < /dev/null &
printf 'Launched PID %s; evidence: %s\n' "$!" "$PRO6000_SEED_DIR"
```

Watch separately; Ctrl-C stops only the viewer:

```bash
tail -n 60 -f "$PRO6000_SEED_DIR/seed.log" || true
```

Only after `seed.exit-code` contains `0`, review `seed.log` for the new UUID and
run the readback below. Missing/nonzero status requires investigation of both
logs and database/API state before any retry. Readback is a separate manual step;
interrupting its prompt or HTTP request cannot cancel the detached seed job.

```bash
if [[ -f "$PRO6000_SEED_DIR/seed.exit-code" ]] && \
   [[ "$(cat "$PRO6000_SEED_DIR/seed.exit-code")" == 0 ]]; then
  cat "$PRO6000_SEED_DIR/seed.log"
  read -r -p 'New campaign UUID printed by seed: ' PRO6000_CAMPAIGN_ID &&
  curl -fsS "https://api.pareton.ai/v1/campaigns/$PRO6000_CAMPAIGN_ID" \
    > /tmp/pro6000-campaign-readback.json &&
  jq -e '
    .status == "open" and .patch_visibility == {"mode":"private"} and
    .sampling_rule.algo_version == 5 and .sampling_rule.n_prompts == 32 and
    .sampling_rule.request_concurrency == 4 and .sampling_rule.request_timeout_s == 600 and
    .scoring_rule.name == "weighted_tier_completion_speedup" and
    .scoring_rule.tier_weights == {"2k":0.25,"4k":0.25,"8k":0.25,"16k":0.25} and
    .scoring_rule.failure_penalty == 0.1 and .submission_fee.amount_tao == "0.1" and
    .gpu_skus == ["RTXPRO6000"] and .bench.gpu_count == 1
  ' /tmp/pro6000-campaign-readback.json
else
  printf 'Seed success is unconfirmed; inspect %s and database/API state before proceeding.\n' "$PRO6000_SEED_DIR"
fi
```

Also compare both image digests, source/model revisions, serving arguments,
correctness thresholds, natural-EOS policy, emissions, block-zero initial fee
history/recipient and signoff against the reviewed pins. Inspect the first live
report and frontend C4/tier display. For a real submission, verify the scoped
`patch-availability` response remains private with no download location and the
`patch` route returns 403 `patch_private`, including after a finalized evaluation.
Do not change this campaign to `public_after_reveal`; privacy is mutable outside
`manifest_hash`, so monitor the operational policy separately. Existing campaigns
and their signed contracts remain unchanged.

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
