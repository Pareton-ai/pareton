# Docker Compose operations

Pareton's API, submission worker, round worker, watcher, weights, GPU reaper,
builder cleanup, Vector and Caddy run in `compose.yaml`. Postgres remains Neon;
S3, GHCR and GPU providers retain their existing roles. Application containers
share one runtime image with a virtualenv at `/opt/venv`. Host Python is not needed.

## First start

Use a Linux Docker host with Docker Engine and Compose v2.39 or newer. The host
still supplies disk, swap and the Docker daemon. GPU hosts additionally need
working NVIDIA drivers; bootstrap verifies them and installs the NVIDIA container
runtime only when absent. Do not restart the builder daemon during active work.

```sh
cd /opt/pareton
cp .env.example .env
# Fill .env with the existing database, S3, GHCR, provider and Axiom credentials.
# Set host paths, API domain and the existing Buildx builder name.
chmod 600 .env
mkdir -p /opt/pareton/.pareton-work /var/log/pareton/builds
mkdir -p /root/.cache/pareton/gpu /root/.docker
# Install the existing validator wallet in PARETON_WALLET_DIR before starting.
docker compose config --quiet
PARETON_CODE_SHA=$(git rev-parse HEAD) docker compose build api
docker compose run --rm --no-deps cli python -m builder.preflight
docker compose up -d --wait
```

The default topology co-locates API and submission worker, since build-log API
responses read the shared build-log filesystem. Configure the same mounts across
all services. For a split-host deployment, explicitly provide shared log storage;
a named Docker volume does not share files between machines. Run only one weights
process for a validator wallet. The `cli` service is opt-in and never runs at boot.

Validators running only the standalone auditor can use the independent
`ops/compose.auditor.yaml` project; see [auditor setup](../docs/auditor.md).

The API binds loopback port 8000. Caddy exposes 80/443 and routes to `api:8000`.
API readiness checks both HTTP and Postgres. Worker health checks use the same
background heartbeat that continues during long builds and rounds. A Docker
health failure is visible in `compose ps`; restart policies handle process exits,
while Axiom alerts continue to detect missing heartbeats and stalled progress.

## Builder and ccache

Keep `PARETON_BUILDER_NAME=default` when migrating an existing Docker-driver
builder. Worker, cleanup, cache CLI and deployment builds all select that builder.
`PARETON_DOCKER_CONFIG_DIR` mounts the Docker client's credentials and Buildx
metadata; copying only the socket would lose discovery of a named builder.
The Docker storage filesystem is mounted read-only so cleanup measures its actual
usage. Worker, cleanup and cache commands share the same file lock under WORK_DIR.

For the `docker` driver, merge `ops/docker/daemon.json` into the existing host
configuration. Preserve every unrelated setting and the current image store.
Validate the merged JSON with `dockerd --validate`, then restart Docker during a
maintenance window. `builder.preflight` requires `builder.gc.enabled=false`.
Containers read the host config through a read-only bind mount; no container
rewrites it. The configured file and Docker storage directory must already exist.

For a **new, separate named builder**, create it once using the checked-in policy:

```sh
docker buildx create --name pareton --driver docker-container \
  --buildkitd-config ops/buildkitd.toml --bootstrap
# Set PARETON_BUILDER_NAME=pareton in .env, then:
docker compose run --rm --no-deps cli python -m builder.preflight
```

For this driver, startup checks the selected node's mounted BuildKit configuration
and requires both workers' GC settings to be disabled. Unsupported drivers and
multiple nodes fail closed. Use a builder attached to this same Docker host.
Do not delete/recreate a warmed builder: its compiler caches belong to that
builder. Restore a trusted snapshot when intentionally switching builders.

```sh
docker compose run --rm --no-deps cli python -m builder.cleanup --dry-run --force
docker compose run --rm --no-deps cli python -m builder.cache --help
```

See [cache backup/restore](../docs/build-cache.md) for the full arguments. Run all
manual build/cache commands through `cli` so they share the selected builder and
storage lock. Filtered cleanup retains active baselines and excludes
`exec.cachemount`; it never runs a daemon-wide image/system prune or deletes GHCR
artifacts. Cleanup runs hourly and skips when the build lock is busy.

The host's RAM/swap must still support CUDA compilation. Existing 16 GB build
hosts relied on a 64 GB swapfile. Compose does not provision swap or put a memory
limit on sibling BuildKit builds; keep `PARETON_BUILD_MAX_JOBS` appropriate for the
actual builder host. Check disk and swap before the first real miner build.

The A2b baseline helper also runs in the Compose CLI image:

```sh
export BASE=ghcr.io/pareton-ai/pareton-baseline@sha256:YOUR_A2_DIGEST
export TORCH_CUDA_ARCH_LIST=9.0
bash ops/a2b-build.sh --detach
# Follow the returned container ID with docker logs -f ID.
```

It validates the selected builder, smoke-checks the base, builds and pushes the
empty-patch engine, then checks its imports. Credentials come from `.env` with
explicit shell overrides supported. Compiler job/timeout defaults are two jobs
and eight hours. Its context uses the shared work directory, and build logs remain
in the shared build-log directory after the one-off container exits.

## Updates and rollback

Run an update from the host with `bash ops/deploy.sh`, or enable the independent
Docker deployment controller below. The script fetches `origin/main`, merges only
fast-forward changes, builds the new runtime image, pulls infrastructure images,
and validates Vector and the selected builder **before stopping services**.
Tracked operator changes or divergent Git history stop the update.

It stops the watcher, drains both execution workers and weights while API/Vector
remain available, then performs `docker compose down` followed by `up -d --wait`.
The full down/up step has an API interruption; this is not a zero-downtime rollout.
Workers have an eight-hour grace period and weights fifteen minutes. No timeout
flag shortens these settings. An expired grace period can still force-kill a job;
inspect failed/stranded work and GPU teardown before resuming after such a failure.

No deployment removes volumes. Runtime images use the source commit as their tag.
After successful startup, the `pareton-runtime:local` alias advances to the same
image so ordinary `docker compose run cli` commands use the deployed code.
Rendered configuration, environment, and copies of Vector/Caddy configuration live
under `.deploy-state` (mode 0700/0600, excluded from Git and image contexts). This
contains secrets and must remain private. The deployed marker advances only after
startup succeeds. A startup failure restores the previous saved Compose release;
the next polling tick retries the new commit. Previous images must remain locally
available for rollback. Database schema changes still need their own compatible
migration; this script never rewrites production schema.

Enable the original automatic main-to-production behavior with a second project:

```sh
# PARETON_REPO_DIR and PARETON_HOST_WORK_DIR must be absolute and match the host.
docker compose -p pareton-deploy --env-file .env -f ops/compose.deploy.yaml up -d --build
docker compose -p pareton-deploy -f ops/compose.deploy.yaml logs -f
```

The controller polls every 60 seconds and executes its baked-in script; a Git
fetch cannot rewrite the running script. It has the host Docker socket, Docker
client configuration, source checkout and read-only deployment SSH keys. Configure
a working Git remote/authentication before enabling it. It manages the `pareton`
application project, so application `down` never stops the controller itself.
Rebuild this separate project when the deployment script or controller definition
changes. All host bind paths must agree inside/outside the controller.

Pause automatic deployment before maintenance or rollback:

```sh
docker compose -p pareton-deploy --env-file .env -f ops/compose.deploy.yaml stop
# Restart the last successful release, if needed:
docker compose -f .deploy-state/current.yaml up -d --no-build --wait
```

`bash ops/deploy.sh --local` deploys the currently checked-out commit without
fetching, including for an operator-selected rollback. Stop the controller before
checking out an older commit, and resume it only when main should deploy again.

## GPU execution

Remote bootstrap installs/checks Docker, NVIDIA support and rsync, ships the
trusted source, then builds the `runtime` Dockerfile target on the GPU host. Python
and `python -m venv` run **inside that image**; no host virtualenv is installed.
The image records the source revision, including when the coordinating worker has
no `.git` directory. Repeated runs on a retained/static host can reuse image layers.

The harness uses the host socket to create sibling engine containers, host
networking for engine health/completion access, and GPU utility access for its
hardware evidence. `/opt/pareton`, `/workspace/hf-cache`, and
`/workspace/engine-cache` are mounted at identical paths inside and outside the
harness so engine mounts resolve correctly. Phase/status files and final evidence
remain visible to SSH polling and rsync; benchmark exit codes survive cleanup.
Registry authentication uses `/opt/pareton/.docker` in both host and harness.
Secrets, keys, environments, runtime state and old output are excluded from the
build context/source transfer.

Provider fallback, per-round pod reuse, model and engine caches, evidence upload,
`--keep`, and TTL teardown retain their existing behavior. The GPU reaper runs
every ten minutes and shares the durable registry/SSH directory with round-worker
and `cli`. `static_ssh` keys must be placed under that mounted directory (and
`PARETON_GPU_SSH_KEY_PATH` set accordingly), or explicitly mounted at their
configured path. The inference containers themselves receive no Docker socket.

## Vector and Axiom

Vector collects this Compose project's Docker stdout/stderr, removes DEBUG noise,
and parses lifecycle JSON into the existing top-level event fields. Service labels
supply `_SYSTEMD_UNIT`, preserving the current Axiom filters:

```apl
['pareton-prod']
| where event == "heartbeat" and _SYSTEMD_UNIT == "pareton-worker.service"
| summarize count()
```

The round monitor uses `pareton-round-worker.service`. Retain separate fixed
filters, **Below 1 over 15 minutes**, and **Alert on no data**. Weights, watcher and
GPU lifecycle event names are unchanged. Builder cleanup telemetry is now included.
`PARETON_AXIOM_DATASET` controls the dataset, and `PARETON_AXIOM_TOKEN` must contain
the working ingest token; historical production notes recorded a mismatch between
the old inline token and the environment token, so verify the actual credential.

Vector's 512 MiB disk buffer survives recreation in `vector-data`. Docker logs are
rotated at 20 MiB x 5 per application service. Vector starts before application
services and stops after them; the Docker log source is still best effort, so the
buffer protects already-collected events, not logs missed during collector outages.
Confirm ingestion and both heartbeat monitors during cutover. No live Axiom
configuration is changed by this repository.

## One-time migration from systemd

The old unit files have been removed from the repository. They remain installed
on existing hosts until an operator disables them. Before the first Compose start:

1. Disable the old deploy timer first so it cannot restart services underneath the
   migration. Stop/disable the GPU reap and builder cleanup timers, then the
   watcher, both workers and weights, allowing their running jobs to drain.
2. Preserve the existing work directory, build logs, GPU registry and keys. Point
   `.env` at those paths. GPU state is mounted at its original absolute path,
   because `pods.json` stores absolute key paths. Preserve Docker/Buildx metadata,
   the selected builder, all ccache state and the validator wallet.
3. Stop host Caddy and Vector. Copy Vector's existing data directory into the
   `pareton_vector-data` volume. Copy host Caddy's data into
   `pareton_caddy-data` under `caddy/` so certificates/account state survive.
   These volumes can be created ahead of time with `docker volume create`.
   Preserve ownership and do not copy live, actively written buffer files.
4. Disable every old Pareton service/timer plus host Vector/Caddy; ensure no old
   weights process or execution worker remains. Retain their previous configuration
   privately until the migration has been verified. Resolve the old 4h/8h worker
   drop-in discrepancy in the explicit Compose grace setting.
5. Run the startup commands, inspect `docker compose ps` and logs, verify API,
   Axiom, build-log access and a real build/round, then enable the Docker deploy
   controller. A rollback to systemd must stop the Compose stack/controller first.

## Daily commands and checks

```sh
docker compose ps
docker compose logs -f --tail 100 worker round-worker
docker compose logs -f vector
docker compose run --rm --no-deps cli python -m campaign --help
docker compose run --rm --no-deps cli python -m gpu reap --dry-run
```

Container integration checks (isolated test database; no cloud spend or signing):

```sh
PARETON_CODE_SHA=$(git rev-parse HEAD) docker compose build api
python3 scripts/smoke_compose.py
```

CI also runs the mock engine lifecycle from inside the runtime image through the
Docker socket. Real provider/GPU and production Axiom verification is a cutover
check; mock tests cannot establish those external services are configured correctly.
