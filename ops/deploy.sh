#!/usr/bin/env bash
# Run once on the host, or --watch in the independent deploy Compose project.
set -euo pipefail
umask 077

# Snapshot a host invocation before fetch can replace the running script.
if [[ ${PARETON_DEPLOY_SNAPSHOT:-0} != 1 ]]; then
    deploy_copy=$(mktemp)
    cp -- "$0" "$deploy_copy"
    export PARETON_DEPLOY_SNAPSHOT=1
    export PARETON_DEPLOY_SNAPSHOT_PID=$$
    # Keep the watch shell as the signal recipient, including under Docker's
    # init. A wrapper shell would exit on SIGTERM and interrupt the rollout.
    exec bash "$deploy_copy" "$@"
fi
if [[ ${PARETON_DEPLOY_SNAPSHOT_PID:-} == $$ ]]; then
    # One-shot children reuse this file; only its owning watch shell removes it.
    trap 'rm -f -- "$0"' EXIT
fi

REPO=${PARETON_REPO_DIR:-/opt/pareton}
cd "$REPO"
ENV_FILE=${PARETON_ENV_FILE:-$REPO/.env}
# Same shell-compatible environment file as the former deployment script.
set -a
source "$ENV_FILE"
set +a
if [[ ${COMPOSE_PROJECT_NAME:-pareton} == pareton-deploy ]]; then
    echo 'pareton-deploy is reserved for the independent deployment controller' >&2
    exit 2
fi
export PARETON_ENV_FILE="$ENV_FILE"
STATE="$REPO/.deploy-state"
mkdir -p "$STATE"
chmod 700 "$STATE"

compose() {
    docker compose --project-directory "$REPO" -f "$REPO/compose.yaml" "$@"
}

redeploy() (
    exec 9>"$STATE/lock"
    flock -n 9 || exit 0
    if [[ ${1:-} != --local ]]; then
        git fetch --quiet origin main
        remote=$(git rev-parse origin/main)
        if [[ -f "$STATE/done" && $(cat "$STATE/done") == "$remote" ]]; then
            exit 0
        fi
        # Never discard operator changes or diverged history.
        test -z "$(git status --porcelain --untracked-files=no)"
        git merge --ff-only --quiet "$remote"
    fi
    export PARETON_CODE_SHA
    PARETON_CODE_SHA=$(git rev-parse HEAD)
    mkdir -p "$STATE/releases"
    # A retry or --local redeployment can reuse the SHA with new config or
    # dependencies. Never overwrite the last working image or mounted configs.
    release=$(mktemp -d "$STATE/releases/$PARETON_CODE_SHA.XXXXXXXX")
    export PARETON_RUNTIME_IMAGE="pareton-runtime:${release##*/}"
    cp "$REPO/ops/vector/vector.toml" "$release/vector.toml"
    cp "$REPO/ops/caddy/Caddyfile" "$release/Caddyfile"
    export PARETON_VECTOR_CONFIG="$release/vector.toml"
    export PARETON_CADDY_CONFIG="$release/Caddyfile"
    next="$STATE/next.yaml"
    # Retain opt-in services such as cli in the saved release. Enable profiles
    # only for rendering; their profile gates still apply to ordinary startup.
    compose --profile '*' config > "$next"
    # Build before stopping production; use the same builder and storage lock
    # as miner builds, cleanup and ccache backup/restore.
    work=${PARETON_HOST_WORK_DIR:-$REPO/.pareton-work}
    mkdir -p "$work"
    flock "$work/builder-storage.lock" \
        docker compose --project-directory "$REPO" -f "$next" build \
        --builder "${PARETON_BUILDER_NAME:-default}" api
    docker compose --project-directory "$REPO" -f "$next" pull vector caddy
    docker compose --project-directory "$REPO" -f "$next" run --rm --no-deps \
        --entrypoint vector vector validate --no-environment /etc/vector/vector.toml
    docker compose --project-directory "$REPO" -f "$next" run --rm --no-deps \
        --entrypoint python cli -m builder.preflight

    previous="$STATE/current.yaml"
    running="$next"
    if [[ -f "$previous" ]]; then running="$previous"; fi
    # Keep API/Vector available while the current work and weight cycle drain.
    docker compose --project-directory "$REPO" -f "$running" stop watcher
    docker compose --project-directory "$REPO" -f "$running" stop worker round-worker weights
    docker compose --project-directory "$REPO" -f "$running" down --remove-orphans
    if docker compose --project-directory "$REPO" -f "$next" up -d --no-build --wait --wait-timeout 180; then
        # Normal `docker compose run cli` must use the deployed code too.
        docker image tag "$PARETON_RUNTIME_IMAGE" pareton-runtime:local
        mv "$next" "$previous"
        printf '%s\n' "$PARETON_CODE_SHA" > "$STATE/done"
        printf 'deploy: %s running\n' "$PARETON_CODE_SHA"
    else
        echo 'deploy: startup failed; restoring the previous Compose release' >&2
        docker compose --project-directory "$REPO" -f "$next" down --remove-orphans
        if [[ -f "$previous" ]]; then
            docker compose --project-directory "$REPO" -f "$previous" up -d --no-build --wait --wait-timeout 180
        fi
        exit 1
    fi
)

case ${1:-} in
    --watch)
        stopping=0
        trap 'stopping=1' TERM INT
        while [[ "$stopping" == 0 ]]; do
            # Run a fresh oneshot shell: bash suppresses errexit throughout a
            # function called in an if/|| condition, including its subshell.
            if ! bash "$0"; then echo 'deploy: failed; retrying next tick' >&2; fi
            [[ "$stopping" == 0 ]] || break
            sleep "${PARETON_DEPLOY_INTERVAL_S:-60}" &
            wait $! || true
        done
        ;;
    --local|"") redeploy "${1:-}" ;;
    *) echo 'usage: pareton-deploy [--watch|--local]' >&2; exit 2 ;;
esac
