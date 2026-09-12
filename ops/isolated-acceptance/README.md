# Isolated acceptance (spec section 9: systemd-required scenarios)

Runs A2 (install), A6 (restart failure + rollback), A8 (ops self-update),
A19 (Vector restart continuity) and the systemd side of A9/A17 (OnFailure
chain, notifier state) against **real systemd and real Vector 0.57.0** inside a
disposable privileged container. Verified 2026-09-10: 25/25 assertions pass
(see EVIDENCE-2026-09-10.md).

## Build the image

The container needs `ubuntu:24.04`, systemd, python3, git, and the vector
binary. When the default registries/releases are unreachable, both can come
from mirrors:

```sh
docker pull docker.m.daocloud.io/library/ubuntu:24.04
docker tag docker.m.daocloud.io/library/ubuntu:24.04 ubuntu:24.04
docker pull docker.m.daocloud.io/timberio/vector:0.57.0-debian
docker create --name vx docker.m.daocloud.io/timberio/vector:0.57.0-debian
docker cp vx:/usr/bin/vector ./vector && docker rm -f vx   # match your arch
docker build -t pareton-systemd-verify .
```

The Dockerfile swaps apt to a CN mirror; drop that line on an unrestricted
network.

## Run

```sh
docker run --privileged --cgroupns=host -d --name pareton-verify \
  pareton-systemd-verify
# wait for "running"
docker exec pareton-verify systemctl is-system-running
COPYFILE_DISABLE=1 tar cf - --exclude .venv --exclude .venv-ops \
  --exclude __pycache__ --exclude .pytest_cache --exclude '._*' . |
  docker exec -i pareton-verify sh -c 'mkdir -p /opt/pareton && tar xf - -C /opt/pareton'
docker cp ops/isolated-acceptance/acceptance.sh pareton-verify:/root/
docker exec pareton-verify bash /root/acceptance.sh
```

## Documented deviations from production

- The Axiom sink is replaced by a file sink (no real token in isolation);
  token-presence gating still runs against a fake value.
- The webhook points at `.invalid`, so the send path fails by design — this
  exercises A17 (send failure must not open the suppression window). Real
  channel delivery stays in the production acceptance window (A9).
- `docker.service` is stubbed (the production host runs Docker; per-file
  `systemd-analyze verify` resolves unit dependencies against the live fs).
- `.venv/bin/python` is a stand-in: long-running modules become a stable
  sleep, quick helpers exit 0, and `-c` probes run the real interpreter with
  a fake `db` module (idle or injected error).

## Stage-2 acceptance (stage2-acceptance.sh)

Covers the systemd-facing slice of the stage-2 matrix on real systemd +
real Vector: bootstrap via `request reset`, failure drill + acceptance
record + `request verify`, `request unpause` and a full A→B release
(drain → quiesce → apply → re-exec → verify with ExecCondition, ExecStartPre
probes, in-process probe threads, one-shot GPU reap dispatch, per-unit Axiom
checks), the ExecCondition gate matrix (apply-block=skip, corrupt=255/failed
+ `release_gate_error`), a live activity-lock holder holding the tick at
`active-work`, GPU dispatch consumption semantics, a missing-source
fault injection (B15), and — S8 — a real rollback: after a verified A→B
release (with a B-era marker planted in the live venv), `request rollback`
must restore the venv from the recovery copy (marker gone), move the
checkout and `.deploy-done` back to A, and keep the hold. Verified
2026-09-12: 47/47 assertions pass.

```sh
docker run --privileged --cgroupns=host -d --name pareton-s2 pareton-systemd-verify
# wait for "running", then stage the stage2 worktree WITHOUT its .git
# (a worktree's .git is a file pointing at the main repo) and re-init:
tar cf - --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
  --exclude '._*' --exclude .git . |
  docker exec -i pareton-s2 sh -c 'mkdir -p /opt/pareton && tar xf - -C /opt/pareton'
docker exec pareton-s2 sh -c 'git config --system --add safe.directory /opt/pareton;
  cd /opt/pareton && git init -q -b main && git config user.email iso@test &&
  git config user.name isolated && git add -A && git commit -qm baseline'
docker cp ops/isolated-acceptance/stage2-acceptance.sh pareton-s2:/root/
docker exec pareton-s2 bash /root/stage2-acceptance.sh
```

Additional documented deviations (on top of the stage-1 list): the checker
queries an in-container mock Axiom HTTP server that replays Vector's file
sink; a managed `pareton-deploy.service.d/isolation.conf` drop-in (committed
container-side) points the checker at it; long-running apps are stand-ins
that run the REAL `observability.probe` loop and a stub HTTP API; the drill
notifier's Discord send fails against `.invalid` by design while its
structured event still ships; "reboot the host" scenarios are approximated
by stopping all units and clearing `/run` coordination files.

The stage-1 `acceptance.sh` remains valid for its sync-config-only
scenarios; its deploy-unit paths (A8 self-install, A9 probe-failure via the
old bash script) are superseded by `stage2-acceptance.sh` now that
`ops/deploy.sh` is a wrapper.
