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
