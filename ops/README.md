# ops/

Deployment artifacts for the production VPS. Files here are **verbatim copies of
what runs in production** — not templates. Change the copy here first, then
re-install it on the box, so the two never drift.

## Layout

| Path | Installed to | Notes |
|---|---|---|
| `systemd/pareton-api.service` | `/etc/systemd/system/` | uvicorn on `0.0.0.0:8000` |
| `systemd/pareton-worker.service` | `/etc/systemd/system/` | Gate + bench worker |
| `systemd/pareton-watcher.service` | `/etc/systemd/system/` | Chain ingest, `python -m worker.watcher` |
| `systemd/pareton-deploy.service` | `/etc/systemd/system/` | Oneshot, invoked by the timer |
| `systemd/pareton-deploy.timer` | `/etc/systemd/system/` | **Fires every 60s** |
| `deploy.sh` | `/usr/local/bin/pareton-deploy` | The pull-deploy script itself |
| `gpu/pareton-gpu-reap.service` | `/etc/systemd/system/` | Oneshot GPU TTL reap |
| `gpu/pareton-gpu-reap.timer` | `/etc/systemd/system/` | Fires every 10 min |
| `vector/vector.service` | `/etc/systemd/system/` | Log shipping |
| `vector/vector.toml` | `/etc/vector/` | Axiom sink, dataset `pareton-prod` |
| `caddy/Caddyfile` | `/etc/caddy/` | TLS terminator, proxies to `127.0.0.1:8000` |

`deploy.sh` installs to `/usr/local/bin` rather than running from the repo
checkout so that a `git pull` cannot rewrite the script while it is executing.
That is also why it needs re-installing by hand after a change here.

## A merge to `main` is a production deploy

`pareton-deploy.timer` polls `origin/main` every 60 seconds. There is no
separate promote step. Any merge restarts `pareton-api` and `pareton-watcher`
within a minute, and restarts `pareton-worker` on the next tick where no
`submission_jobs` row is `running`.

**During a maintenance window, stop this timer first.** Stopping any other unit
while the timer is live means the timer may restart it underneath you.

## Known drift — needs a decision

Captured from the live boxes on 2026-08-17. These are *not* resolved here,
because each one changes production behavior:

1. **`systemd/pareton-worker.service` does not match the live unit.** The
   committed file carries a `[Unit]` section, `Type=simple`, `User=root` and
   `Restart=on-failure`. The live unit has none of those and uses
   `Restart=always`. Installing the committed file would change restart
   behavior. Decide which is intended, then make both sides agree.

2. **`TimeoutStopSec` is set in two places with different values.** The
   committed unit says `8h`. Both boxes carry a hand-installed drop-in at
   `/etc/systemd/system/pareton-worker.service.d/timeout.conf` pinning `4h`,
   and **drop-ins override the unit file** — so the effective value today is
   `4h`, not the `8h` the unit asks for. Either delete the drop-in when this
   deploys, or change the unit to `4h`.

3. **`aws/pareton-api-iam-policy.json` overstates the live IAM policy.** It
   grants `s3:ListBucket` and `s3:DeleteObject`; the live `pareton-api` user
   has neither. Only `PutObject`/`GetObject` on `stage0/*` actually work.
   Harmless today — `storage/s3.py` only calls `put_object` — but the file
   should not be treated as an accurate record of live permissions.

4. **`vector/vector.toml` does not match the live config.** The committed file
   reads the Axiom token from `${PARETON_AXIOM_TOKEN}`; the live file has a
   literal token inlined, and the env var holds a *different* token that the
   Vector Axiom sink rejects. The env-var form is the better design — it needs
   the correct token in `.env` before it will work.

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
