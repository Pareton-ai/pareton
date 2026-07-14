# Stage 0 Ops Checklist

Everything code-side is done; these are the manual/account steps to go from
"mock build on a laptop" to "real gate pipeline on SN10". Work top to bottom.

## 1. AWS S3 (`pareton-s3`)

Bucket: `s3://pareton-s3` in **`us-east-2`** (not us-east-1), prefix `stage0/`.

- [x] Bucket `pareton-s3` exists (us-east-2).
- [x] IAM user `pareton-api` with `PutObject`/`GetObject` on `stage0/*` (policy `pareton-api-s3-stage0`).
- [x] Access key created; credentials in local `.env` (never commit).
- [x] Bucket policy: public **read** on `stage0/campaigns/*` only (worker plain-HTTPS GET).
- [x] Smoke test passed: presigned PUT + public GET via `storage/s3.py`.

Set on the VPS (same values as local `.env`):

```dotenv
PARETON_S3_ACCESS_KEY=...
PARETON_S3_SECRET_KEY=...
PARETON_S3_BUCKET=pareton-s3
PARETON_S3_REGION=us-east-2
```

## 2. GitHub / GHCR

Org: [`Pareton-ai`](https://github.com/Pareton-ai) (already exists, holds the
vLLM fork). GHCR namespaces are lowercase: `ghcr.io/pareton-ai/...`.

- [x] Fix local auth: `gh auth refresh -h github.com` (done 2026-07-09; scopes: gist, read:org, repo, workflow).
- [x] Create the repo [`github.com/Pareton-ai/pareton`](https://github.com/Pareton-ai/pareton) (private) and push `main` (done 2026-07-09).
- [x] Create a classic PAT with `write:packages` (authorized for the Pareton-ai
      org) for the builder (done 2026-07-14; in local `.env`, verified vs ghcr.io).
      Set on the VPS:

```dotenv
PARETON_GHCR_OWNER=pareton-ai
PARETON_GHCR_USERNAME=xavierlyu
PARETON_GHCR_TOKEN=ghp_...
```

## 3. Baseline image

The campaign manifest needs a **real** `base_image_digest` before real builds.

- [ ] On the VPS (or any amd64 box with Docker):

```bash
docker build -t ghcr.io/pareton-ai/pareton-baseline:v0 images/baseline
docker push ghcr.io/pareton-ai/pareton-baseline:v0
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/pareton-ai/pareton-baseline:v0
```

- [ ] Re-seed (or seed the first real campaign) passing that digest so the
      manifest stops using the `sha256:bbb...` placeholder.

Baseline pin (already the seed default): vLLM **v0.24.0** =
`ee0da84ab9e04ac7610e28580af62c365e898389`.

There are **two** baseline images (don't conflate them):

| Image | Purpose | Status |
|---|---|---|
| `pareton-baseline:v0` (build base) | compile environment miners' patches build in | this section |
| `pareton-engine:baseline` (serving engine) | vanilla vLLM at the pin, serving-ready; the "before" side of every bench comparison and the manifest's future `baseline_engine_image_digest` | see roadmap WS-A item 2b |

- [ ] Build + push `ghcr.io/pareton-ai/pareton-engine:baseline` (hermetic builder
      with an empty patch is the simplest path) and record its digest.

## 4. Bittensor wallet / hotkey (SN10)

Chain **reads** (metagraph, revealed commitments) need no wallet, so the watcher
runs without one. A wallet is needed for miner test commits now and
weight-setting later (Stage 1+).

- [ ] Create owner coldkey + a hotkey:

```bash
btcli wallet new_coldkey --wallet.name pareton-owner
btcli wallet new_hotkey --wallet.name pareton-owner --wallet.hotkey watcher
```

- [ ] Back up both mnemonics offline. The coldkey controlling SN10 is the crown jewel.
- [ ] For an end-to-end test commit on finney, register a throwaway miner hotkey
      on netuid 10 and run `miner/commit_patch.py` against the live API.
- [ ] Set on the VPS once weight-setting lands:

```dotenv
PARETON_WALLET_NAME=pareton-owner
PARETON_WALLET_HOTKEY=watcher
```

## 5. VPS deploy (worker + builder + watcher, one box)

One dedicated CPU VPS (8+ cores / 16 GB+ RAM recommended — vLLM compiles are
30–60+ min; more cores = faster gate turnaround). Hetzner/OVH dedicated CPU
tiers are fine.

- [ ] Install Docker + Python 3.11+, clone the repo, `pip install -r requirements.txt` in a venv.
- [ ] Copy `.env` (Neon URL + S3 keys + GHCR token). Never commit it.
- [ ] Run the API and worker (systemd units or `docker compose`; simplest is two units):

```ini
# /etc/systemd/system/pareton-worker.service
[Service]
WorkingDirectory=/opt/pareton
EnvironmentFile=/opt/pareton/.env
ExecStart=/opt/pareton/.venv/bin/python -m worker.main --scan-chain
Restart=always

# /etc/systemd/system/pareton-api.service
[Service]
WorkingDirectory=/opt/pareton
EnvironmentFile=/opt/pareton/.env
ExecStart=/opt/pareton/.venv/bin/python -m api
Restart=always
```

- [ ] Put the API behind TLS (Caddy/nginx) so miners can hit the presign endpoint.

`--scan-chain` makes the single worker process poll SN10 revealed commitments
each cycle, enqueue new submissions, and use the live metagraph hotkeys for the
identity gate. Without the flag it only drains the DB queue (local/mock mode).

## 6. Open the first campaign

Whenever ready — no fixed cadence.

- [ ] Seed with real pins: `python -m campaign.seed --base-image-digest sha256:<real> ...`
- [ ] Flip status to `open` (seed CLI flag or SQL) and share the campaign id +
      API base URL with miners.
