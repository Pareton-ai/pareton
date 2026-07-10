# Pareton (SN10)

**Inference optimization campaigns** on Bittensor SN10.

Stage 0: profile → pinned campaign manifest → miner patch commitment → provenance & build gate → content-addressed engine image.

Docs:

- [`Pareton_Engineering_Architecture_v0.pdf`](Pareton_Engineering_Architecture_v0.pdf)
- [`Pareton_Optimization_Profile.pdf`](Pareton_Optimization_Profile.pdf)
- [`docs/technical-decisions.md`](docs/technical-decisions.md)
- [`docs/stage0-ops-checklist.md`](docs/stage0-ops-checklist.md)

## Layout

| Path | Role |
|---|---|
| `campaign/` | Profiles, manifests, seed CLI |
| `chain/` | Patch commitment parse + chain watcher/RPC |
| `gate/` | Provenance gates a–d |
| `builder/` | Hermetic build + GHCR tagging |
| `storage/` | Pareton-presigned S3 uploads |
| `db/` | Neon schema + connection |
| `worker/` | Job loop (gates → build) |
| `api/` | HTTP API (campaigns, submissions, presign) |
| `miner/commit_patch.py` | Miner commit CLI |
| `fixtures/` | Synthetic campaign fixtures |
| `images/baseline/` | Baseline Dockerfile |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set PARETON_DATABASE_URL

python -m campaign.seed
python -m api                 # API on :8000
python -m worker.main --mock-build --once

pytest tests -q
```

Production worker (VPS) also polls SN10 for commitments:

```bash
python -m worker.main --scan-chain
```

Miner commit (after patch upload via API presign):

```bash
python miner/commit_patch.py \
  --campaign-id <uuid> --patch ./change.diff \
  --api-base http://127.0.0.1:8000 \
  --wallet-name <wallet> --wallet-hotkey default \
  --network finney --netuid 10
```

## License

MIT
