# Run the weight setter

`scripts/auditor.py` fetches `GET https://api.pareton.ai/v1/weights` and
submits that weight vector on mainnet (Finney) Subnet 10. It does not change the vector.

After a successful submission it waits 360 blocks. After a failure it waits
36 blocks. SN10 uses commit-reveal, so a successful call means the commitment
was accepted; the vector goes live after reveal.

## Install with Docker Compose

Use Docker Engine and Compose v2. The auditor runs in its own Compose project and
only needs its validator wallet; it does not require Pareton backend credentials.
Use it as the weight-setting process for that wallet, with no second weights
process signing from the same hotkey.

```bash
git clone https://github.com/Pareton-ai/pareton.git
cd pareton
export PARETON_WALLET_NAME=my-validator-coldkey
export PARETON_WALLET_HOTKEY=my-validator-hotkey
export PARETON_WALLET_DIR="$HOME/.bittensor/wallets"
PARETON_CODE_SHA=$(git rev-parse HEAD) docker compose -f ops/compose.auditor.yaml build
docker compose -f ops/compose.auditor.yaml run --rm auditor python scripts/auditor.py --help
```

## Configure

Network, netuid, and the API URL are fixed in the script. Put the wallet settings
in a private `.env` file at the repository root for subsequent Compose commands,
or keep exporting them in your shell. `PARETON_WALLET_DIR` is the existing host
wallet directory mounted read-only into the container.

| Variable                | Flag        | Meaning                              |
| ----------------------- | ----------- | ------------------------------------ |
| `PARETON_WALLET_NAME`   | `--coldkey` | Local Bittensor wallet name          |
| `PARETON_WALLET_HOTKEY` | `--hotkey`  | Local hotkey name inside that wallet |

A flag overrides the environment variable. Both are required. These are local
wallet names, not addresses or seeds. The hotkey must be registered on netuid
10 with a validator permit; the script checks that and exits before signing
if it is not.

## Run

```bash
# One attempt, exit 0 on acceptance, 1 on failure:
docker compose -f ops/compose.auditor.yaml run --rm auditor python scripts/auditor.py --once
# Run continuously, including after host restart:
docker compose -f ops/compose.auditor.yaml up -d
docker compose -f ops/compose.auditor.yaml logs -f --tail 100
```

Stop with:

```bash
docker compose -f ops/compose.auditor.yaml down
```

## Update

```bash
git fetch origin main
git merge --ff-only origin/main
PARETON_CODE_SHA=$(git rev-parse HEAD) docker compose -f ops/compose.auditor.yaml build
docker compose -f ops/compose.auditor.yaml down
docker compose -f ops/compose.auditor.yaml up -d
```

The Python file remains independently runnable for one-off use; the container
provides its dependencies and virtualenv for the managed service.
