# Run the weight setter

`scripts/auditor.py` fetches `GET https://api.pareton.ai/v1/weights` and
submits that weight vector on mainnet (Finney) Subnet 10. It does not change the vector.

After a successful submission it waits 360 blocks. After a failure it waits
36 blocks. SN10 uses commit-reveal, so a successful call means the commitment
was accepted; the vector goes live after reveal.

## Install

```bash
# Download the standalone script
curl -O https://raw.githubusercontent.com/Pareton-ai/pareton/main/scripts/auditor.py
# Install the two dependencies
python -m pip install 'requests>=2.31' 'bittensor==11.0.2'
# Confirm the CLI loads
python auditor.py --help
```

## Configure

Only the wallet is configurable. Network, netuid, and the API URL are fixed
in the script.

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
export PARETON_WALLET_NAME=my-validator-coldkey
export PARETON_WALLET_HOTKEY=my-validator-hotkey

python auditor.py --once   # one attempt, exit 0 on acceptance, 1 on failure
python auditor.py          # run forever
```

Stop the foreground process with `Ctrl-C`.

## Run in the background

With PM2:

```bash
pm2 start auditor.py --name pareton-auditor --interpreter "$(command -v python)"
pm2 save
pm2 logs pareton-auditor
```

PM2 inherits the environment of the shell that started it. Run `pm2 startup`
once if it should start after a reboot.

With nohup:

```bash
nohup python auditor.py > pareton-auditor.log 2>&1 &
echo $! > pareton-auditor.pid
tail -f pareton-auditor.log

kill "$(cat pareton-auditor.pid)"
```
