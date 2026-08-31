# Run the auditor weight setter

`scripts/auditor.py` mirrors Pareton's published weight vector onto Bittensor.
It fetches `GET https://api.pareton.ai/v1/weights`, checks that the signing
hotkey may set weights on the netuid, and submits the vector unchanged with
`bt.SetWeights`.

The script decides nothing. Pareton's `pareton-weights` service computes the
vector, stores it, and publishes it; this process only signs and relays what
the endpoint returns.

After a successful submission it waits 360 chain blocks. After a failed
submission or fetch it waits 36 blocks. It polls the chain every 12 seconds to
measure those intervals, and logs every fetch, metagraph read, submission,
error, and wait. SN10 has commit-reveal enabled, so a successful submission
means the commitment was accepted; the vector goes live after the reveal.

## Install

The script is standalone. Copy the single file anywhere; it imports nothing
else from this repository.

```bash
curl -O https://raw.githubusercontent.com/Pareton-ai/pareton/main/scripts/auditor.py
python -m pip install 'requests>=2.31' 'bittensor==11.0.2'
python auditor.py --help
```

Nothing else from `requirements.txt` is needed.

## Configure

The network, netuid and API URL are fixed constants at the top of the file:
Finney, netuid 10, `https://api.pareton.ai/v1/weights`. Only the wallet is
configurable.

| Variable | Flag | Meaning |
| --- | --- | --- |
| `PARETON_WALLET_NAME` | `--coldkey` | Local Bittensor wallet name |
| `PARETON_WALLET_HOTKEY` | `--hotkey` | Local hotkey name inside that wallet |

A command-line flag overrides the environment variable. Both are required.

These are local wallet names, not addresses and not seed material. The script
never reads or prints seed material. The hotkey must be registered on netuid 10
and hold a validator permit; the script checks this and exits before signing
anything if it does not.

## Run

Test one foreground start first:

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
once if PM2 is not already set to start after a reboot.

With nohup:

```bash
nohup python auditor.py > pareton-auditor.log 2>&1 &
echo $! > pareton-auditor.pid
tail -f pareton-auditor.log

kill "$(cat pareton-auditor.pid)"
```

Run it under systemd, supervisor, or a container if you prefer. The script is
plain Python with no runtime assumptions beyond the two packages above.
