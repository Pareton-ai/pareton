# Run the auditor weight setter

The production `pareton-weights` service computes the dense UID-indexed weight
vector, stores it in Postgres, publishes it at `GET /v1/weights`, and submits it
to Bittensor. `scripts/auditor.py` performs only the last step. It fetches the
published vector and its `version_key`, verifies the validator hotkey on SN10,
and submits the vector with `bt.SetWeights`.

After a successful submission it waits 360 chain blocks. After a failed
submission or fetch it waits 36 chain blocks. It polls the Finney chain every
12 seconds to measure those intervals. It logs each API fetch, metagraph read,
submission, error, and block wait. The chain has commit-reveal enabled, so a
successful Finney submission means the commitment was accepted. The vector
becomes live after the chain reveals it.

## Setup

Run from a Pareton checkout on the VM. Install the repository dependencies and
confirm that the validator wallet and hotkey already exist locally:

```bash
cd /opt/pareton
python -m pip install -r requirements.txt
python scripts/auditor.py --help
```

`--coldkey` is the local Bittensor wallet name. `--hotkey` is the local hotkey
name inside that wallet. The hotkey must be registered on netuid 10 and allowed
to set weights. The script never accepts or prints seed material.

Test one foreground start before moving it to the background:

```bash
cd /opt/pareton
python scripts/auditor.py --coldkey pareton-validator-ckey --hotkey pareton-validator-hkey
```

Stop it with `Ctrl-C` after the first successful log line.

Use `--network` and `--netuid` to target another subnet. `--once` performs one
submission attempt and returns exit code 0 on acceptance or 1 on failure:

```bash
python scripts/auditor.py \
  --coldkey dev_coldkey \
  --hotkey dev_hotkey \
  --network test \
  --netuid 292 \
  --once
```

## Run with PM2

Use the Python executable from the environment where the dependencies were
installed:

```bash
cd /opt/pareton
pm2 start scripts/auditor.py \
  --name pareton-auditor \
  --interpreter "$(command -v python)" \
  -- --coldkey pareton-validator-ckey --hotkey pareton-validator-hkey
pm2 save
pm2 logs pareton-auditor
```

Use `pm2 startup` once if PM2 is not already configured to start after a VM
reboot.

## Run with nohup

```bash
cd /opt/pareton
nohup python scripts/auditor.py \
  --coldkey pareton-validator-ckey \
  --hotkey pareton-validator-hkey \
  > pareton-auditor.log 2>&1 &
echo $! > pareton-auditor.pid
tail -f pareton-auditor.log
```

Stop the nohup process with:

```bash
kill "$(cat /opt/pareton/pareton-auditor.pid)"
```
