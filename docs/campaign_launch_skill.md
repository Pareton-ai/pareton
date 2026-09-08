---
name: campaign-launch
description: "Take a Pareton campaign from an idea (this model, this GPU, this traffic shape, etc.) to an open row in the campaigns table (under Neon postgres). Covers model feasibility math, engine image build, ccache warming, a paid dry run on the target GPU to measure what cannot be derived offline, and the seed invocation. Use whenever the user says they want to launch or seed a new campaign, picks a model/hardware pair, or asks what is left before a campaign can open."
version: 1.0.0
category: ops
metadata:
  hermes:
    tags: [pareton, campaign, launch, vllm, ccache, seed, bittensor, gpu]
  trigger: "User wants to launch a new campaign, names a model + GPU pair, or asks what blocks opening a campaign."
---

# Pareton campaign launch

Turn "let's optimize model X on GPU Y" into a row in `campaigns` with `status='open'`
that miners can actually submit against.

The row is the last step, not the first. A campaign seeded before the engine image,
the warm ccache, and the measured thresholds exist is a campaign that rejects every
miner who touches it.

**Never invent a measured number.** Sampling revisions, model SHAs, image digests and
correctness thresholds are either read from a live source or measured on hardware.
A guessed digest fails closed; a guessed threshold fails silently and corrupts scoring.

## Order of operations

Each step's output is the next step's input. Do not reorder.

```
1. Feasibility math        (free, offline)      -> does it fit the GPU at all
2. Engine image            (~8h build, VPS)     -> base_image_digest
3. Warm the ccache         (same build)         -> miner builds take ~6min not ~6h
4. Verify warmth           (~7min, VPS)         -> proof, not inference
5. Dry run on target GPU   (~1h, ~$6-12)        -> thresholds + does it even load
6. Seed the row            (seconds)            -> status=draft, then open
```

## 1. Feasibility math (do this before spending anything)

Fetch the model config directly. Do not rely on model cards or memory.

```bash
curl -s "https://huggingface.co/api/models/<org>/<model>" | jq -r '.sha, .safetensors.parameters'
curl -s "https://huggingface.co/<org>/<model>/raw/main/config.json"
```

Record the `sha` as `--bench-model-revision`. It is the pin; a moving `main` breaks
reproducibility across rounds.

Compute, from `config.json` (nested under `text_config` on multimodal models):

| Quantity | Formula | Why |
| --- | --- | --- |
| Weights | `parameters x 2` bytes for BF16 | The floor of GPU memory |
| KV per token | `full_attn_layers x num_key_value_heads x head_dim x 2 x 2` bytes | With `full_attention_interval: N`, only every Nth layer has KV |
| KV total | `KV/token x n_prompts x (prompt + max_tokens)` | Peak concurrent |
| Mamba/SSM state | `linear_num_value_heads x value_head_dim x key_head_dim x 4` bytes per linear layer per sequence | Hybrid models only; float32, and it is not small |
| **Vocab** | `vocab_size` | **See the scorer trap in step 5** |

Weights + KV + SSM should leave comfortable headroom. If it needs more than ~75% of
the card, pick a bigger card or fewer concurrent prompts.

Also confirm the pinned vLLM registers the architecture:

```bash
git -C <vllm-clone> grep -l "<ArchitectureName>" -- vllm/model_executor/models/registry.py
```

Registration is necessary, not sufficient. A checkpoint published after the pinned
commit can still fail to load. Only step 5 proves it.

## 2 and 3. Build the engine image and warm the ccache (one operation)

These are the same build. `builder/hermetic.py` mounts the ccache **writable only**
for the trusted empty-patch build (`skip_apply`); miner builds mount it `readonly`
with `CCACHE_READONLY=1`. A miner submission therefore **can never warm the cache**.
Submitting a no-op patch to "warm it up" does not work, and an empty diff is rejected
outright at `gate/surface.py:137` before it ever builds.

On the build VPS, in a **separate clone** so the live worker's checkout is untouched:

```bash
# /opt/pareton is the live deployment. Never check a feature branch out there.
git -C /opt/pareton fetch origin <branch>
git clone -q /opt/pareton /root/pareton-build
cd /root/pareton-build
git fetch -q /opt/pareton "refs/remotes/origin/<branch>:refs/heads/buildbranch"
git checkout -q buildbranch
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

set -a; . /opt/pareton/.env; set +a
echo "$PARETON_GHCR_TOKEN" | docker login ghcr.io -u "$PARETON_GHCR_USERNAME" --password-stdin
BASE=$(docker inspect --format="{{index .RepoDigests 0}}" ghcr.io/pareton-ai/pareton-baseline:v0)

export PARETON_BUILD_MAX_JOBS=4        # <= vCPUs and <= RAM/3GB. cicc peaks 6-12GB/job.
export PARETON_BUILD_TIMEOUT_S=39600   # .env pins 21600 for miners; a cold build exceeds it
setsid nohup .venv/bin/python -m builder \
  --baseline-repo https://github.com/vllm-project/vllm.git \
  --baseline-commit <commit> \
  --base-image "$BASE" \
  --image-ref ghcr.io/pareton-ai/pareton-engine:<new-tag> \
  --empty-patch --torch-cuda-arch-list <9.0|12.0> --push \
  > /root/warm-<arch>.log 2>&1 < /dev/null &
```

`setsid` matters. The build outlives the SSH session; without it a dropped connection
kills an 8-hour compile.

Use a **new tag**, never an existing one. `ops/a2b-build.sh` defaults to
`pareton-engine:baseline` and will overwrite it. Existing campaigns pin digests, but
an overwritten tag makes provenance unreadable later.

`ops/a2b-build.sh` force-checkouts `origin/main` (`git checkout -B main`). If the build
needs an unmerged branch, run `python -m builder` directly as above instead of the script.

### Arch is per box and per campaign

`TORCH_CUDA_ARCH_LIST` is not in the campaign manifest. Miner builds inherit it from
the pinned base image's `Config.Env` (`_base_image_torch_arch`). So the warm build must
use the arch that matches the campaign's GPU, and a box that has only ever built 12.0
has nothing useful for a 9.0 campaign.

`_ccache_mount_id` (`builder/hermetic.py:42`) keys on `baseline_commit` **only**, so two
archs on one commit share a cache dir. They do not corrupt each other (the arch flag is
in every nvcc command line and therefore in the ccache key); it is purely a size
question. Ensure `CCACHE_MAXSIZE` is raised above ccache's 5G default.

### Protect the cache from Docker's GC

BuildKit reclaims idle cache mounts. A 499MB warm ccache was silently deleted this way.
Before building, confirm a floor exists:

```bash
cat /etc/docker/daemon.json   # needs builder.gc.defaultKeepStorage above current usage
docker buildx du | tail -2
```

If absent, write it and restart docker (nothing may be building):

```json
{"builder": {"gc": {"enabled": true, "defaultKeepStorage": "150GB"}}}
```

## 4. Verify the cache is warm (do not infer it)

`docker buildx du --verbose` is ambiguous: records carry the description of whichever
exec last touched them, so a probe build's text can sit on the real record. Reading
`/var/lib/docker/buildkit/.../cachemounts` is also useless, since mounts are only
materialised during a build.

The only honest test is a real miner-path build. Use a trivial but non-empty patch
that creates a file under an allowed path:

```bash
cat > /root/noop.diff <<'EOF'
diff --git a/vllm/pareton_warm_check.py b/vllm/pareton_warm_check.py
new file mode 100644
index 0000000000..e69de29bb2
--- /dev/null
+++ b/vllm/pareton_warm_check.py
@@ -0,0 +1 @@
+# pareton warm-cache verification; no functional change
EOF
```

Then build with `--patch /root/noop.diff --base-image <the new engine digest>` and no
`--push`. A Python-only patch still triggers a full cmake/ninja reconfigure, so it
exercises the cache properly.

**Warm looks like:** compile step ~4min, total ~6min. Cold is hours. If the compile
phase is still running at 20 minutes, the cache is cold; kill it and investigate.

## 5. Dry run on the target GPU

This is the only step that costs money, and the only one that can tell you three
things: whether the model loads, what the thresholds should be, and whether the
harness survives this model's shape.

Build a bench request. **Model pins are not automatic.** `worker/round_job.py:256`
prepends them in a real round; a hand-written request must include them or vLLM
falls back to its built-in default model and tries to download it:

```
--model /model --max-model-len <N> --dtype bfloat16 [--quantization X] <campaign serve_args>
```

Generate the trace with the real sampler so it matches what a round will produce:

```python
from bench.sampler import sample_workload
t = sample_workload(rule=<rule dict>, commit_block=1_000_000,
                    block_hash="0x"+"ab"*32, campaign_id="dryrun-<name>")
```

Leave `min_distinct_ngram_ratio` **out** of the request. It is optional
(`bench/schemas.py`), and when absent nothing fails, yet
`evidence/correctness/*.jsonl` still records `distinct_ngram_ratio` per request.
That is how you measure a bar without first knowing it.

```bash
set -a && . ./.env && set +a
export PARETON_BENCH_HEALTH_TIMEOUT_S=1200
python -m gpu bench --gpu-type <H200> --gpu-count 1 --ttl-hours 3 --keep \
  --request <request>.json --output-dir out/dryrun
```

`--keep` plus a TTL lets you retry on the same pod without re-downloading weights.
**Destroy the pod the moment the measurement is banked** (`python -m gpu destroy <name>`);
do not leave it to the TTL.

### Traps this step exists to catch

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Can't load the configuration of 'Qwen/Qwen3-0.6B'` | No `--model /model`; vLLM used its default | Add the pins above |
| `health check timed out after 600s` | 27B+ spends ~350s in warmup, ~80s in torch.compile | `PARETON_BENCH_HEALTH_TIMEOUT_S=1200` in prod `.env` **and** restart the worker |
| `torch.OutOfMemoryError` in `compute_logprobs` | Scorer builds a `positions x vocab x 4B` logits tensor. A 248k vocab over 6k tokens is 6.2GB, and the scorer inherits the baseline's 0.9 memory fraction | `--gpu-memory-utilization 0.80`. Scorer args come from `scorer_engine_spec` (`bench/main.py:112`) and cannot be tuned separately today |
| `workload trace not found` | Request references a path you moved | Keep request + trace together |

Verify env changes landed in the **running** process, not just on disk:

```bash
tr '\0' '\n' < /proc/$(systemctl show -p MainPID --value pareton-worker)/environ | grep BENCH_HEALTH
```

### Read the thresholds out of the evidence

```python
rows = [json.loads(l) for l in open('out/dryrun/evidence/sla_bench/baseline/rep_1/requests.jsonl')]
# distinct_ngram_ratio over r['text'], plus confirm every r['completion_tokens'] is identical
```

Set each bar below the honest baseline's **minimum**, with margin. If the baseline's
own minimum is near the default bar, the default is wrong for this campaign.

## 6. Seed the row

```bash
python -m campaign.seed --status draft \
  --baseline-repo <repo> --baseline-commit <commit> \
  --base-image-digest "$ENGINE_DIGEST" --baseline-engine-image-digest "$ENGINE_DIGEST" \
  --gpu-skus <TOKEN> --bench-gpu-count 1 \
  --bench-model-repo <org/model> --bench-model-revision <sha> \
  --bench-dtype bfloat16 --bench-max-model-len <N> \
  --bench-serve-args --tensor-parallel-size --bench-serve-args 1 \
  --bench-serve-args --gpu-memory-utilization --bench-serve-args 0.80 \
  --bench-correctness-num-prompts <n> \
  --bench-correctness-min-distinct-ngram-ratio <measured> \
  --sampling-rule-json <rule>.json \
  --emission-start-weight 0.10
```

Seed `draft` first, confirm the row, then re-seed `open`.

### Field traps

- **`gpu_skus` must be a bare provider token.** `H200`, `RTXPRO6000`. The default
  `H200-SXM-141GB` (`campaign/seed.py:49`) matches no provider and silently starves the
  campaign of pods. Confirm against a live row before trusting it.
- **`base_image_digest` and `baseline_engine_image_digest` are both the engine digest**
  on existing campaigns. Not the A2 base.
- **Emission sum.** `campaigns_emission_sum_guard()` caps `SUM(start_weight)` over
  `status='open'` at 1.0. Check what live campaigns already hold.
- **The sampling rule is stored by value**, as `jsonb`. The JSON file is a seed-time
  convenience and is never read again; it does not belong in the repo.
- **`parse_sampling_rule` whitelists keys** and silently drops unknown ones. A new
  field (e.g. `ignore_eos`) needs code in `bench/sampler.py` before it can reach
  `manifest_hash`.
- **Everything in the rule is frozen at seed.** Changing `max_tokens` on a live
  campaign changes `manifest_hash` and invalidates it. Decide before seeding.

## Design note: forcing a fixed output length

Miners change kernels, so outputs diverge in content *and* length even at
`temperature=0`. `bench/score.py` aligns at `min(baseline_tokens, candidate_tokens)`
and zeroes anything under `DEFAULT_SPEED_TOLERANCE = 0.9`, so natural-length
generation injects scoring noise.

Forcing every request to a fixed length removes that entirely. The cost: past its
natural stopping point the model pads and repeats, so the honest baseline looks
degenerate to an absolute repetition bar. Measured on Qwen3.8-27B at 5120 tokens:
distinct-16gram ratio min **0.027**, median **0.192**, against a **0.15** default.
Roughly half the baseline's own outputs would fail their own gate.

An absolute bar low enough to pass the baseline catches nothing, and checking only
the first N tokens is defeated by running the real model for N then emitting garbage.
The workable shape is a **relative** bar: compare a candidate's repetition to the
baseline's on the same prompt in the same round, the way `max_mean_logprob_drop`
already compares quality. Decide this before seeding; it is a manifest field.

## Definition of done

- [ ] Engine image pushed; digest recorded
- [ ] ccache proven warm by a real miner-path build under ~10min
- [ ] Docker GC floor set above current build-cache usage
- [ ] Model confirmed to load and serve on the target GPU
- [ ] Correctness bars measured, not defaulted
- [ ] `PARETON_BENCH_HEALTH_TIMEOUT_S` sized for this model, in prod, worker restarted
- [ ] Pod destroyed
- [ ] Row seeded `draft`, verified, then `open`


