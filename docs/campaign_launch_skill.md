---
name: campaign-launch
description: "Build, verify and launch a pinned vLLM or SGLang campaign, including an open campaign with pinned emissions."
version: 3.2.0
category: ops
metadata:
  hermes:
    tags: [pareton, campaign, launch, vllm, sglang, ccache, seed, bittensor, gpu]
  trigger: "User wants to launch a campaign or select a model and GPU."
---

# Pareton campaign launch

A campaign needs a published engine image, a working offline miner build, pinned
model weights, a workload and correctness policy. Verify those before creating the
open row. Never invent image digests, source revisions or GPU measurements.

## 1. Select the engine, model and hardware

Read `GET https://api.pareton.ai/v1/campaigns` and the pinned upstream source.
Fetch the model revision and `config.json` from Hugging Face. For multimodal models,
inspect `text_config` as well as the top-level architecture. Confirm that the pinned
engine implements that architecture. Registration alone does not prove loading.

Estimate BF16 weights as `parameter_count * 2` bytes. Add KV cache, recurrent state,
CUDA graphs and scorer logits. A scorer can materialize `tokens * vocabulary * 4`
bytes of logits. Use a bare provider SKU, such as `H200`, and verify availability
through the configured Lium or Shadeform provider.

These fields select the framework without a database schema change:

| Field | vLLM | SGLang |
| --- | --- | --- |
| `baseline_repo` | `https://github.com/vllm-project/vllm.git` | `https://github.com/sgl-project/sglang.git` |
| Seed option | Omit `--engine`, or use `--engine vllm` | `--engine sglang` |
| `allowed_paths` default | `["vllm/**"]` | `["python/sglang/**", "rust/**"]` |
| Install command | `pip install --no-deps --no-build-isolation -e .` | `/usr/local/bin/pareton-install-sglang` |
| Entrypoint | `python -m vllm.entrypoints.openai.api_server` | `python3 -m sglang.launch_server` |
| Engine cache | `/root/.cache/vllm` | `/root/.cache/sglang` |
| Generated model flag | `--model /model` | `--model-path /model` |
| Generated context flag | `--max-model-len N` | `--context-length N` |

`campaigns.bench.serve_args` is the JSON launch-argument list. There is no separate
`campaigns.serve_args` column. `worker/round_job.py` adds the model, dtype, context
limit and optional quantization before appending that list. Use framework-native
flags and avoid duplicating the generated model or context flags.

Seed chooses the upstream repository and patch paths from the engine. SGLang
requires an explicit `--baseline-commit`. Repeatable `--allowed-path` and
`--denied-path` options replace their respective defaults. Keep packaging, tests
and Dockerfiles denied. The SGLang default permits in-tree CMake registration so
miners can compile new kernel files. It denies `python/sglang/test/**`, AOT tests
and packaged test helpers, `rust/**/tests/**` and `rust/**/benches/**`. External
dependency repositories remain separately pinned inputs.

## 2. Build and publish the baseline

Use an isolated checkout and new image tags. Do not switch the live worker's
checkout to a feature branch. A campaign pins both its builder base and baseline
serving image to the published **engine** digest. Existing campaigns remain tied
to their original digests.

For SGLang commit `4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc`, use
`images/baseline-sglang/Dockerfile`. Its dependency recipe follows the
[pinned pyproject](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/pyproject.toml)
and [upstream Dockerfile](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/docker/Dockerfile):
CUDA 13.0.3, Torch 2.13.0, torchvision 0.28.0 and sglang-kernel 0.4.6.post1.
Do not reuse the older v0.5.17 dependency image for this source commit.

The trusted base stages all seven AOT dependencies at the URL/SHA256 pins in the
upstream CMake files, retains Cargo's dependency cache, and installs the build
tools. Its installer lives outside the patchable `/src` tree. Both the trusted
empty-patch build and miner builds rebuild Python, Rust extensions and the AOT
`sglang-kernel` package with network access disabled. The installed AOT wheel is
replaced with the build from `/src/python/sglang/kernels/aot`.

The trusted baseline carries private Rust/CMake build outputs and an immutable
ccache snapshot. Miner builds read that snapshot even on a fresh validator host;
changed files compile in their own image layer. Shared host cache mounts remain
read-only for miners. `MAX_JOBS`, CMake parallelism and CUDA compiler threads are
bounded. The upstream CMake recipe selects its CUDA architectures; setting
`TORCH_CUDA_ARCH_LIST` alone does not restrict all its targets.

### Kernel source coverage at this pin

`python/sglang/**` permits all file extensions, including CUDA and C++.
At this commit, the former `sgl-kernel/` source tree lives inside
`python/sglang/kernels/aot/`. The upstream
[kernel layout](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/sglang/kernels/README.md)
and [AOT build guide](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/sglang/kernels/aot/README.md)
distinguish these paths:

| Path | Optimization surface | Current recipe |
| --- | --- | --- |
| `python/sglang/srt/**` | Qwen model, FP8 dispatch, scheduling, attention and cache management | Patched Python is installed |
| `python/sglang/kernels/ops/**` | Triton, CuTe, fusions, quantization and backend selection | Patched source is installed; invoked kernels can compile at runtime |
| `python/sglang/kernels/jit/**` | CUDA/C++ JIT sources, headers and loaders | Patched sources compile when the engine dispatches them |
| `python/sglang/kernels/aot/{csrc,include,python}/**` | Native kernels, FP8 GEMM and `sgl_kernel` bindings | Rebuilt and installed from patched source |
| `python/sglang/kernels/aot/{CMakeLists.txt,cmake/**}` | Register additional compiled kernel sources | Allowed; uses staged offline dependency sources |
| `rust/**` | Python extensions, including the native prefix cache | Discovered Python extensions are rebuilt with Cargo offline and locked manifests |

Adding a kernel file does not route model inference through it. Miners must also
wire the operation into the relevant dispatch path. The native probe registers
and runs a new CUDA operator, runs a new kernel through SGLang's JIT loader, and
reads a marker from a rebuilt Rust module. These checks prove compilation and
execution; the full model benchmark separately checks the actual inference path.

FlashInfer, CUTLASS, FlashAttention and other dependencies are separately pinned
build inputs. Adding a top-level `3rdparty/**` glob does not expose all their
sources in this SGLang checkout. Custom in-tree replacements can instead be
selected through the kernel dispatch code.

The other native source roots at this pin are `sgl-model-gateway/` and
`experimental/`. They contain separate routing services and experiments; this
campaign launches `sglang.launch_server` directly. They are outside this campaign's
build and serving path, so broadening those globs would not make their edits run.

The image also pins `SGLANG_USE_SGL_FA3_KERNEL=1`, selecting FlashAttention-3
from the installed `sglang-kernel` wheel. The community FA3 download has no
Torch 2.13 variant for this pin. Selecting the bundled implementation avoids
that download in the offline evaluation container.

Use a dedicated Linux x86_64 CPU build host with Docker and repository Python
dependencies installed, logged in to GHCR. Run in `tmux` so an SSH disconnect does
not stop compilation. The full native build exceeded twelve hours with one
compiler job on an 8-vCPU, 15-GiB validator VPS. The retry ceiling below is not an
estimate of completion time:

```bash
PARETON_BUILD_TIMEOUT_S=172800 PARETON_BUILD_MAX_JOBS=6 \
  PARETON_BUILD_LOG_DIR="$PWD/out/sglang-build/logs" \
  bash ops/build-sglang-baseline.sh sglang-4c3d47f-<unique-suffix> out/sglang-build
```

This publishes a new build base, builds the empty-patch engine with `--network=none`,
then builds a nonempty CUDA/CMake/JIT/Rust patch through the same miner path. It
checks the rebuilt Rust marker and records both ccache hits and misses. It writes
`image-pins.json` with baseline and probe digests only after those checks pass.
The script defaults to six concurrent build jobs and forty-eight hours per build,
respecting explicit environment overrides. It streams Docker output to
the terminal while keeping durable logs. Verbose mode also enables `PIP_VERBOSE=1`
inside the build, exposing backend compiler output instead of just pip's
"still running" messages. It reports trusted ccache statistics before installation
without clearing the cache or resetting counters. It records each published base/engine
reference immediately, even if a later stage fails. Production miner build
deadlines are unchanged. GPU validation is a separate step.

[GitHub-hosted runners have a six-hour job limit](https://docs.github.com/en/actions/reference/limits).
Use the direct VPS build below for the long native compile. No GPU is needed.

The SGLang script publishes both roles under the workflow-writable
`pareton-baseline` package, with `-engine` appended to the serving-image tag.
Use full `ghcr.io/...@sha256:...` references in both campaign image fields.
Bare engine digests default to the separate `pareton-engine` package. To retry
an interrupted build after the trusted base was published, pass that base's
digest reference as the script's third argument or the workflow's
`sglang_build_base` input. The script checks its source and native-build labels
before reuse. The original Python-only image is not a native build base.
Reusing the dependency image skips dependency staging. It does not restore an
unfinished native build from a deleted runner. On the same persistent Docker
builder, completed entries in the trusted ccache mount can be reused on retry.

### Build directly on the validator VPS

Run inside a `tmux` session as the same OS user as the worker. The example uses
the existing `/opt/pareton/.venv` and `/opt/pareton/.env`. This feature branch has
no `requirements.txt` change relative to `main`. A separate detached checkout
provides SGLang support without changing the live service checkout.

```bash
set -e
cd /opt/pareton
source .venv/bin/activate
set -a
source .env
set +a

# Resolve the production lock before changing directories, including overrides.
export PARETON_BUILDER_LOCK_PATH="$(python -c 'import config; print(config.BUILDER_LOCK_PATH)')"
git fetch origin arpan/sglang-campaigns
git worktree add --detach /opt/pareton-sglang-build origin/arpan/sglang-campaigns
cd /opt/pareton-sglang-build

python -m builder.gc_config
printf '%s' "$PARETON_GHCR_TOKEN" | docker login ghcr.io \
  --username "$PARETON_GHCR_USERNAME" --password-stdin

sglang_build_stamp=$(date -u +%Y%m%dT%H%M%SZ)
PARETON_BUILD_TIMEOUT_S=172800 PARETON_BUILD_MAX_JOBS=6 \
PARETON_BUILD_LOG_DIR="/var/log/pareton/sglang-baseline/$sglang_build_stamp" \
python -m builder \
  --engine sglang \
  --baseline-repo https://github.com/sgl-project/sglang.git \
  --baseline-commit 4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc \
  --base-image ghcr.io/pareton-ai/pareton-baseline@sha256:97e1f4e868fc988355f91bb20a6d6f3a9b90c3a901d030730a2646ecbdf00688 \
  --image-ref "ghcr.io/pareton-ai/pareton-baseline:sglang-4c3d47f-vps-$sglang_build_stamp-engine" \
  --empty-patch --push --stream-build-logs
```

Each attempt gets a separate log directory, preserving earlier failure logs.
The GHCR credential needs package read and write access. The final stdout line is
the published serving image's full digest reference; retain it for validation.
The forty-eight-hour timeout applies to compilation; waiting for the shared lock and
clone/push stages have separate limits. This command builds and uploads the
empty-patch engine. Native mutation probes and FP8 GPU validation still follow.

SGLang uses the BuildKit cache mount
`pareton-ccache-4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc`. The vLLM campaign's
different source commit selects a separate cache mount. There is no prune,
ccache clear, daemon restart or production configuration change in these commands.
The GC command validates the existing policy without changing it. The shared lock
waits for any current build and excludes cleanup while this build runs. New vLLM
builds wait for the lock; API, chain and weights services remain running.

For a retry in the existing build worktree, replace `git worktree add` with:

```bash
git -C /opt/pareton-sglang-build switch --detach origin/arpan/sglang-campaigns
```

Keep the same base digest, source commit, compiler flags and Docker builder to
reuse completed cache entries. A canceled `RUN` does not produce a complete image
layer; its Rust/CMake output directories are not a checkpoint. The persistent
ccache mount can supply completed, retained objects, but the next run must show
its actual hit rate before saved time can be claimed. An object still compiling
at cancellation must be rebuilt. Do not assume that `TORCH_CUDA_ARCH_LIST=9.0`
prunes this pinned SGLang recipe: its CMake explicitly emits SM90, SM100 and SM120
code and builds both common-library variants plus attention extensions.

The operator selected six concurrent jobs on the observed 8-vCPU, 15-GiB VPS.
`MAX_JOBS=6` sets `CMAKE_BUILD_PARALLEL_LEVEL=6` and the trusted installer's
`CARGO_BUILD_JOBS=6`. NVCC's internal thread count stays at one per compiler job;
six jobs do not mean six NVCC threads per job. This override applies to the ops
build, leaving production miner defaults unchanged. Monitor compiler progress,
memory and swap use during the run. The source pin, compiler flags, cache IDs and
shared storage lock remain unchanged.

For vLLM, use `images/baseline/Dockerfile`, then run:

```bash
python -m builder \
  --baseline-repo https://github.com/vllm-project/vllm.git \
  --baseline-commit <commit> --base-image <published-build-base-ref> \
  --image-ref ghcr.io/pareton-ai/pareton-engine:<new-tag> \
  --empty-patch --torch-cuda-arch-list 9.0 --push
```

vLLM needs a trusted empty-patch build to warm its CUDA ccache. Miner builds mount
that cache read-only and cannot warm or poison it. Verify warmth with a non-empty
allowed Python patch using `--patch-file <file>`. SGLang uses the same isolated
build path and warms its native cache through the trusted empty-patch build.

On the persistent builder, Docker's `builder.gc.enabled` must be `false`.
Pareton's cleanup owns pruning and preserves `exec.cachemount` records. A storage
floor with daemon GC enabled is insufficient. Follow `ops/README.md` for any
maintenance change; preserve the existing image store and other daemon settings.

## 3. Verify on the campaign GPU

Correctness thresholds are pinned policy, not automatically measured constants.
Use explicit values, then verify that an honest baseline passes and that the
harness extracts enough logprobs. Record actual observations separately.

Build a full round request through `worker.round_job.build_round_request`, using
the baseline engine as an unchanged candidate, or the native mutation probe after
its dedicated GPU checks pass. This ensures the dry run carries
the same launch arguments, engine name, cache path and thresholds as production.
Handwritten SGLang requests must set `name: "sglang"` and
`cache_dir: "/root/.cache/sglang"` on every engine. Omitting the name selects vLLM
for compatibility with older requests. Scorer flags use this name, including when
SGLang omits tensor-parallel arguments or uses another accepted alias.

Generate a trace with `bench.sampler.sample_workload` and the campaign's pinned
sampling rule. Run the same trace against the baseline and candidate. Use all
three stages: streaming replay, shared correctness scoring and baseline drift.
Verify `/v1/models`, streamed token counts, scoring coverage and cleanup.
The model mount is `/model`; do not let the engine fetch a default model.

For an explicit SGLang `--context-length`, the scorer alone reserves seven extra
context slots. At this pin, the scheduler requires input length strictly below
`context_length - 6`, so an output that fills the replay window otherwise cannot
be teacher-forced. An 8192-token campaign therefore starts its scorer with 8199;
baseline, candidate and drift replay keep 8192. The scorer also sets
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` so this allocation can exceed a model's
declared limit without rejecting startup. The forced input still fits the original
window, and the single extra sampled token is excluded from scoring and never fed
back into the model. Keep the worker-generated numeric context pin in the request;
no tokens are truncated and no correctness thresholds are changed.

SGLang runs two full, untimed warmups before each measured replay set, including
the closing baseline. On the pinned Qwen model, one warmup left a startup stall
in the first measured repetition. Both warmups are saved under `warmup/` and
`warmup_2/` and excluded from scores. vLLM keeps one full warmup.

vLLM scoring uses OpenAI echo logprobs. SGLang scoring uses `/tokenize`,
`/detokenize` and native `/generate` input logprobs. Its OpenAI adapter reports
`-1` offsets and decodes each byte token separately, so emojis can become
replacement characters. If tokenizing the combined text merges tokens across the
prompt boundary, the harness preserves the prompt IDs and tokenizes the output
separately without adding special tokens. It verifies the decoded continuation
and every returned token ID before scoring. Generated clamp tokens are excluded,
and decoded token prefixes still feed the repetition checks.

Check `entries[].status` and `entries[].score` in the report as well as its
top-level verdict. A completed harness run can still contain an `infra_failed`
candidate, including a failure of the unchanged reproducibility bar.

```bash
# Load the configured provider credentials without printing them.
set -a
. ./.env
set +a
export PARETON_BENCH_HEALTH_TIMEOUT_S=1200
python -m gpu bench --gpu-type H200 --gpu-count 1 --ttl-hours 3 \
  --request out/sglang-dryrun/bench_request.json --output-dir out/sglang-dryrun/results
```

The provisioner installs `PARETON_GPU_EXTRA_SSH_PUBKEYS` on rented instances.
Those public keys do not supply a persistent build-host address or GHCR credentials.
Keep the existing provider price cap. The command destroys its pod by default;
if retrying with `--keep`, destroy that specific pod when finished.

A 27B model can need a longer health timeout and memory headroom for the shared
scorer. For SGLang, tune `--mem-fraction-static`; vLLM uses
`--gpu-memory-utilization`. Set the required timeout in the running worker's
configuration before opening, using the normal deployment process.

## 4. Open the Qwen campaign

### Initial submission fee

Before seeding, verify the validator has the fee-history schema and code from
PR #126. For an existing Neon database, apply the hand-run migration using the
[fee rollout instructions](campaign-fees.md); deployment alone does not migrate.
The migration's closed/open backfill policy applies only to existing rows.

Choose the initial fee for this campaign explicitly. Pass
`--submission-fee-tao DECIMAL` to `python -m campaign.seed`; the SGLang helper
requires it as its second argument. The CLI option is required, with no default
or environment fallback. Seed rejects fractional RAO and a
configured recipient that differs from the recipient pinned in the miner before
writing the campaign or its profile. It inserts the initial block-zero fee with
the campaign, so an open campaign has the intended fee immediately.

Do not call `campaign.set_fee` after seeding to establish the initial fee. That
command schedules a future change at least 100 blocks ahead and would leave the
old seed fee active in the meantime. Use it for later changes as documented below.
Fee amounts and history are excluded from `manifest_hash`; never re-seed or
rewrite a live campaign's signed terms to change its fee.


The launch helper targets four RTX 5090 GPUs, `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, context length
262144, 32 requests spaced 2 ms apart and up to 5120 output tokens. Sampler
version 3 uses complete conversation prefixes across four groups with eight
requests each. Targets are fixed at 4096, 8192, 16384 and 32768 input tokens,
accepting complete prefixes within 90–100% of each target. All four tiers leave
room for the full output ceiling. This workload covers inputs up to 32K while
retaining the 262144-token model limit. Source preflight must fill every tier
before opening. Thinking is enabled and the failure coefficient is 0.1.
The model revision is
`009632fef96dd349150baa780c984e62e70e91fe`. Its
[model configuration](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead/blob/009632fef96dd349150baa780c984e62e70e91fe/config.json)
declares the Qwen3.5 architecture, BF16 activation dtype and ModelOpt mixed
quantization: NVFP4 for MLP layers, FP8 for attention layers and BF16 for
`lm_head`.
Pin `bench.model.quantization: "modelopt_mixed"` and
`bench.model.dtype: "bfloat16"`. The pinned SGLang source supports this loader;
`fp8` would select the wrong checkpoint format. See the
[RadixArk model card](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead/blob/009632fef96dd349150baa780c984e62e70e91fe/README.md)
for the quantization recipe. The workload pin is in
`fixtures/campaigns/sglang_qwen38_27b/sampling_rule.json`.

The pinned RadixArk tokenizer config has no embedded chat template. Both the
sampler and standalone sample load `chat_template.jinja` from the same model
revision. That template and `tokenizer.json` are byte-identical to the previous
NVIDIA NVFP4 and original Qwen FP8 pins. Sixteen formatter comparisons covering
thinking on/off, conversation history, tool results, Unicode/code and long
inputs produced identical text and token IDs. RadixArk's padding token differs
from Qwen FP8, but the template does not use it and the sampler disables padding.
See the [tokenizer validation record](../fixtures/campaigns/sglang_qwen38_27b/tokenizer-validation.json)
for hashes, inputs and the scope of this CPU check.

The native images and the earlier one-H200, 8192-context FP8 configuration
passed validation on 2026-09-09
with harness commit `bd5aae39fce676d69daee80ee6e3e4e42e663826`. The mutation
candidate passed correctness at full coverage and received score 0.0. Its p99 E2E
relative range was 2.78%, below the unchanged 33.5% reproducibility bar. Baseline
drift was 0.29%, below the unchanged 5% ceiling. Native CUDA/JIT/Rust checks and the
exact 8192-token scorer probe also passed. All validation pods and volumes were
deleted, with provider API readback. See the
[validation record](../fixtures/campaigns/sglang_qwen38_27b/validation-evidence.json)
and [full round report](../fixtures/campaigns/sglang_qwen38_27b/validation/bench_report.json).
Those FP8 checks do not validate the NVFP4 checkpoint or the updated 262K
workload on four RTX 5090 GPUs.
Run source coverage preflight and GPU calibration with the new sampling, thinking and
serving settings before opening. The earlier BF16/Python-only measurements
remain historical evidence in `HANDOFF.md`. See the
[workload proposal](benchmark-scoring-rework-proposal.md) for target ranges and
the Docker settings managed by the harness.

Sample campaign entries, in addition to the source and image pins:

```json
{
  "status": "open",
  "gpu_skus": ["RTX5090"],
  "allowed_paths": ["python/sglang/**", "rust/**"],
  "engine": {
    "name": "sglang",
    "install_cmd": "/usr/local/bin/pareton-install-sglang",
    "entrypoint": ["python3", "-m", "sglang.launch_server"],
    "cache_dir": "/root/.cache/sglang"
  },
  "emission_rule": {
    "name": "linear_decay",
    "start_weight": 0.2,
    "floor_weight": 0,
    "decay_blocks": 201600
  },
  "bench": {
    "model": {
      "hf_repo": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
      "hf_revision": "009632fef96dd349150baa780c984e62e70e91fe",
      "dtype": "bfloat16",
      "quantization": "modelopt_mixed",
      "max_model_len": 262144
    },
    "gpu_count": 4,
    "serve_args": [
      "--trust-remote-code", "--served-model-name", "qwen3.8-27b",
      "--tp-size", "4", "--mem-fraction-static", "0.85",
      "--kv-cache-dtype", "bfloat16",
      "--attention-backend", "flashinfer", "--chunked-prefill-size", "8192",
      "--mamba-radix-cache-strategy", "extra_buffer", "--max-running-requests", "40",
      "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
      "--enable-cache-report"
    ],
    "correctness": {
      "num_prompts": 32,
      "thresholds": {
        "min_mean_logprob": -4,
        "min_token_logprob": -12,
        "min_token_quantile": 0.001,
        "min_coverage_ratio": 0.5,
        "max_mean_logprob_drop": 1.5
      }
    }
  }
}
```

The seed command supplies both image fields, loads `sampling_rule.json` and
`scoring_rule.json`, checks source coverage and signs the completed manifest.
Sample fields are in `fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json`.
The companion `image-pins.json` records the native serving and mutation image
digests validated under the earlier 8K workload. The sample uses the NVFP4 checkpoint, the native installer, Rust/CMake
allowances and the nested AOT test exclusions. It contains no campaign ID and
does not represent a created row. Use its `engine_image` for both campaign image
fields. The dependency build base and mutation image have separate roles.

### Static SSH host

Set these variables in the worker environment:

```ini
PARETON_GPU_PROVIDERS=static_ssh
PARETON_GPU_STATIC_SSH=user@host:port
PARETON_GPU_SSH_KEY_PATH=/path/to/key
```

`host` can be a reachable DNS name or IPv4 address; the optional port defaults
to 22. The current parser does not accept IPv6 literals. The private key must
exist at the configured path on the worker and support noninteractive SSH.
The target needs the campaign's four RTX 5090 GPUs, working NVIDIA drivers and
Docker host access; bootstrap uses root or sudo for host setup.

The VM's provider name and hostname have no Pareton naming requirement.
The orchestrator generates `pt-<UTC timestamp>-<ttl>h-<8 hex digits>` as an
internal run name, without renaming the VM. Static SSH does not register a
managed rental, and its destroy operation is a no-op. The reaper can clean idle
benchmark containers on the static host, but does not delete the VM.
Keep manually managed cloud VMs outside that `pt-...` naming
pattern: the reaper also scans credentialed cloud providers and can delete
expired resources with matching names, even when static SSH is selected.

After successful image and GPU checks, run this once with the published engine ref:

```bash
INITIAL_FEE_TAO=0.15
bash ops/seed-sglang-qwen38-27b.sh "$NATIVE_ENGINE_REF" "$INITIAL_FEE_TAO"
```

It uses `--status open --emission-start-weight 0.20 --emission-floor-weight 0 --force`.
A fresh leader starts at 20% of subnet emissions, declining linearly to the
existing zero floor over 201600 blocks held. `--force` creates the new campaign
alongside the existing open campaign. Omit `--no-bench`: submissions must still
be evaluated.

A draft cannot accept normal uploads or chain commitments. Seed creates a new
row on every forced invocation; seeding draft and then seeding open creates two
rows, so that sequence does not promote a draft.

Verify the returned ID through `GET /v1/campaigns/<id>`. Check the source and model
revisions, both image digests, engine, patch surface, sampling rule, correctness
bars, status, the 20% starting emission rule, initial fee history and customer signoff. Keep the
existing campaign and its manifest unchanged. Do not claim the new campaign is live until that readback
succeeds and the deployed worker supports its engine request fields.


Use the exact campaign UUID printed by seed for fee readback. With the existing
validator environment loaded and its virtualenv active:

```bash
read -r -p 'New campaign UUID from seed output: ' CAMPAIGN_ID
export CAMPAIGN_ID
python - <<'PYTHON'
import os
from campaign.fees import TRUSTED_PAYMENT_RECIPIENT
from campaign.store import get_campaign
campaign = get_campaign(os.environ["CAMPAIGN_ID"])
if campaign is None:
    raise SystemExit("Campaign not found")
print(campaign.submission_fee_history)
assert campaign.submission_fee_history[0]["effective_from_block"] == 0
assert campaign.submission_fee_history[0]["recipient"] == TRUSTED_PAYMENT_RECIPIENT
PYTHON
curl -fsS "https://api.pareton.ai/v1/campaigns/$CAMPAIGN_ID" \
  | jq -e --arg amount "$INITIAL_FEE_TAO" \
    '.submission_fee.amount_tao == $amount and
     .submission_fee.recipient == "5CiieAa5nzSMbw4LPkh2hqv9rfMPZX9ZfEcSjh3SYWNBzk3K"'
```

Use the canonical decimal printed by seed (for example, `0.1500` becomes `0.15`)
for the API comparison. Verify the fee on the campaign page as well. Announce the
fee and tell scripted submitters to add `--yes`, optionally with
`--max-fee-tao 0.15`. Miners do not set fee environment variables.

### Subsequent fee changes

Use `campaign.set_fee` on the validator for an existing campaign, including a
draft that already has an initial fee. Do not run the seed helper again: it
creates another campaign. Choose an activation block at least 100 blocks ahead
at execution and after any scheduled entries. This example leaves 200 blocks:

```bash
ACTIVATION_BLOCK=$(python - <<'PYTHON'
import os
import bittensor as bt
import config
from campaign.store import get_campaign
campaign = get_campaign(os.environ["CAMPAIGN_ID"])
if campaign is None:
    raise SystemExit("Campaign not found")
with bt.Subtensor(network=config.SUBTENSOR_NETWORK) as subtensor:
    head = int(subtensor.block)
last = campaign.submission_fee_history[-1]["effective_from_block"]
print(max(head + 200, last + 1))
PYTHON
)
python -m campaign.set_fee --campaign-id "$CAMPAIGN_ID" \
  --amount-tao 0.20 --effective-from-block "$ACTIVATION_BLOCK"
curl -fsS "https://api.pareton.ai/v1/campaigns/$CAMPAIGN_ID" \
  | jq '{submission_fee, submission_fee_history, submission_fee_at_block}'
```

Run promptly; recalculate if the safety margin expires. Confirm the new entry is
present, then verify `submission_fee` switches at activation. Earlier payments
retain the fee from their payment block. Scheduling preserves existing history,
manifest hashes and signoffs; the API needs Subtensor access once changes exist.
