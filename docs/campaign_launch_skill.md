---
name: campaign-launch
description: "Build, verify and launch a pinned vLLM or SGLang campaign, including an open campaign with zero emissions."
version: 2.0.0
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
| `allowed_paths` default | `["vllm/**"]` | `["python/sglang/**"]` |
| Editable install | `pip install --no-deps --no-build-isolation -e .` | `pip install --no-deps --no-build-isolation -e python/` |
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
`--denied-path` options replace their respective defaults. Keep packaging, tests,
Dockerfiles and build configuration denied. The SGLang default also denies
`python/sglang/test/**`; Rust source and external kernel repositories are outside
its allowed surface.

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

The current trusted base compiles the pinned Rust extensions in `/src` and
installs the runtime and CUDA kernel wheels. It then sets
`SGLANG_BUILD_RUST_EXTS=none` so subsequent editable installs retain the baked Rust
binaries. The miner install rebuilds only the `python/` package. It does not
rebuild the separate `sglang-kernel` wheel from the patched AOT sources.

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
| `python/sglang/kernels/jit/**` | CUDA/C++ JIT sources, headers and loaders | Sources pass the gate; custom kernel execution needs a GPU probe |
| `python/sglang/kernels/aot/{csrc,include,python}/**` | Native kernels, FP8 GEMM and `sgl_kernel` bindings | Sources pass the gate, but the installed wheel is not rebuilt |
| `python/sglang/kernels/aot/CMakeLists.txt` | Register additional compiled kernel sources | Denied by `**/CMakeLists.txt` |
| `rust/sglang-radix-tree/**` | Native prefix-cache implementation | Outside the allowlist; its compiled extension is retained |

The existing Python import probe and unchanged-image BF16 evaluation do not
establish full native-kernel patch support. That requires rebuilding and installing
`sglang-kernel` from patched AOT source, permitting the kernel build definitions
needed to register new sources, and staging the pinned CMake dependencies in the
trusted base for offline builds. Miner ccache mounts must remain read-only.
Verify an actual CUDA change by compiling it and checking its GPU output.
Opening Rust source requires an offline Rust rebuild as well.

FlashInfer, CUTLASS, FlashAttention and other dependencies are separately pinned
build inputs. Adding a top-level `3rdparty/**` glob does not expose all their
sources in this SGLang checkout. Custom in-tree replacements can instead be
selected through the kernel dispatch code.

The image also pins `SGLANG_USE_SGL_FA3_KERNEL=1`, selecting FlashAttention-3
from the installed `sglang-kernel` wheel. The community FA3 download has no
Torch 2.13 variant for this pin. Selecting the bundled implementation avoids
that download in the offline evaluation container.

With repository dependencies installed and Docker logged in to GHCR:

```bash
PARETON_BUILD_LOG_DIR="$PWD/out/sglang-build/logs" \
  bash ops/build-sglang-baseline.sh sglang-4c3d47f-<unique-suffix> out/sglang-build
```

This publishes a new build base, builds the empty-patch engine with `--network=none`,
builds a non-empty Python patch through the same miner path and checks that the
patched module imports offline. It writes `image-pins.json` only after those checks
pass. The `Build baseline images` GitHub Actions workflow also accepts
`engine=sglang` and uploads this evidence. GPU validation is a separate step.

The SGLang script publishes both roles under the workflow-writable
`pareton-baseline` package, with `-engine` appended to the serving-image tag.
Use full `ghcr.io/...@sha256:...` references in both campaign image fields.
Bare engine digests default to the separate `pareton-engine` package. To retry
an interrupted build after the trusted base was published, pass that base's
digest reference as the script's third argument or the workflow's
`sglang_build_base` input. The script checks its source-pin label before reuse.

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
build path, but its prebuilt CUDA wheels do not require a vLLM-style CUDA compile.

On the persistent builder, Docker's `builder.gc.enabled` must be `false`.
Pareton's cleanup owns pruning and preserves `exec.cachemount` records. A storage
floor with daemon GC enabled is insufficient. Follow `ops/README.md` for any
maintenance change; preserve the existing image store and other daemon settings.

## 3. Verify on the campaign GPU

Correctness thresholds are pinned policy, not automatically measured constants.
Use explicit values, then verify that an honest baseline passes and that the
harness extracts enough logprobs. Record actual observations separately.

Build a full round request through `worker.round_job.build_round_request`, using
the baseline engine as an unchanged candidate. This ensures the dry run carries
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

SGLang runs two full, untimed warmups before each measured replay set, including
the closing baseline. On the pinned Qwen model, one warmup left a startup stall
in the first measured repetition. Both warmups are saved under `warmup/` and
`warmup_2/` and excluded from scores. vLLM keeps one full warmup.

vLLM scoring uses OpenAI echo logprobs. SGLang scoring uses `/tokenize`,
`/detokenize` and native `/generate` input logprobs. Its OpenAI adapter reports
`-1` offsets and decodes each byte token separately, so emojis can become
replacement characters. The harness verifies the prompt boundary, decoded
continuation and every returned token ID before scoring. Generated clamp tokens
are excluded, and decoded token prefixes still feed the repetition checks.

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

## 4. Open the zero-emission Qwen campaign

The sample targets one H200, BF16 `Qwen/Qwen3.8-27B`, context length 8192, 32
requests and up to 5120 output tokens. The model revision verified on 2026-09-08 is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It contains 27,781,427,952 BF16
parameters, about 51.7 GiB of weights before KV cache and workspace. The workload
pin is in `fixtures/campaigns/sglang_qwen38_27b/sampling_rule.json`.

The completed GPU validation used that BF16 model, not the FP8 checkpoint.
For `Qwen/Qwen3.8-27B-FP8`, the separately verified model revision is
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Its
[model configuration](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json)
declares the Qwen3.5 architecture, BF16 activation dtype and dynamic FP8 E4M3
quantization with 128-by-128 weight blocks. An FP8 campaign should pin this repo
and revision with `bench.model.quantization: "fp8"` and
`bench.model.dtype: "bfloat16"`, then undergo its own full GPU validation.

Sample campaign entries, in addition to the source and image pins:

```json
{
  "status": "open",
  "gpu_skus": ["H200"],
  "allowed_paths": ["python/sglang/**"],
  "engine": {
    "name": "sglang",
    "install_cmd": "pip install --no-deps --no-build-isolation -e python/",
    "entrypoint": ["python3", "-m", "sglang.launch_server"],
    "cache_dir": "/root/.cache/sglang"
  },
  "emission_rule": {
    "name": "linear_decay",
    "start_weight": 0,
    "floor_weight": 0,
    "decay_blocks": 201600
  },
  "bench": {
    "model": {
      "hf_repo": "Qwen/Qwen3.8-27B",
      "hf_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
      "dtype": "bfloat16",
      "quantization": null,
      "max_model_len": 8192
    },
    "gpu_count": 1,
    "serve_args": ["--tp-size", "1", "--mem-fraction-static", "0.80", "--max-running-requests", "32"],
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

The seed command supplies both image fields and signs the completed manifest.
Complete sample fields with the published digest are in
`fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json`. The companion
`image-pins.json` records the trusted base, serving image and successful offline
build run. These files contain no campaign ID; they do not represent a created row.

After successful image and GPU checks, run this once with the published engine ref:

```bash
bash ops/seed-sglang-qwen38-27b.sh ghcr.io/pareton-ai/pareton-baseline@sha256:18cc454de82eebf0fcc3afae9f843645fa1bb2ba64f742cf3b8bba3af5a7ae11
```

It uses `--status open --emission-start-weight 0 --emission-floor-weight 0 --force`.
Both zero weights are required. `--force` creates the new campaign alongside the
existing open campaign. Omit `--no-bench`: submissions must still be evaluated.

A draft cannot accept normal uploads or chain commitments. Zero emissions do not
remove GPU costs or limit submissions to the operator. Seed creates a new row on
every forced invocation; seeding draft and then seeding open creates two rows, so
that sequence does not promote a draft.

Verify the returned ID through `GET /v1/campaigns/<id>`. Check the source and model
revisions, both image digests, engine, patch surface, sampling rule, correctness
bars, status, zero weights and customer signoff. Keep the existing campaign and
its manifest unchanged. Do not claim the new campaign is live until that readback
succeeds and the deployed worker supports its engine request fields.
