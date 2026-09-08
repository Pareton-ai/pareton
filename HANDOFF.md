# SGLang campaign support handoff

Prepared on 2026-09-08 from this chat, the working tree, published build evidence,
and saved GPU reports. Repository: `/Users/arpantripathi/Documents/Github/pareton`.
Times below are UTC. Resource and database observations are timestamped historical
checks, not a guarantee about changes made by other operators after those checks.

The subsequent PR review fix for scorer context exhaustion is recorded in section
12. The resumed native/FP8 work and the operator's VPS launch plan are in section
13. Sections 1 through 11 describe the original handoff and its historical image
pins and BF16 GPU measurements; do not use those old pins for the native campaign.

## 1. State at the original handoff

SGLang campaign plumbing, a published baseline image, offline Python patch builds,
and a complete BF16 H200 benchmark round are implemented and verified. The user's
latest requirement also includes Qwen3.8-27B-FP8 and custom native kernels. That
scope is unfinished: the current miner install does not rebuild the AOT kernel
package, and the completed GPU validation used the separate BF16 checkpoint.

**No campaign was created, no production deployment was performed, and the PR is
still a draft.** All three validation pods and their volumes were deleted, with
provider API readback confirming cleanup. No GPU was provisioned for the later
kernel audit or this handoff.

| Item | Verified state |
| --- | --- |
| Branch | `arpan/sglang-campaigns` |
| HEAD before this handoff | `9c06f4426a33f1a0acb45fc07881013bf360b7a9` |
| Remote | `https://github.com/Pareton-ai/pareton.git` |
| Pull request | [PR #148](https://github.com/Pareton-ai/pareton/pull/148), `feat: add pinned SGLang campaigns for Qwen3.8-27B` |
| PR state | Open, draft, unmerged; rechecked while preparing this document |
| Last functionally tested harness | `fdc5b7bc2b083f08ff50994628ea617f6e48b08e` |
| Latest commit | Documentation of native-kernel gaps and the BF16/FP8 distinction |
| Database migration | None; `db/schema.sql` was not changed |
| Campaign/profile inserts | None; seed preparation used mocked insert functions |
| Production worker | Deployment and running version remain unverified |
| Remaining launch blockers | Native build support, FP8 validation, confirmed database target, verified deployed worker |

The implementation commits listed below were pushed. This `HANDOFF.md` is a new
local document and was not part of those commits. Preserve the unrelated,
untracked `docs/production-map.md` and `idempotent.diff`; neither was modified or
included in the implementation PR.

## 2. User request and decisions that remain in force

The original request was to add SGLang throughout campaign seeding, patch gating,
container builds, and evaluation on provisioned Lium/Shadeform GPUs. The requested
source reference is the exact SGLang commit
`4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc`. Keep that pin; do not resolve a newer
branch head just because the original message called it the latest commit.

The original model name was `Qwen/Qwen3.8-27B`. The follow-up explicitly named
`Qwen/Qwen3.8-27B-FP8` and required miners to be able to implement custom kernels
wherever practical in the repository. Treat FP8 and native-kernel support as the
remaining target. The checked-in sample and seed helper still describe the
earlier BF16 configuration and need updating or a clearly separate FP8 sample.

The user also requested an actual campaign with zero allocated emissions:

- Use `status: open`. A draft is rejected by normal upload handling in
  [api/server.py](api/server.py) and skipped by [chain/watcher.py](chain/watcher.py).
- Set both `emission_rule.start_weight` and `emission_rule.floor_weight` to zero.
  The curve in [weights/build.py](weights/build.py) then contributes zero throughout.
- Use `--force` to create another campaign while one is already open. Without it,
  seeding can return the existing open campaign's ID.
- Supply correctness thresholds and keep benchmarking enabled; do not pass
  `--no-bench`.
- This is a normal public open campaign. It accepts other miners' submissions and
  incurs evaluation GPU costs even though its emissions contribution is zero.
- A forced seed creates a new row on each run. It is not an idempotent update or a
  draft-to-open promotion. Inspect the intended database before executing it once.

There was no instruction to modify the existing live campaign or its historical
manifest, events, model pins, images, or emissions.

## 3. Implementation history

These commits follow the starting commit `8c42c0a` (`Create campaign_launch_skill.md`).
Some SGLang scaffolding already existed; this work completed and corrected its
integration rather than introducing every engine abstraction from scratch.

| Commit | Work |
| --- | --- |
| `0645511` | Complete SGLang seeding and evaluation integration, engine-aware arguments, tests and build/launch helpers |
| `6e3f2e2` | Make image smoke checks work on isolated builders |
| `dd8132a` | Select bundled SGLang FA3 kernels for offline evaluation |
| `7a137b4` | Update the campaign launch guide for vLLM and SGLang |
| `8d1ffef` | Publish the serving image under the writable baseline GHCR package |
| `65710f7` | Record verified image digests and sample campaign fields |
| `333a0e6` | Score SGLang output with verified native token IDs to handle Unicode |
| `fdc5b7b` | Add the second untimed SGLang warmup before measured replays |
| `9c06f44` | Document native-kernel build gaps and distinguish FP8 from BF16 validation |

The PR was initially a draft, briefly marked ready to run review/CI, then returned
to draft after the kernel audit. Refresh reviews before completing it; an earlier
check found no inline comments while an automated reviewer was still running.

Architecture decisions were appended under dated 2026-09-08 entries in
[docs/technical-decisions.md](docs/technical-decisions.md). That file is intentionally
gitignored and is available in this local workspace, not through the PR. Linear
was unavailable during the implementation; no Linear issue or decision was
created. The referenced `.cursor/rules/tech-writing.mdc` and
`.cursor/rules/karpathy-guidelines.mdc` files were absent in this checkout.

## 4. What changed in the code

### Campaign metadata and engine selection

[campaign/seed.py](campaign/seed.py) now selects upstream defaults and patch paths
from the engine profile. SGLang requires an explicit source commit, uses the
SGLang repository, and defaults to `allowed_paths: ["python/sglang/**"]`. Its denylist
adds `test/**`, `benchmark/**`, and `python/sglang/test/**` to the existing packaging,
build, test and infrastructure restrictions. Repeatable `--allowed-path` and
`--denied-path` flags replace their respective defaults rather than appending.

Seeded profile metadata now reflects the chosen framework, model and GPU count,
instead of the previous fixed model/TP profile. Implicit legacy vLLM defaults and
manifest/hash behavior are preserved. Existing campaign JSON fields carry the
engine settings, so no schema migration was needed.

[campaign/engine.py](campaign/engine.py) retains the SGLang preset:

```json
{
  "name": "sglang",
  "install_cmd": "pip install --no-deps --no-build-isolation -e python/",
  "entrypoint": ["python3", "-m", "sglang.launch_server"],
  "cache_dir": "/root/.cache/sglang"
}
```

This install command is central to the remaining native-build gap. It installs
the engine Python package but does not rebuild the separate AOT `sglang-kernel`
package from patched sources.

### Worker requests and server arguments

[bench/schemas.py](bench/schemas.py) and [bench/validate.py](bench/validate.py) carry
and validate `EngineSpec.name`, accepting `vllm` and `sglang`; legacy requests
default to `vllm`. The engine cache path is propagated as well.

[worker/round_job.py](worker/round_job.py) generates SGLang's
`--model-path /model --context-length N`, plus dtype and optional quantization,
then appends the campaign's framework-specific arguments. vLLM arguments remain
unchanged. Baseline and candidate engine specs receive the explicit name and cache.

The actual database field is **`campaigns.bench.serve_args`**, a list inside the
`bench` JSON value. There is no standalone `campaigns.serve_args` SQL column.
Do not copy vLLM launch flags into an SGLang campaign or duplicate generated model
and context arguments.

[bench/main.py](bench/main.py) selects scorer flags by the explicit engine name,
instead of inferring SGLang from the presence of `--tp-size`. It omits vLLM-only
scorer flags for SGLang and passes the name into correctness and SLA replay.

### Unicode-safe shared correctness scoring

The first GPU run exposed an incompatibility in SGLang's OpenAI echo-logprob
adapter. It reports `-1` text offsets and decodes individual byte tokens, causing
emoji in prompts to become replacement characters and breaking text alignment.
The pinned tokenizer reproduced the exact discrepancy offline: one captured
prompt/output had 5,929 actual characters but 5,941 characters after joining
separately decoded token pieces. There were 12 extra replacement characters.

[bench/http.py](bench/http.py) factors out a general `post_json` helper while
preserving existing completion behavior and error wording. The SGLang path in
[bench/correctness.py](bench/correctness.py) now:

1. Calls trusted-scorer `/tokenize` for the prompt and prompt-plus-continuation.
   Requires nonempty integer token IDs and an exact prompt-token prefix.
2. Calls `/detokenize` for the prompt, full sequence, and selected trusted prefix
   with `skip_special_tokens: false`, and validates continuation text in context.
3. Permits a single trailing replacement character only for a trusted token
   prefix cut inside a UTF-8 character, with the remaining decoded prefix matching
   the full output. It does not generally waive text validation.
4. Calls native `/generate` with the exact `input_ids`, `return_logprob: true`,
   `logprob_start_len: 0`, `return_text_in_logprobs: true`, temperature zero, and
   one generated clamp token.
5. Validates the full returned input-logprob array and every token ID, including
   prompt tokens. Scores finite continuation logprobs; missing/null values reduce
   actual coverage. The generated clamp token is excluded.
6. Returns a token-aligned decoded trusted prefix for the existing repetition
   checks, preserving those checks and all correctness thresholds.

`score_captured_output` now returns positions, span, and decoded prefix, with
engine and prefix-limit inputs. vLLM keeps the existing OpenAI echo scoring path.
Fifteen offline SGLang scorer regression tests cover the native path in
[tests/test_sglang_correctness.py](tests/test_sglang_correctness.py). An additional
offline audit verified 256 captured output token boundaries.

### Warmup and reproducibility

The second GPU run passed correctness but the identical-image candidate was
`infra_failed`: its p99 E2E relative range was 1.8687 against the unchanged 0.335
bar. Both baseline and candidate had an approximately three-second stall at the
start of the first measured replay after one full warmup; later repetitions and
the closing baseline were stable.

[bench/sla_bench.py](bench/sla_bench.py) now performs two fixed full-trace untimed
warmups for each SGLang baseline, candidate and drift baseline. The second pass
exercises paths reached with a populated prefix cache. Evidence is retained under
`warmup/` and `warmup_2/`; both are excluded from measured metrics, selected outputs
and scores. There are still three measured repetitions. vLLM retains one warmup.
Tests verify warmup counts and exclusion for each engine and role. Neither the
reproducibility bar nor correctness thresholds were relaxed.

### Build, workflow, examples and supporting tests

- [images/baseline-sglang/Dockerfile](images/baseline-sglang/Dockerfile) and its
  [requirements](images/baseline-sglang/requirements-runtime.txt) target the pinned
  source's CUDA/Torch/runtime stack and bake Rust extensions in the trusted base.
- [ops/build-sglang-baseline.sh](ops/build-sglang-baseline.sh) builds/publishes the
  trusted base and serving image, exercises the real offline miner build path with
  a nonempty Python patch, and writes digests after success. It supports reuse of
  an existing trusted base after checking its source-pin label.
- [.github/workflows/build-baseline-images.yml](.github/workflows/build-baseline-images.yml)
  adds manual `engine=sglang` and optional `sglang_build_base` inputs, an isolated
  Linux builder, and evidence upload even after failure. vLLM remains the default.
- [ops/seed-sglang-qwen38-27b.sh](ops/seed-sglang-qwen38-27b.sh) prepares the original
  BF16 open, zero-emission campaign with forced insertion and normal evaluation.
  It has not been run against a real database.
- [fixtures/campaigns/sglang_qwen38_27b](fixtures/campaigns/sglang_qwen38_27b) contains
  full sample fields, image pins and the sampling rule. These are examples, not a
  created campaign record.
- [docs/campaign_launch_skill.md](docs/campaign_launch_skill.md) was updated to
  version 2.0.0 for both frameworks, then corrected to state the kernel and FP8
  limitations. README references were updated too.
- Seed, round-request, schema, CLI, scorer-argument and SLA tests were updated.
  [images/mock-engine/Dockerfile](images/mock-engine/Dockerfile) now separates the
  module entrypoint from default arguments, and lifecycle Docker tests use unique
  container names. These fixes were needed to run the existing suite reliably.

Existing local Docker containers were preserved, including
`pareton-bench-itpub0000000-baseline` and `pareton-bench-it2en0000000-baseline`.
Do not remove them as part of cleanup for this task.

## 5. Published images and build evidence

These are real published references, verified by registry digest checks and an
authenticated GPU pull:

```text
Trusted build base:
ghcr.io/pareton-ai/pareton-baseline@sha256:38c742f147cb8926200e429022b7979b5d359578e6d790c62def91cc2ab78a05

Serving image:
ghcr.io/pareton-ai/pareton-baseline@sha256:18cc454de82eebf0fcc3afae9f843645fa1bb2ba64f742cf3b8bba3af5a7ae11
```

The current campaign sample uses the **serving image reference in both**
`base_image_digest` and `bench.baseline_engine_image_digest`. Miner builds start
from that serving baseline. The distinct trusted build base above is an input to
creating the serving baseline, not the image selected by the current seed helper.

Always retain the full repository-and-digest reference. A bare engine digest can
resolve to the separate `pareton-engine` package and fail. Both SGLang roles were
published under `pareton-baseline`, with an `-engine` serving tag suffix, because
the Actions token could write that package but received `write_package` denial
for the existing `pareton-engine` package. No package ACL was changed.

| Dependency/property | Published image |
| --- | --- |
| Platform | `linux/amd64` |
| OS and CUDA toolkit | Ubuntu 24.04, CUDA 13.0.3 devel |
| Python | 3.12 |
| Torch | `2.13.0+cu130` |
| torchvision / torchaudio | `0.28.0` / `2.11.0` |
| Kernel wheel | `sglang-kernel==0.4.6.post1` |
| FlashInfer | `0.6.18` |
| Transformers / tokenizers | `5.12.1` / `0.22.2` |
| Runtime entrypoint | `python3 -m sglang.launch_server` |
| Dependency record | `/opt/sglang-dependencies.lock` inside the image |
| Trusted build source label | `ai.pareton.sglang.commit` equals the requested SHA |

The recipe pins key dependencies; the published digest fixes the resolved OS and
remaining packages. Rebuilding the Dockerfile later is not a promise of the same
digest, since some build-tool and OS inputs are resolved during the trusted build.

Rust is compiled once at `/src` in the network-enabled trusted build, checked for
extension `.so` files, and retained by subsequent editable installs. The final
environment includes:

```text
SGLANG_BUILD_RUST_EXTS=none
CARGO_NET_OFFLINE=true
SGLANG_USE_SGL_FA3_KERNEL=1
```

The community FA3 download had Torch 2.10/2.11 binaries but no matching Torch 2.13
variant. The pinned source supports using FA3 from `sglang-kernel`; selecting it
avoids that Hugging Face download from the offline evaluation container.

The miner probe adds `python/sglang/pareton_build_probe.py` containing
`PATCH_APPLIED = True`, builds with network disabled and a read-only ccache, then
imports it offline and checks that baked Rust extensions remain. **It does not
compile or execute a modified CUDA/AOT kernel.** The published trusted base was
approximately 8.15 GiB compressed.

| Actions run | Outcome |
| --- | --- |
| [34208344508](https://github.com/Pareton-ai/pareton/actions/runs/34208344508) | Initial attempt cancelled after a builder log-directory issue |
| [34208819955](https://github.com/Pareton-ai/pareton/actions/runs/34208819955) | Rust built; missing community FA3 variant blocked the smoke |
| [34211065540](https://github.com/Pareton-ai/pareton/actions/runs/34211065540) | Trusted base published and offline serving build completed; serving-package push denied |
| [34213253195](https://github.com/Pareton-ai/pareton/actions/runs/34213253195) | Successful publication and offline Python patch verification at `8d1ffef`, reusing the trusted base |

Downloaded successful build evidence is in
`out/sglang-build-run-34213253195/out/sglang-build/`, including
`baseline-build.txt`, `miner-build.txt`, `miner-import.txt`, `probe.diff`, logs and
`image-pins.json`. The concise committed pin record is
[image-pins.json](fixtures/campaigns/sglang_qwen38_27b/image-pins.json).

Native build changes will require new image digests. The current images must not
be relabeled as verified source-built AOT kernel baselines.

## 6. Campaign fields, models and workload

### Existing live campaign observed during this chat

The saved [public campaigns API](https://api.pareton.ai/v1/campaigns) response is
`/tmp/pareton-current-campaigns.json`. It contained one open campaign:

| Field | Observed value |
| --- | --- |
| Campaign ID | `7e0462e4-5806-44ac-9f5b-af0542a4bb86` |
| Engine | `null`, meaning legacy vLLM |
| Source commit | `ee0da84ab9e04ac7610e28580af62c365e898389` |
| Baseline engine digest | `sha256:3dd3b00926e626e6d96f37ad1c4c6feb925d6dd415ee9b7668ead996f5dba1bf` |
| Model | `Qwen/Qwen3.8-27B-FP8` |
| Model revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| GPU | One H200 |
| Emissions | Start `0.1`, floor `0.02`, decay `201600` blocks |

Its vLLM arguments include `--tensor-parallel-size`,
`--gpu-memory-utilization`, prefix caching, chunked prefill, sequence/token limits,
and `--gdn-prefill-backend triton`. They informed the equivalent SGLang settings;
they were not copied literally. No live campaign was modified.

### Original BF16 sample and tested configuration

The full sample is
[campaign-fields.json](fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json).
It does not contain an actual inserted campaign ID, profile ID, manifest hash, or
launch signoff. Its important values are:

| Field | Value |
| --- | --- |
| Source | `https://github.com/sgl-project/sglang.git` at `4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc` |
| Images | Both campaign image fields use the full serving reference in section 5 |
| Model | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Model dtype / quantization | `bfloat16` / `null` |
| GPU and context | `gpu_skus: ["H200"]`, `bench.gpu_count: 1`, context `8192` |
| Additional serve args | `["--tp-size", "1", "--mem-fraction-static", "0.80", "--max-running-requests", "32"]` |
| Correctness prompts | `32` |
| Correctness thresholds | Mean LP `-4`, token LP `-12`, quantile `0.001`, coverage `0.5`, max mean LP drop `1.5` |
| SLA | p99 TTFT `2000` ms, p99 ITL `50` ms |
| Scoring rule | `median_e2e_speedup` |
| Status and emissions | `open`; linear decay with start `0`, floor `0`, decay `201600` |

The sample retains legacy descriptive fields `priority_metric: gpu_hours`,
`success_threshold: >=10% GPU-hour reduction at SLA`, and
`sla.quality_floor_spec: greedy token-match >= 0.99 vs baseline`. Actual execution
uses `scoring_rule` and `bench.correctness`; the descriptive quality text was not
rewritten as an unrelated cleanup. Review those descriptions when preparing the
final FP8 campaign so they accurately explain the selected policy.

The BF16 checkpoint contains approximately 27,781,427,952 BF16 parameters, about
51.7 GiB of weight storage before KV cache, recurrent state, CUDA graphs and scorer
workspace. The completed GPU run proves this checkpoint can load with the tested
image/settings, not that every workload fits the same memory budget.

### FP8 target from the latest user instruction

The verified FP8 revision is distinct from the BF16 revision:

```json
{
  "hf_repo": "Qwen/Qwen3.8-27B-FP8",
  "hf_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
  "dtype": "bfloat16",
  "quantization": "fp8",
  "max_model_len": 8192
}
```

This is a proposed `bench.model` entry requiring its own full GPU validation.
The [pinned model configuration](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json)
declares `Qwen3_5ForConditionalGeneration`, `text_config.model_type: qwen3_5`, BF16
activation dtype, dynamic FP8 E4M3 quantization, and 128-by-128 weight blocks. A
local copy is `/tmp/pareton-sglang-reference/model-fp8-config.json`.

Neither the seed helper nor the sample JSON was converted to FP8 during the
audit. No FP8 model load, FP8 full round, or modified native-kernel run occurred.

### Pinned workload and dry-run request

The [sampling rule](fixtures/campaigns/sglang_qwen38_27b/sampling_rule.json) is:

```json
{
  "algo_version": 2,
  "config": "default",
  "dataset": "nebius/SWE-agent-trajectories",
  "max_tokens": 5120,
  "n_prompts": 32,
  "n_rows": 80036,
  "revision": "68195a1450865274106246d0d0296a1d6807b88e",
  "seed_block_offset": 10,
  "split": "train",
  "type": "hf_rows"
}
```

The smoke used a real sampled HF workload and the pinned model's prompt formatter,
not a synthetic token-count fixture. Its 32 prompts contained 748 to 1,888 tokens,
with a 5,120-token output budget; all fit the 8,192-token context limit.

`out/sglang-launch/bench_request.json` was generated through the actual seed
arguments and `worker.round_job.build_round_request`, with campaign/profile writes
mocked. Baseline and unchanged candidate both used the same serving digest.
The benchmark task ID `c58b8431-98b8-4d4d-b4e3-b330fa467ba4` is **not a campaign ID**.

The final report records these input fingerprints:

```text
Trace SHA:
sha256:bca27c74eec91e7899dc6e11d2718bc9ee48dccaaed2e756ca3734377d22f7bb
Model weights SHA:
sha256:eb0f4fb1867147004a05cd93118a9389220394a3b562d605d828abeededdf2a8
Remote request SHA, shared by the three runs:
sha256:2121c11c852fbbd23472f8e3e6b35f1a88eabfd725baad4f3487be9067149e20
```

## 7. Validation performed

### Local tests and CI

The final functional code at `fdc5b7b` passed **1,133 tests with 41 skipped and two
deprecation warnings in 82.40 seconds**. The log is
`/tmp/pareton-sglang-tests-warmup-final.log`. The warnings concerned the
Starlette/httpx test client and an AnyIO alias. The earlier native-scorer suite
passed 1,127 tests before warmup regression coverage was added.

A temporary Python 3.12 environment at `/tmp/pareton-sglang-test-venv` was used;
the user's `.venv` was not changed. Both database environment variables were
explicitly blank for the suite. No Neon DB e2e tests were run. Some tests bind
local HTTP ports and initially failed under sandbox restrictions; the authorized
rerun outside that restriction passed.

Local Ruff format checking passed for 167 files using Ruff 0.16.6. Relevant lint
checks and `git diff --check` passed. GitHub's formatter used Ruff 0.15.21.

- [Tests run 34220262230](https://github.com/Pareton-ai/pareton/actions/runs/34220262230):
  Python 3.10 and 3.11 passed, finishing around 11:23 UTC.
- [Ruff run 34220262196](https://github.com/Pareton-ai/pareton/actions/runs/34220262196):
  passed around 11:21 UTC.

These are checks of the tested harness commit. The subsequent `9c06f44` change was
documentation only; these results do not validate future native build changes.

### Three H200 attempts

All attempts used Lium, one H200, a $3/hour offer, driver `580.178.04`, and
143,771 MiB reported GPU memory. Aggregate allocated GPU time was approximately
37 minutes. This is elapsed allocation time, not a verified provider invoice.

| Attempt | Allocation and teardown, 2026-09-08 UTC | Result | Evidence directory |
| --- | --- | --- | --- |
| 1 | 10:12:59 to about 10:24:05 | Model loaded and streaming replays completed; Unicode scorer alignment failed | `out/sglang-launch/gpu-results/` |
| 2 | 10:46:41 to about 10:59:15 | Native correctness passed; candidate `infra_failed` on E2E reproducibility | `out/sglang-launch/gpu-results-unicode/` |
| 3 | 11:09:41 to about 11:22:36 | Full BF16 round completed; candidate `scored`, correctness and reproducibility passed | `out/sglang-launch/gpu-results-warmup/` |

Attempt 1 also verified offline GPU imports of Torch, SGLang, and
`sgl_kernel.flash_attn`. A status-poller callback initially lacked `**kwargs`,
causing a status-thread exception without stopping the benchmark; the temporary
runner callback was fixed.

Attempt 2 ran the Unicode fix at `333a0e6`. Correctness passed for 32 prompts and
1,337 positions with coverage `1.0`, mean logprob `-0.05691681365493746`, and baseline
drift `0.018364789501426906`. Its first measured replay had roughly 3.08 seconds
p99 TTFT and 4.34 seconds p99 E2E; later replays were near 90-95 ms TTFT and 1.51
seconds E2E. Candidate E2E relative range `1.8687` exceeded `0.335`, so its score was
null despite CLI exit zero and report-level `verdict: pass`.

Attempt 3 used `fdc5b7bc2b083f08ff50994628ea617f6e48b08e`. The report ran from
11:13:38 to 11:22:33. Cleanup API verification completed at 11:25:04 UTC.

| Final candidate metric | Observed value |
| --- | --- |
| Entry status / reason | `scored` / `null` |
| Score | `0.01652870388394577` |
| Correctness | `pass`, 32 prompts, 1,372 positions |
| Coverage | `1.0` |
| Mean logprob | `-0.05618443367390867` |
| Minimum logprob | `-1.03700590133667` |
| Quantile logprob | `-0.9287145137786865` |
| p99 TTFT relative range | `0.06693795468315601` |
| p99 ITL relative range | `0.1122357667809897` |
| p99 E2E relative range | `0.03205821853844916` |
| Baseline E2E relative range | `0.023056087432088222` |
| Closing baseline E2E relative range | `0.021208072127370026` |
| Baseline drift | `0.004323276662979636`, about 0.43% |

The approximately 1.65% score compares identical images and represents measurement
variation. It is not evidence of an optimization or the sample's stated 10%
improvement target. The startup stall was observed in the second untimed warmup,
outside the measured repetitions.

The report also retains SLA limitations: aggregate baseline/candidate p99 ITL was
approximately 53.543/51.025 ms against the sample's 50 ms target, with
`sla_goodput_ratio` values of `0.125` and `0.5`. A scored pipeline result does not
mean every request satisfied the configured SLA. This was an integration and
correctness/reproducibility validation, not a claim of complete SLA attainment.

Always inspect `entries[].status`, `score`, `correctness`, and `reason`, as well as
the aggregate metrics. A top-level `pass` or exit code zero alone only establishes
that the harness completed its reporting path.

### Resource cleanup evidence

| Attempt | Pod name | Pod ID | Volume ID |
| --- | --- | --- | --- |
| 1 | `pt-20260908101259-3h-ad921feb` | `d9ebb69d-d6b7-4936-962c-5b6cbc876ec3` | `ce1508b4-3b9a-4488-9831-ac1853930a92` |
| 2 | `pt-20260908104641-2.5h-681daf69` | `1ddf3a7e-5249-46ae-8bd4-889a7d7d7ad2` | `8e75bd65-63b0-459e-83dd-aedd35ef016c` |
| 3 | `pt-20260908110941-2h-89d55b6c` | `b7e2e188-b9a1-4550-b674-36b81b0cc509` | `3269d606-1101-433e-9947-a54bab77adc5` |

Deletion of each pod and volume was verified through Lium API readback and local
registry cleanup. The provider-independent request path was exercised on Lium;
there was no real Shadeform GPU validation in this chat.

## 8. Native-kernel audit and required engineering work

The user's concern is valid. The issue is both patch permission and whether the
build/install path actually uses modified source.

At the pinned source commit, `python/sglang/**` is not limited to Python files.
The exact tree contains 405 files with C/C++/CUDA source/header extensions
(`.cu`, `.cuh`, `.cpp`, `.cc`, `.c`, `.h`, `.hpp`); 404 are under `python/sglang/`.
The other is `rust/sglang-radix-tree/torch_2_13_compat.h`. Rust `.rs` sources are
additional native implementation files and are not included in that count.

There is no top-level `sgl-kernel/` directory at this pin; AOT sources live in
`python/sglang/kernels/aot/`. No git submodules were present in the audited tree.
The audit used actual gate behavior, saved in
`out/sglang-launch/kernel-surface-audit.json`.

| Existing path at the pin | Current gate | Actual build/runtime support |
| --- | --- | --- |
| `python/sglang/srt/models/qwen3_5.py` | Allowed | Patched Python model implementation is installed |
| `python/sglang/srt/layers/quantization/fp8.py` | Allowed | FP8 dispatch can be patched; no FP8 GPU round yet |
| `python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py` | Allowed | In-tree kernel/dispatch changes are available when selected |
| `python/sglang/kernels/jit/csrc/gemm/per_token_quant_fp8.cuh` | Allowed | JIT source admitted; a modified CUDA execution probe is still required |
| `python/sglang/kernels/aot/csrc/gemm/fp8_gemm_kernel.cu` | Allowed | Current editable install does not rebuild the installed kernel wheel |
| `python/sglang/kernels/aot/include/sgl_kernel_ops.h` | Allowed | Same AOT rebuild gap |
| `python/sglang/kernels/aot/python/sgl_kernel/gemm.py` | Allowed | Separate AOT bindings are not installed from this patched subtree |
| `python/sglang/kernels/aot/CMakeLists.txt` | Denied | New AOT source registration blocked by `**/CMakeLists.txt` |
| `rust/sglang-radix-tree/src/lib.rs` | Denied | Outside allowlist; baked extension retained |
| `rust/sglang-radix-tree/torch_2_13_compat.h` | Denied | Outside allowlist; baked extension retained |

Some AOT source edits can therefore pass the gate while having no effect on the
running wheel. Simply broadening `allowed_paths` does not fix this, and deny rules
take precedence over allowed globs. Adding an allowed CMake path without revising
the matching deny policy will still reject it.

The upstream [kernel layout](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/sglang/kernels/README.md)
and [AOT build guide](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/sglang/kernels/aot/README.md)
describe adding CUDA source, declaring the operation in `include/sgl_kernel_ops.h`,
registering it in `csrc/common_extension.cc`, adding it to CMake, and exposing a
Python `sgl_kernel` wrapper. Many operation wrappers select AOT implementations by
default; adding a new JIT source file does not automatically route Qwen inference
through it. Dispatch must be tested too.

Required follow-up work:

1. Define the effective campaign patch surface to include in-tree engine code,
   Triton/CUDA JIT, and AOT kernels with the build definitions needed to register
   additional sources. Preserve unrelated test/infra restrictions. Handle the
   current blanket CMake denial explicitly and add observable gate coverage.
2. Extend the trusted build to stage all pinned AOT build dependencies. Extend the
   miner install to rebuild and install `sglang-kernel` from the patched AOT source
   offline, replacing the prebuilt runtime package. The existing `pip -e python/`
   command alone is insufficient.
3. Compile the empty-patch trusted baseline to populate ccache, then keep miner
   cache mounts read-only with `CCACHE_READONLY=1`. Verify compiler cache use and
   invalidation for a changed native file, rather than assuming a cache mount is
   enough. Keep miner builds network-disabled.
4. Prove a nonempty CUDA/AOT patch compiles, installs, and changes the invoked GPU
   operation. Cover a new-source registration case as well as an existing-source
   edit. Validate numerical output and dispatch. Add a real JIT mutation probe for
   the claimed JIT surface; the Python import probe is insufficient.
5. Address the Rust prefix-cache surface if it is included in the promised scope.
   That requires an explicit allowlist and a pinned offline Rust rebuild, including
   available Cargo inputs. Retaining baked `.so` files while allowing Rust edits
   would create the same ineffective-patch problem.
6. Publish fresh digests and run the full FP8 baseline/candidate/scorer/drift round
   using the actual worker request path. Record which native implementations the
   model exercised, separate from standalone kernel probes.
7. Update engine build settings, sample entries, launch helper and guide to match
   what was actually validated. Append a dated local technical decision.

### AOT build details already investigated

The source package is
[python/sglang/kernels/aot/pyproject.toml](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/sglang/kernels/aot/pyproject.toml).
It declares `sglang-kernel` version `0.4.6.post1`, uses scikit-build-core, and
packages `python/sgl_kernel` with the `cp310` stable ABI. Build requirements include
`scikit-build-core>=0.10`, Torch 2.13.0, and wheel. scikit-build-core was not
explicitly added to the current trusted recipe; verify and pin all build tools.

The AOT README calls for CMake >=3.31, while its CMake file states a lower minimum
and uses newer policies. Check the actual installed version instead of relying on
the lower declaration. CUDA compiler threads default to 32; bound jobs and compiler
threads for the real builder's CPU/RAM capacity.

[CMakeLists.txt](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/sglang/kernels/aot/CMakeLists.txt)
uses URL/hash-pinned FetchContent inputs including:

| Dependency | Referenced source |
| --- | --- |
| CUTLASS | `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` |
| fmt | `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28` |
| Triton | `v3.7.1` |
| FlashInfer | `bc29697ba20b7e6bdb728ded98f04788e16ee021` |
| sgl-attn | `f89bc2306632d1ec5f97b014dded4254f5b4a907` |

Use the exact URL hashes in the pinned CMake files when staging them. Additional
FlashMLA dependencies are declared in `cmake/flashmla.cmake`; the table is not a
complete offline dependency inventory.

The CMake build includes SM90 and SM100 common-op targets, FA3, InfLLM, spatial
kernels and FlashMLA. CUDA 13 adds SM100/SM120 architecture flags in its logic.
Setting `TORCH_CUDA_ARCH_LIST=9.0` alone may not prune all targets. Full AOT build
time and peak RAM were not measured. Do not silently rewrite the pinned source
while describing the resulting baseline as an unmodified build of that commit.

JIT tooling searches for FlashInfer-bundled headers/CUTLASS and optional packages
such as DeepGEMM/mathdx. The CUDA-devel image contains compilation tools, but the
availability of every dependency needed by arbitrary new kernels is unverified.
FlashInfer, FlashAttention, CUTLASS and DeepGEMM remain separately pinned inputs.
The top-level `3rdparty/` tree includes AMD material; a broader glob does not make
all external NVIDIA dependency sources part of the SGLang checkout.

No native-builder prototype, path-policy fix, new image, or FP8 run was produced
during the audit. Commit `9c06f44` only corrected documentation.

## 9. Credentials, approvals and launch blockers

### Configuration and prior authorization

Credentials stay in `.env`/the configured secret stores. Do not copy values into
this document, logs, shell transcripts or the PR.

| User input/configuration | Meaning and verified use |
| --- | --- |
| `PARETON_GPU_EXTRA_SSH_PUBKEYS` set in `.env` | User supplied this after the build-host question. One extra key was installed on rented GPUs. It does not establish a persistent build-host or production SSH connection. |
| `PARETON_GHCR_USERNAME`, `PARETON_GHCR_TOKEN` configured; user answered "done" | Private `pareton-baseline` images require authenticated pulls; the GPU pull then succeeded. Actions publication used its workflow token. |
| "Approve this validation run" | Explicit authorization for H200 validation up to $3/hour for at most three hours, with automatic teardown. |

The validation approval allowed temporary overrides of the local $2.50/hour cap
and 15-minute TTL. The three recorded runs consumed roughly 37 aggregate GPU
minutes; all were torn down. The approval was not a permanent `.env` change or an
unbounded recurring allocation. Preserve its cost/duration limits when considering
further validation; do not repeatedly ask for already granted actions within scope.

No automatic approval-review rejection occurred during this work.

### Unresolved database target

Read-only checks found **zero campaigns** in the database configured by local
`PARETON_DATABASE_URL`, while the public API returned the open campaign in section
6. The live campaign ID was absent locally. No inherited environment override
explained the mismatch. The last recorded read was around 10:58 UTC on 2026-09-08.

The following clarification remains unanswered:

> Where should I create the open, zero-emission campaign? `PARETON_DATABASE_URL`
> still points to an empty database, while `api.pareton.ai` has an existing live
> campaign.

The choice presented was the production database or the currently configured empty
database. No answer was received. Keep insertion pending until the target is
unambiguous; the original instruction to create a campaign does not identify which
of these conflicting databases is intended.

The dry-run preparation mocked `list_campaigns`, `insert_profile` and
`insert_campaign`. It did not write a real profile/campaign or bypass this blocker.
`out/sglang-launch/seed-command.txt` is a prepared command, not proof of execution.

### Unverified production deployment

An earlier root SSH attempt to the API host failed public-key authentication. The
local untracked [production map](docs/production-map.md) mentions alias
`pareton-vps` and host `pareton-prod-02`, but `ssh -G pareton-vps` showed no working
configured alias. A persistent production SSH connection was not established.

The repository deployment flow in [ops/deploy.sh](ops/deploy.sh) and
[ops/README.md](ops/README.md) polls main approximately every 60 seconds. API and
watcher are restarted, while worker restart is deferred if a submission job is
running. Code present on disk does not prove the running worker loaded it. The
local production map also records a historical difference between the installed
deploy script and the repository version; verify the real host behavior.

Before opening an SGLang campaign, deploy and verify the actual worker/harness
with explicit engine-name propagation, correct SGLang context flags, native
Unicode-safe scoring, two warmups, and the completed native build changes. Check
the worker's startup timeout and private-image pull configuration too. The tested
GPU path used a 1,200-second health timeout. No worker git-SHA endpoint was verified;
an API health response alone does not establish its running revision.

The PR has not been merged. No production service was restarted, no existing
campaign was reseeded, and no production schema was changed in this chat.

## 10. Artifact map and resumption commands

Paths beginning with `out/` are gitignored local evidence. `/tmp/` paths are
temporary and can disappear. They do not travel with the PR. Preserve needed
evidence in the project's approved artifact storage before deleting the workspace;
do not copy `.env`, SSH material or unrelated files with it.

| Artifact | Purpose |
| --- | --- |
| `fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json` | Committed BF16 sample fields, still needing FP8/native-scope follow-up |
| `fixtures/campaigns/sglang_qwen38_27b/image-pins.json` | Published build/serving references and successful build run |
| `fixtures/campaigns/sglang_qwen38_27b/sampling_rule.json` | Pinned real workload recipe |
| `out/sglang-launch/validation.json` | Compact test/image/GPU history and cleanup results |
| `out/sglang-launch/kernel-surface-audit.json` | Native source counts and actual gate results |
| `out/sglang-launch/bench_request.json` | Prepared BF16 request from the real worker path |
| `out/sglang-launch/workload_trace.json` | The sampled trace used on the GPU |
| `out/sglang-launch/campaign-fields.json`, `seed-command.txt` | Dry-run preparation outputs; no DB insertion |
| `out/sglang-launch/gpu-results*/bench_report.json` | All three reports, including both failure cases |
| `out/sglang-launch/gpu-run.log`, `gpu-run-unicode.log`, `gpu-run-warmup.log` | Provisioning, execution and teardown logs |
| `out/sglang-launch/gpu-results-warmup/evidence/` | Final replay, correctness, environment and model-weight evidence |
| `out/sglang-build-run-34213253195/out/sglang-build/` | Downloaded image build/probe evidence |
| `/tmp/pareton-sglang-tests-warmup-final.log` | Final full local test result |
| `/tmp/pareton-sglang-tests-native-final.log` | Earlier scorer-fix test result |
| `/tmp/pareton-sglang-test-venv/` | Disposable Python 3.12 test environment |
| `/tmp/pareton-sglang-reference/` | Pinned source tree metadata and selected upstream/model files |
| `/tmp/pareton-sglang-smoke/` | Temporary request preparation, GPU runner and status helpers |
| `/tmp/pareton-sglang-hf-cache/` | Downloaded tokenizer/dataset cache |
| `/tmp/pareton-sglang-validation-repo/` | Isolated clean clone used for remote harness upload; last used at `fdc5b7b` |
| `/tmp/pareton-current-campaigns.json` | Historical public API snapshot |
| `/tmp/pareton-sglang-pr-body.md`, `/tmp/pareton-sglang-pr-update.json` | Prepared PR description and API update payload |

`validation.json` reports `gpu_validation: passed` and `offline_miner_patch: passed`
for the historical BF16/Python-probe configuration. Its campaign blocker string
predates the native-kernel audit. Read it together with this handoff; it does not
claim FP8 or modified AOT validation.

Engine logs are stored under each run's `evidence/correctness/engine_logs/`,
including logs for SLA baseline/candidate roles. Use targeted `rg --files
--no-ignore` when locating ignored evidence.

### Safe inspection and tests

Run from the repository root:

```bash
git status --short
git log -10 --oneline
gh api repos/Pareton-ai/pareton/pulls/148 \
  --jq '{url:.html_url,title,state,draft,merged,head:.head.sha}'
```

```bash
PARETON_DATABASE_URL='' PARETON_TEST_DATABASE_URL='' \
  /tmp/pareton-sglang-test-venv/bin/python -m pytest tests -q
```

If the temporary environment is gone, create a new disposable environment from
repository requirements. DB e2e must use only `PARETON_TEST_DATABASE_URL` for the
Neon test branch, never `PARETON_DATABASE_URL`. No schema reset is required by the
current change because the schema was not edited.

Read the final result without dumping large timing arrays:

```bash
python3 - <<'PY'
import json
from pathlib import Path

r = json.loads(Path('out/sglang-launch/gpu-results-warmup/bench_report.json').read_text())
print('report verdict:', r['verdict'])
print('baseline drift:', r['baseline_drift'])
for entry in r['entries']:
    print(json.dumps({
        'status': entry['status'],
        'score': entry['score'],
        'reason': entry['reason'],
        'correctness': entry.get('correctness'),
        'metrics': entry.get('sla', {}).get('metrics'),
        'variance': entry.get('sla', {}).get('cross_rep_variance'),
    }, indent=2))
PY
```

### Build and helper cautions

The current publishing script accepts:

```text
bash ops/build-sglang-baseline.sh UNIQUE_TAG_SUFFIX OUTPUT_DIR [PUBLISHED_BUILD_BASE_REF]
```

Run it on an isolated Linux/amd64 builder with GHCR write authorization and
repository dependencies. It publishes images. It currently verifies only the
Python patch path; extend it before using it as native-kernel signoff. Use a fresh
tag suffix and evidence directory for each new build. Reuse an old trusted base
only when it contains the prerequisites for the intended build.

`/tmp/pareton-sglang-smoke/prepare-request.py` uses the actual seed helper arguments
with mocked DB functions. It writes request/trace outputs and generates a task ID;
rerunning it can overwrite the historical evidence. Parameterize fresh output
paths when adapting it to FP8.

`/tmp/pareton-sglang-smoke/run-gpu.py` currently points to the historical BF16 request
and `gpu-results-warmup` output directory. It loads configured credentials without
printing them, uploads the isolated checkout, uses a bounded H200 allocation and
teardown path, and had a 1,200-second health timeout, 100-minute benchmark timeout
and 115-minute alarm for its last attempt. **Do not blindly rerun it:** update the
request, source revision, output directory and allocation limits deliberately.
`status-gpu.py` still targets the last deleted pod.

The isolated validation clone avoided uploading the unrelated local diff. Update
that clone to the intended tested revision before any future remote run; do not
assume it follows the main workspace automatically.

The current seed helper accepts one full engine digest reference:

```text
bash ops/seed-sglang-qwen38-27b.sh PUBLISHED_ENGINE_DIGEST_REF
```

It creates a public open BF16 campaign with `--force`. It is not the final FP8
command and is not a dry run. Update its model/build policy and resolve the
database/deployment blockers before executing a real seed.

The installed `gh pr edit` failed because it queried the deprecated Projects
classic GraphQL `projectCards` field. The PR body was updated successfully by
writing exact JSON to a temporary file and using:

```text
gh api --method PATCH repos/Pareton-ai/pareton/pulls/148 --input PREPARED_JSON_FILE
```

`gh pr ready 148 --undo` successfully returned it to draft. Inspect current state
before changing review status. No need to repeat those mutations just to read it.

## 11. Ordered path to completion

1. Resume from this branch and review the native audit against the pinned source.
   Keep the passing scorer/warmup changes and preserve unrelated work.
2. Implement effective AOT build/install and scoped path-policy support. Address
   Rust explicitly if it is included in the promised custom-native surface.
   Prefetch dependencies in the trusted base and preserve offline miner builds and
   read-only ccache behavior.
3. Add meaningful native mutation and gate/build coverage, publish new images, and
   verify both empty-patch and nonempty-patch builds. Record exact image references.
4. Prepare the FP8 sample and actual worker-generated request using revision
   `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Verify a real custom kernel on H200
   and complete the full unchanged-candidate FP8 round with correctness, measured
   replays and drift. Inspect entry status and SLA metrics; retain all evidence.
5. Run the relevant local suite and CI for the final implementation, review the
   substantial build/scoring changes, and update the PR description around the
   final validated scope. Keep historical BF16 results labeled accurately.
6. Resolve the outstanding database-target ambiguity. Deploy through the normal
   process and verify the running worker/harness, registry credentials and startup
   timeout before opening a campaign.
7. Prepare one concrete seed command with the final source/model/image pins,
   effective patch policy, correctness thresholds, `--status open`, both emission
   weights zero, normal benchmarking, and `--force`. Inspect for an already-created
   equivalent campaign, then execute once against the confirmed target.
8. Read back the new row and public API entry. Record the real campaign ID,
   profile/manifest information, full image references, model revision, engine,
   allowed/denied paths, serve args, `open` status, enabled evaluation, and both
   zero emission weights. Confirm the existing live campaign remains unchanged.
9. Verify the deployed submission/evaluation path for the new campaign and retain
   actual results. Verify teardown of every validation resource. Report any
   remaining limitation rather than treating publication or row insertion alone
   as end-to-end completion.

Completion requires all three outstanding technical/operational outcomes:
effective native modifications, verified FP8 evaluation, and the requested open
zero-emission campaign on the intended deployed system. None should be inferred
from the successful historical BF16 smoke alone.

## 12. PR review follow-up: scorer context exhaustion

The user reported that the SGLang scorer teacher-forces the full captured sequence
and requests one extra token, so an output filling the 8192-token replay context
can make correctness fail and void a round when the relative quality bar is enabled.

The pinned source confirms two limits. `TokenizerManager._validate_one_request`
rejects an input equal to the context length, even with `max_new_tokens: 0`.
`TpModelWorker.get_worker_info` derives `max_req_input_len` as context length minus
six when the memory pool is sufficient, and `validate_input_length` requires input
length strictly below that limit. Zero new tokens or one extra context slot alone
therefore does not fix the complete boundary case.

The fix in `bench/main.py:scorer_engine_spec` reserves seven additional context
slots for explicit numeric SGLang context arguments, handling both separate and
equals-form arguments. For this campaign the scorer uses 8199, while baseline,
candidate and drift replay remain at 8192. The scorer sets
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` to allow the allocation even when the
campaign uses the model's full declared context. Forced input positions still fit
the original replay window; the sampled clamp token is excluded and is never fed
back into the model. The existing native input-logprob path and all full-token,
Unicode, coverage and quality checks remain intact. No output is truncated.

Tests cover separate/equals/duplicate context flags, isolation of the scorer's
arguments and environment, and the real worker-to-round-plan path. The new
correctness regression supplies 3072 prompt tokens plus 5120 continuation tokens,
reproduces the old length failure, then verifies that baseline and candidate both
pass with `max_mean_logprob_drop: 1.5`, 5120 scored positions and coverage 1.0. A
distinct final-token logprob ensures the last captured token affects the result.

The actual pinned tokenizer length checks, worker limit derivation and scheduler
input validator were also executed with CPU-only stand-ins. An 8192-token input
was rejected at contexts 8192, 8193 and 8198, and accepted at 8199. Evidence is in
`out/sglang-launch/context-scorer-fix/pinned-length-checks.json`. This is execution
of the source's validation logic, not GPU inference. No new image was published,
GPU rented, campaign seeded or production service deployed for this review fix.

The full suite passed 1137 tests with 41 skipped and two deprecation warnings in
82.94 seconds. Its log is `/tmp/pareton-sglang-tests-context-final.log`; relevant
Ruff lint/format checks and `git diff --check` passed too. The launch guide and
local ignored technical-decisions file were updated with the headroom requirement.
Continue to use a numeric context pin generated from the campaign's model settings;
omitted context arguments do not provide a pinned value from which to derive
headroom. The earlier H200 results have not been rerun with this scorer setting.

## 13. Resumed native/FP8 completion and VPS launch

The operator now has access to the validator VPS and requests the minimal Linux
commands to create the campaign there after the technical work is complete. Use
`/opt/pareton` and `/opt/pareton/.env` on that host. This resolves the earlier local
database-target ambiguity. Prepare commands for the operator; do not seed from the
local laptop or switch the production checkout to this feature branch.

### Code completed in the resumed work

- `dd8c287`: new SGLang profiles use `/usr/local/bin/pareton-install-sglang`, a
  trusted installer outside the patchable tree. It rebuilds Python, all discovered
  Rust Python extensions and the AOT `sglang-kernel` package from patched source.
  Both empty-patch and miner builds now use `--network=none` for SGLang.
- `images/baseline-sglang/prepare-deps.py` stages and verifies all seven upstream
  URL/SHA256-pinned CMake dependencies, including FlashMLA's nested CUTLASS tree.
  It writes disconnected CMake source overrides and dependency receipts.
- The new image stage reuses the previously published dependency bootstrap by
  digest, adds scikit-build-core 0.11.6 and CMake 3.31.10, and reenables offline
  Rust builds. The upstream source commit remains unchanged.
- Trusted builds retain private Rust/CMake outputs and include a ccache snapshot
  in the immutable serving image. Miner builds read that snapshot on fresh hosts;
  cache misses compile in their private image layer. Shared host cache mounts
  remain read-only for miners.
- SGLang defaults now allow `python/sglang/**` and `rust/**`. In-tree CMake files
  can register new compiled sources. Packaging, Dockerfiles and tests remain
  denied; Rust test/bench subdirectories are additionally denied.
- `ops/make-sglang-native-probe.py` generates a patch that changes AOT CMake and
  `common_extension.cc`, adds a CUDA add-seven operator, adds an SGLang JIT
  add-eleven kernel, and adds a marker to the compiled Rust radix-tree module.
  The publisher builds this through the miner path and publishes a separate probe
  image for GPU verification. Its receipts require ccache hits and misses.
- `67e56be`: the seed helper now pins `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, BF16 activation dtype and
  `quantization: fp8`. It uses the full native path defaults, an open status,
  both zero emission weights, normal correctness/evaluation and `--force`.
- `ops/validate_sglang_gpu.py` runs the native mutation probes, stages pinned
  model weights, starts the scorer derived by `scorer_engine_spec`, and sends
  exactly 8192 forced IDs with one clamp token. It requires all 8192 returned
  IDs, finite logprobs after the unscored first position, and one output token.
  This is followed by a separate normal full FP8 round.

### Evidence collected before the new GPU run

The full offline suite passed **1139 tests, 41 skipped**, with two existing
Starlette/AnyIO warnings, in 88.01 seconds. Log:
`/tmp/pareton-sglang-native-tests.log`. It ran with both database URLs empty.
The first sandboxed attempt could not bind mock HTTP ports; the successful run
had localhost permissions. No DB e2e or GPU tests were represented as unit tests.
Ruff formatting, targeted lint for new scripts, shell syntax and `git diff --check`
passed. PR CI remains skipped while the PR is a draft.

The actual native probe patch passes `gate.surface.check_surface` with the new
seed defaults and `git apply --check` against the exact pinned sources. Receipts:
`out/sglang-native-fp8/native-gate-check.json` and `probe-source-check.diff`.
These checks do not establish that the native binaries run; the GPU probes do.

The FP8 checkpoint contains 30,866,866,928 bytes of safetensors weights, versus
55,563,006,776 for the BF16 checkpoint. The pinned tokenizer JSON, tokenizer
configuration and chat template match between both repositories. The same audited
workload trace can be reused. Hugging Face metadata receipts are in
`out/sglang-native-fp8/model-pins.json`.

A public API read on 2026-09-08 around 14:53 UTC still showed one open campaign:
`7e0462e4-5806-44ac-9f5b-af0542a4bb86`, vLLM, `Qwen/Qwen3.8-27B-FP8`, emissions
start 0.1 and floor 0.02. Snapshot:
`out/sglang-native-fp8/public-campaigns-before.json`. No rows were changed.

### Native image build in progress

[Native build run 34240317476](https://github.com/Pareton-ai/pareton/actions/runs/34240317476)
runs the recipe at `dd8c287` on an isolated GitHub Actions Linux/amd64 host.
The host reported 15 GiB total RAM, 14 GiB available and 106 GB free disk.
Build jobs and compiler threads are bounded at one; workflow timeout is six hours.

The trusted dependency build base published successfully:

```text
ghcr.io/pareton-ai/pareton-baseline@sha256:97e1f4e868fc988355f91bb20a6d6f3a9b90c3a901d030730a2646ecbdf00688
```

**This is a build base, not a validated serving image.** At this checkpoint, the
empty-patch native compilation is still running. The serving/probe image pins,
ccache results and real CUDA/JIT/Rust execution results are not available yet.
No new GPU has been rented during the resumed work.

### Prepared GPU continuation

The clean validation clone is at `67e56be` in
`/tmp/pareton-sglang-validation-repo`. The new helper scripts are:

- `/tmp/pareton-sglang-native/prepare-request.py`: captures the actual FP8 seed
  helper arguments with mocked database inserts and writes the actual worker
  request and sample fields under `out/sglang-native-fp8/`.
- `/tmp/pareton-sglang-native/run-gpu.py`: extends image pull in the local driver
  to run native/context probes before invoking the normal remote round harness.
  It writes separate probe and round evidence and retains default teardown.

The prepared allocation is one Lium H200 at at most $3/hour, TTL two hours,
115-minute local deadline and 70-minute full-round timeout. Together with the
previous roughly 37 GPU minutes, that remains within the operator's existing
three-hour validation approval. No persistent production budget settings are
changed. Run only once the serving and probe images have published and their
build checks passed, and confirm provider cleanup afterwards.

Still required before supplying a launch-ready pin: finish the native build,
inspect its receipts, run native/context probes and a full FP8 round, update the
sample pins and launch guide, and run final PR CI/review. Then provide the operator
with the VPS seed and API readback commands. Production deployment and the actual
campaign insertion/readback will be performed on the operator's VPS.
