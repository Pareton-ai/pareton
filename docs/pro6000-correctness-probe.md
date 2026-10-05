# Optional RTX PRO 6000 LongWriter correctness diagnostic (PAR-144)

`ops/pro6000-correctness-probe.py` is an opt-in operator tool. It can be run from
its branch without merging it. It changes no campaign, database, production
harness, scoring rule, launcher, or pinned memory fraction. Neither the campaign
launcher nor the concurrency/scoring work depends on merging this diagnostic.
This is partial support for PAR-144, not completion of its GPU acceptance criteria.

The tool targets Qwen3.8-27B-FP8 on one dedicated RTX PRO 6000 with TP1/PP1. It
runs the actual round harness with the pinned baseline image as its own candidate:

1. Generate a CPU LongWriter preview with the pinned tokenizer/template, retaining
   the source trace and sampling receipt. An existing preview may be supplied.
2. Select the longest distinct rendered inputs up to 16,384 tokens. If the 16k
   tier is scarce, fill the remaining slots with the longest available shorter
   inputs, including inputs between the usual tier bands. Record every fallback;
   never pad, truncate, or duplicate distinct-mode inputs. Retokenize and check
   input hashes before starting a GPU.
3. Run baseline, drift baseline, and identical-image candidate containers with
   the usual cold/cached-prefix warmups and three measured repetitions.
4. Stop generation containers, start the trusted scorer with the production
   scorer specification (including seven tokens of SGLang context headroom), and
   grade baseline and candidate outputs three times in the same scorer process.
   `--scorer-repetitions` can extend this repeated-load test.
5. Preserve failures, logs, runtime configuration, Docker state, likelihood and
   coverage results, scoring-span evidence, and sampled GPU memory.

All hooks are local to the diagnostic process. Duplicate context, memory, TP and
PP scalar aliases are resolved to their last value before launch; repeatable
unrelated flags are preserved. Both original and resolved arguments are saved.
Requested scalar settings must be confirmed by `/server_info` (top level or
`server_args`), otherwise the run fails instead of assuming an effective value.
The scorer never coexists with the generation containers in this harness.

## Inputs and scope

Supply a campaign-fields JSON with the finished engine digest, exact model
revision, serving arguments, correctness thresholds, and LongWriter v4 sampling
rule. The TP4 NVFP4 fixture is rejected. A matching TP1 FP8 fixture can be exported
from [draft PR #188](https://github.com/Pareton-ai/pareton/pull/188) without merging
that PR, or supplied independently:

```bash
git fetch origin arpan/pro6000-fp8-campaign
git show FETCH_HEAD:fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json \
  > /var/tmp/pro6000-campaign-fields.json
```

This tool is written against the current v4 harness. It does not implement or
qualify PR #187's v5/C4 scheduling. Rebase and validate a separate v5 adaptation
when that contract is available. The diagnostic's longest-tier trace intentionally
is not represented as a valid balanced four-tier campaign trace or qualification
receipt. It still uses the production tokenizer/capacity preflight for each
running generation engine.

The LongWriter envelope is at most 16,384 input + 5,120 generated tokens. The 16k
input tier permits 90–100% of that input bound, including template/history.
Fallback inputs can be below that band. The summary records actual input lengths
and every fallback, which cannot establish the maximum-input boundary. A successful capacity run claims an exact
21,504-token boundary exercise only if every selected input is exactly 16,384
and the captured/scored continuations reach 5,120. It never establishes 262,144
context support. Natural EOS runs are correctness evidence at observed lengths;
forced-length runs are capacity diagnostics, not output-quality claims.

## Run on the target host

Use a Linux host exposing exactly one RTX PRO 6000, with Docker, the NVIDIA
container runtime, `nvidia-smi`, and the repository Python requirements installed.
The tool takes the normal `/opt/pareton/.static-host.lock` and refuses active GPU
compute processes. It never kills other workloads or prunes caches/images. Have
the digest-pinned engine available and access to the pinned model/dataset assets;
the normal harness stages weights and records their hashes. The same engine image
can be reused across GPU hosts, but only this target run provides memory evidence.

From the repository root, choose experimental memory fractions. The values below
are **test candidates**, not measured or recommended production settings:

```bash
PRO6000_FIELDS=/var/tmp/pro6000-campaign-fields.json
PRO6000_GENERATION_FRACTION=0.80
PRO6000_SCORER_FRACTION=0.60
PRO6000_RUN_ROOT=$(mktemp -d /var/tmp/pro6000-correctness-XXXXXX)
PYTHONPATH=. python ops/pro6000-correctness-probe.py \
  --campaign-fields "$PRO6000_FIELDS" \
  --generation-memory-fraction "$PRO6000_GENERATION_FRACTION" \
  --scorer-memory-fraction "$PRO6000_SCORER_FRACTION" \
  --prompt-count 32 --prefixes distinct --case natural \
  --output-dir "$PRO6000_RUN_ROOT/prepared" --prepare-only
```

Preparation fetches source/tokenizer assets but starts no engine, downloads no
model weights, and performs no GPU work. It scans the pinned corpus without
balanced-tier quotas and selects the 32 longest eligible distinct inputs at or
below 16,384 tokens. A shortage of 16k inputs is filled by shorter inputs, not
by reducing the requested prompt count. Fewer than 32 distinct eligible inputs
in total still fails. The summary lists `shorter_fallback_count` and each
`shorter_fallbacks` entry with its source ID and actual length. Review the input lengths,
model/image pins, thresholds, and generated request before proceeding.

Run a natural-EOS control and a separately labelled forced-capacity test with the
same source inputs. These can be expensive: each full round includes three engine
starts, full warmups, repeated long generation, and a separate scorer start.
Each output directory must be new.

```bash
for PRO6000_CASE in natural capacity; do
  PYTHONPATH=. python ops/pro6000-correctness-probe.py \
    --campaign-fields "$PRO6000_FIELDS" \
    --docker-model-volume \
    --source-preview "$PRO6000_RUN_ROOT/prepared/source_preview" \
    --generation-memory-fraction "$PRO6000_GENERATION_FRACTION" \
    --scorer-memory-fraction "$PRO6000_SCORER_FRACTION" \
    --prompt-count 32 --prefixes distinct --case "$PRO6000_CASE" \
    --scorer-repetitions 3 --request-timeout 1800 \
    --output-dir "$PRO6000_RUN_ROOT/$PRO6000_CASE"
done
```

Capacity sets `ignore_eos=true` only on the diagnostic trace. The shared harness
still captures the trusted natural-stop reference and uses its normal correctness
checks. Every scored continuation must cover all 5,120 tokens, all requested
prompts must remain, every correctness report must pass the original thresholds,
and observed scoring coverage must be 100%. A shorter-than-5,120-token output, excluded or missing result,
mis-tokenized response, or truncated scoring span fails the capacity diagnostic.
Shorter input fallback does not relax the output length or correctness checks. A forced token count
that changes when retokenized for teacher forcing is retained as a failure to
confirm the target scorer span, not silently accepted.

For a one-request control, rerun in a fresh directory with `--prompt-count 1`.
For repeated-prefix cache pressure use `--prefixes repeated`; this intentionally
repeats the single longest source prompt under distinct request IDs. For longer
scorer exposure use e.g. `--scorer-repetitions 10`. Tune only one memory fraction
at a time and retain the failed run directories. A successful candidate setting
must be transferred into the real campaign fixture/launcher and requalified in
separate work; this tool never changes those files.

The 32-request diagnostic uses main's 2 ms interval burst. That is not sustained
C32 or C4 concurrency. Correctness grading itself is sequential in the production
harness; prompt count 32 means 32 full continuations per engine, not 32 simultaneous
scorer calls. Mixed-length traffic, other concurrency levels, larger advertised
contexts, and full campaign acceptance remain separate validation work.

## Evidence and interpretation

- `summary.json`: preparation/failure/completion, elapsed time, exact lengths and
  request/trace hashes. `prepared_only` is never GPU acceptance.
- `source_*`, `effective_campaign_fields.json`, `bench_request.json`: reproducible
  source/override pins, tokenizer/template receipt and the diagnostic request.
- `runtime/<role>/`: raw/resolved flags, server-info before/after, image handle,
  cache-mount state, and Docker state/restart count before teardown.
- `captured_outputs.json`, `scorer_replays/rep-*/`: actual full continuations,
  likelihood/coverage and scored-span evidence, unchanged correctness reports, and
  elapsed time for each scoring batch. Missing/excluded prompts fail.
- `gpu-memory.jsonl`, `telemetry.json`: phase-labelled samples and observed peak
  used/minimum free device memory. These are sampled device totals, not allocator
  peaks; short transients may be missed. Allocated/reserved allocator memory is
  not available through this tool.
- `round/`: standard environment (GPU edition, driver/CUDA, CPU), weight hashes,
  engine logs, baseline natural-stop references, per-request TTFT/ITL/E2E and
  output counts, aggregate throughput, and round report. This altered workload
  is not a score submission or a valid performance comparison.

A zero exit means the requested diagnostic finished with complete scorer evidence.
It does not close PAR-144 or authorize launch. Review logs and before/after server
info for token-pool size, effective KV/SSM/speculative settings, queueing,
retractions/evictions, and remaining headroom. CUDA/OOM/traceback log matches,
container death/restart, failed telemetry, or failed scoring make the run fail.
Subprocess and protocol failures retain evidence; an operator termination can
leave the summary at failed. Preserve output directories for comparison.


## Retrying an older preparation failure

If preparation reported `insufficient distinct long-form prompts ... '16k': 7`,
update the diagnostic branch (`git pull --ff-only` in its clean checkout) and
repeat preparation with a new `PRO6000_RUN_ROOT`. Do not reuse the failed output
directory. The new preparation removes the balanced 128-prompt prerequisite;
for example, 25 eligible 16k inputs plus seven shorter inputs can fill the
32-prompt diagnostic. Completed older balanced previews remain supported through
`--source-preview`; their available shorter rows can also fill a 16k shortfall.
The production campaign sampler and qualification policy remain unchanged.

### Docker model bind-mount failures

Use `--docker-model-volume` on GPU launches if the pod's local staged path cannot
be bind-mounted by its Docker daemon (OCI `procfd` / `/model` mount failure).
This option copies the already staged weights through `docker cp` into a unique
Docker-managed volume, verifies every file with SHA-256 inside the daemon's
namespace, and mounts that volume read-only for all generation and scorer phases.
It preserves the model revision, engine image, serving flags, and offline engine
network. Allow disk space for an additional complete copy of the model and time
for copying and hashing. The volume is removed on normal exit or handled failure;
a killed process can leave its uniquely named volume for manual cleanup.
`model_volume.json` records successful verification.

After a failed GPU launch, reuse `prepared/source_preview` and the prepared
campaign fields, but choose a new output directory (for example `natural-volume`).
No prompt preparation needs to be repeated.

### Speculative streaming and timing

This correctness-only diagnostic accepts multiple output tokens per SSE chunk,
including a complete response in one chunk. It retains completion text, usage
counts, request validation, output-length checks, and full scorer coverage.
Chunk gaps do not measure per-token ITL: `summary.json` and the round's
`bench_report.json` explicitly set `performance_score_valid: false`. Do not use
this diagnostic's ITL, SLA, or speedup fields to qualify or score a campaign.
Diagnostic goodput is conservatively zero; no per-token gaps are fabricated.
The shared HTTP client and metrics helper keep strict token-timing defaults;
only this CLI opts out via hooks scoped to its round. Speculation stays enabled.

### Review previously saved engine logs

The scanner records exact file names and line numbers in `engine_log_review.json`.
It recognizes the exact 262151-versus-262144 scorer headroom warning and complete,
explicitly ignored TorchCodec loading tracebacks from multimodal processors
(including `mimo_audio` and `mimo_v2`). Other tracebacks,
OOMs, and CUDA errors still fail review. These exemptions are specific to this
text-only diagnostic and do not qualify the advertised context boundary.

To rescan a completed run without repeating inference, run from the checkout:

```bash
export PYTHONPATH=.
nohup python -u -c 'import runpy,sys; from pathlib import Path; runpy.run_path("ops/pro6000-correctness-probe.py")["review_engine_logs"](Path(sys.argv[1]))' \
  /path/to/results/natural-volume-v3 \
  > /path/to/results/log-review.log 2>&1 < /dev/null &
```

This writes a separate log review; it preserves the original failed summary.
A successful rescan alone is not a replacement for lifecycle, scorer, or capacity
evidence.
