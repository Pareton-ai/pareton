# Standalone SGLang NVFP4 sample round

Build the included comment-only patch and evaluate it on the warmed, dedicated
GPU VM without creating a campaign or writing to the database or chain.

The sample uses [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/tree/dbb8f445b3145f8a4c18ddc769f032d57d32867c)
at revision `dbb8f445b3145f8a4c18ddc769f032d57d32867c`. Finish caching it
before starting the runner; its tokenizer is read from:

```text
/workspace/hf-cache/nvidia--Qwen3.8-27B-NVFP4/dbb8f445b3145f8a4c18ddc769f032d57d32867c
```

## Run

From a local checkout containing these files, set `VM_HOST` to your GPU host and
copy a source archive to the VM:

```bash
VM_HOST='<gpu-host>'
git archive --format=tar.gz -o /tmp/pareton-sample-source.tar.gz HEAD
scp -P 30499 -i ~/.ssh/pareton /tmp/pareton-sample-source.tar.gz \
  "root@$VM_HOST:/workspace/"
ssh -p 30499 -i ~/.ssh/pareton "root@$VM_HOST"
```

On the VM:

```bash
mkdir -p /workspace/pareton-sample-source
tar -xzf /workspace/pareton-sample-source.tar.gz -C /workspace/pareton-sample-source
cd /workspace/pareton-sample-source
nohup bash ops/sglang-sample-round/run.sh \
  > /workspace/pareton-sample-round-nvfp4.log 2>&1 < /dev/null &
tail -f /workspace/pareton-sample-round-nvfp4.log
```

The runner uses `/workspace/pareton-sample-round-nvfp4` for its virtual environment,
request, trace, build log and reports. Pass a different output directory as its
first argument to generate a fresh sample directory. Rerunning with the same
directory reuses the sampled trace and writes a new timestamped report directory.
Use a fresh directory when switching models; do not pass an old FP8 sample
directory, because its trace was prepared with a different pinned tokenizer.

The VM needs Python 3 with venv support, Git, Docker with the NVIDIA runtime,
`flock`, and four available RTX5090 GPUs. Run as root. Internet access is required
for dependencies, the pinned SGLang source, and dataset sampling. The archive
contains tracked source files; no validator `.env` or registry credentials are
needed.

## What runs

- Build `submission.diff` against SGLang commit
  `4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc`, using the cached baseline runtime
  digest `43d5d33c2d3f61923d7ff96b8c69b77b8ddee28f749c10bb876ed538169fd431`.
  The candidate stays local as `pareton-sample:sglang-minimal`.
- Use the workload and hardware settings from
  [the campaign fixture](../../fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json),
  with the sample's NVFP4 model override: four RTX5090 GPUs, 262144 context configuration,
  32 sampled prompts across the 4K/8K/16K/32K input tiers, and three timing
  repetitions by default.
- Auto-detect the checkpoint's mixed FP8/NVFP4 quantization from its model
  configuration (`quantization: null` in the request, no `--quantization` flag).
- Run baseline, candidate, NVFP4 correctness scoring, and baseline drift replay.
  The fixed local sampling seed is reproducible. It does not represent a
  chain-selected production round or test a full 262K input window.
- Reuse `/workspace/hf-cache` and `/workspace/engine-cache`. Local Docker image
  IDs avoid image pulls and registry authentication. Candidates retain the
  harness's cold-cache isolation.

The patch changes only comments in Python, CUDA and Rust source. It is applied
once to a fresh checkout on each build; no intended performance improvement is
expected. Pulling the baseline image does not populate the BuildKit compiler
cache, so the native rebuild can still take hours.

## Cleanup and results

The runner holds `/opt/pareton/.static-host.lock`, cleans scoped Pareton bench
containers and networks before and after the run, and checks for remaining GPU
compute processes. Unknown processes cause failure and are not terminated.
Checkpoint caches, images, build cache, logs and reports are retained.

A hard kill or host crash can bypass the exit cleanup. An existing VPS reaper
can clean abandoned containers after the lock is released; this runner does not
install a periodic service.

Inspect `output-<UTC timestamp>/bench_report.json`, `harness.log`, and `evidence/`
under the output directory. Exit zero means the harness completed; inspect the
candidate's correctness outcome, errors and speedup in the report. The request,
workload trace and sampling receipt are saved alongside `build.log`.

The runner logs timestamps, elapsed seconds, phase transitions, dependency
installation, streamed build output, dataset row fetches, resolved image IDs,
request settings, and cleanup results. Failures include the phase, script line,
and exit code. Python output is unbuffered so redirected logs update promptly.
Sampling logs contain row indices and timing, not prompt contents.

Local validation covers shell syntax and request-generation/schema checks with
mocked sampling and image inspection. A real build and GPU evaluation are still
required to validate the end-to-end run.
