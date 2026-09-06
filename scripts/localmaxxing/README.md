# Custom vLLM LocalMaxxing run

This directory is a standalone bundle. No separate repository checkout or host
vLLM installation is needed. It includes the measurement helpers and prompt;
bootstrap downloads the checksum-verified LocalMaxxing v0.1.39 Linux x86_64 CLI.
The existing image supplies vLLM and its CUDA userspace toolchain.

Run as root on a Linux x86_64 GPU host with Docker, NVIDIA Container Toolkit,
Python 3.8+, curl, tar, sha256sum, and `nvidia-smi` available.

## Use the container you already launched

Upload `pareton-localmaxxing.tar.gz` to `/workspace`, then:

```bash
mkdir -p /workspace/pareton-localmaxxing
tar -xzf /workspace/pareton-localmaxxing.tar.gz -C /workspace/pareton-localmaxxing
bash /workspace/pareton-localmaxxing/reproduce-pareton.sh \
  --existing-container pareton-vllm
```

This installs the benchmark client, checks the running container, and benchmarks
its API at `127.0.0.1:8000`. It does not pull an image, download weights, launch,
restart, or stop the container. Leave your existing server running and ready.
The image digest and argument list must match the bundled recipe; a mismatch
fails before benchmarking. The published port must resolve to host port 8000.
The actual entrypoint, arguments, import path, and version come from the container.

For a detached run:

```bash
nohup bash /workspace/pareton-localmaxxing/reproduce-pareton.sh \
  --existing-container pareton-vllm \
  > /workspace/pareton-localmaxxing.log 2>&1 < /dev/null &
tail -f /workspace/pareton-localmaxxing.log
```

## Launch the image and benchmark

If port 8000 and GPU 0 are free, omit the flag:

```bash
bash /workspace/pareton-localmaxxing/reproduce-pareton.sh
```

This pulls the pinned image, uses the `pareton-hf-cache` named volume, launches
`pareton-lmx-vllm`, waits for readiness, and benchmarks it. Authenticate with
`docker login ghcr.io -u YOUR_GITHUB_USERNAME` first if the package is private.
The optional `HF_TOKEN` is passed by variable name to Docker. Do not enable shell
tracing. The server remains running afterwards. Remove the managed container with:

```bash
bash /workspace/pareton-localmaxxing/reproduce-pareton.sh stop
```

`stop` and `serve` reject `--existing-container` to protect the manually launched
server. Separate `bootstrap`, `serve`, and `run` steps are available. After the
first bootstrap, `run --existing-container pareton-vllm` skips CLI installation.

## Settings and results

Edit `qwen38-27b-fp8-pareton.recipe` for the model and benchmark settings. It
preserves the supplied model revision and serving flags. The workload remains
reasoning-v1 with cache-busting nonces, greedy generation, 512 maximum output
tokens, two warmups, three timed iterations, and concurrency 256. The server
schedules at most 32 sequences; excess requests queue. Set `CONCURRENCY=32`
in the recipe for a lower-concurrency measurement.

New runs use unique names beneath `/workspace/runs/`. Use `--run-name my-run`
for an explicit name; existing result directories are never overwritten.
Each directory contains `payload.json`, benchmark artifacts, the image digest,
model revision, actual process command, vLLM/Torch/CUDA versions, and server logs.
Results are prepared locally; nothing is submitted publicly.

The bundled power sampler measures GPU 0, and its window includes client warmups.
Use a single-GPU instance or a server using only GPU 0 for consistent power data.
The managed launch exposes only GPU 0. This measures LocalMaxxing throughput,
not the Pareton campaign score or correctness checks.

## Attribution and validation

Measurement helpers and the canonical prompt were bundled from
[lium-localmaxxing commit e9792ebc](https://github.com/Datura-ai/lium-localmaxxing/tree/e9792ebc2f60bdd9275f02f53f1d3eec34b86314).
Their MIT license is included in `helpers/LICENSE`. The local `run_speed.sh`
also propagates benchmark and metadata failures. There is no runtime dependency
on that repository.

Local checks cover shell syntax and simulated lifecycle and artifact flows.
Actual GPU performance remains to be verified on the provisioned instance.
