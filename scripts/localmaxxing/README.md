# Custom vLLM LocalMaxxing run

Run these scripts from a cloned Pareton repository. No separate benchmark
repository or host vLLM installation is needed. The helpers and prompt are included;
bootstrap downloads the checksum-verified LocalMaxxing v0.1.39 Linux x86_64 CLI.
The existing image supplies vLLM and its CUDA userspace toolchain.

Run as root on a Linux x86_64 GPU host with Docker, NVIDIA Container Toolkit,
Python 3.8+, git, curl, tar, sha256sum, and `nvidia-smi` available.

Clone the repo, or enter your existing checkout. All commands below run from the
repo root:

```bash
git clone https://github.com/Pareton-ai/pareton.git
cd pareton
```

## Use the container you already launched

Reference launch commands for `pareton-vllm` are below. Skip this block if the
container is already running with these settings. The first command removes a
stopped or failed container; Docker refuses to remove a running container without
`-f`. If the name does not exist yet, proceed with volume creation.

```bash
sudo docker rm pareton-vllm
sudo docker volume create pareton-hf-cache

IMAGE='ghcr.io/pareton-ai/pareton-engine@sha256:3891dd3de2d04ecbe197af3f8cf93668e54ab4ab8e284a6f02ecf1969ffd7c09'

sudo docker run -d \
  --name pareton-vllm \
  --gpus all \
  --ipc=host \
  -p 127.0.0.1:8000:8000 \
  --mount type=volume,source=pareton-hf-cache,target=/hf-cache \
  -e HF_HOME=/hf-cache \
  "$IMAGE" \
  --model Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gdn-prefill-backend triton

sudo docker logs -f pareton-vllm
```

Wait for server startup, then press Ctrl+C to leave log streaming. The detached
container keeps running. Confirm readiness before starting the benchmark:

```bash
curl -fsS http://127.0.0.1:8000/health
```

Start the benchmark from the repo root:

```bash
bash scripts/localmaxxing/reproduce-pareton.sh \
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
nohup bash scripts/localmaxxing/reproduce-pareton.sh \
  --existing-container pareton-vllm \
  > /workspace/pareton-localmaxxing.log 2>&1 < /dev/null &
tail -f /workspace/pareton-localmaxxing.log
```

## Launch the image and benchmark

If port 8000 and GPU 0 are free, omit the flag:

```bash
bash scripts/localmaxxing/reproduce-pareton.sh
```

This pulls the pinned image, uses the `pareton-hf-cache` named volume, launches
`pareton-lmx-vllm`, waits for readiness, and benchmarks it. Authenticate with
`docker login ghcr.io -u YOUR_GITHUB_USERNAME` first if the package is private.
The optional `HF_TOKEN` is passed by variable name to Docker. Do not enable shell
tracing. The server remains running afterwards. Remove the managed container with:

```bash
bash scripts/localmaxxing/reproduce-pareton.sh stop
```

`stop` and `serve` reject `--existing-container` to protect the manually launched
server. Separate `bootstrap`, `serve`, and `run` steps are available. After the
first bootstrap, `run --existing-container pareton-vllm` skips CLI installation.

For these instances, a detected `NVIDIA H200` is assumed to be SXM. Both bootstrap
and `run` automatically normalize that exact name to LocalMaxxing's canonical
`NVIDIA H200 SXM` before hardware validation. Other GPU names and measured hardware
fields are preserved. Existing metadata from a failed attempt is corrected too;
no manual JSON edit is needed. With the updated scripts in your checkout, resume with:

```bash
bash scripts/localmaxxing/reproduce-pareton.sh run \
  --existing-container pareton-vllm
```

## Settings and results

Edit `scripts/localmaxxing/qwen38-27b-fp8-pareton.recipe` for the model and benchmark settings. It
preserves the supplied model revision and serving flags. The workload remains
reasoning-v1 with cache-busting nonces, greedy generation, 512 maximum output
tokens, two warmups, three timed iterations, and concurrency 128. The recipe sets
`MAX_NUM_SEQS=128` and derives both client `CONCURRENCY` and server `--max-num-seqs`
from it. This sends 128 concurrent requests, each with one prompt; it does not put
128 prompts in a single API request. Default run names include `c128`.

Earlier `c256` results used 256 client requests against the same server limit of
32 active sequences. Those measurements represent a different offered load and
must not be relabeled as `c128` results. Recreate the server with
`--max-num-seqs 128` and rerun to measure concurrency 128.

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
The previous concurrency-32 and concurrency-256 runs completed on the provisioned
H200 SXM instance. The updated concurrency-128 configuration requires a new GPU run.
