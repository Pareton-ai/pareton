# Pareton Stage 0 baseline (build) image -- SGLang variant.
# Pre-bake CUDA/Python/torch + SGLang's runtime deps so hermetic builds can
# run with --network=none and `pip install --no-deps --no-build-isolation -e python/`.
# Aligned to SGLang pin v0.5.17 (b6a09f38): CUDA 13.0.2, torch 2.11.0, Python 3.12.
# Pin and publish; put the resulting digest into the campaign manifest.
#
# Unlike the vLLM baseline this image bakes NO TORCH_CUDA_ARCH_LIST: SGLang
# compiles no CUDA kernels. sglang-kernel ships as a prebuilt wheel, so no GPU
# is needed to build. The wheel is CUDA-flavor specific (+cu130 for this pin),
# not a generic "any GPU" binary: PyPI's plain 0.4.5 fails on RTX 5090 (sm_120).
#
# The kernel wheel MUST stay version-matched to the source tag the campaign
# pins. sglang-kernel 0.4.5 pairs with v0.5.17; a mismatched pair builds green
# and then dies at serve time, which is how the PAR-54 spike first failed.

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.12 (deadsnakes) + build toolchain + Rust (setuptools-rust).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        gnupg \
        git \
        build-essential \
        ninja-build \
        cmake \
        ccache \
        pkg-config \
        libssl-dev \
        libnuma-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.12 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && . "$HOME/.cargo/env" \
    && rustc --version \
    && rm -rf /tmp/* /var/tmp/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Build-system requires from SGLang python/pyproject.toml at v0.5.17 (except torch).
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        "setuptools>=61.0" \
        "setuptools-rust>=1.10" \
        "setuptools-scm>=8.0" \
        wheel \
        ninja \
        "cmake>=3.26.1" \
        "packaging>=24.2" \
    && rm -rf /tmp/* /root/.cache/pip

# Torch from the cu130 index (plain PyPI 2.11.0 is not +cu130). Official
# lmsysorg/sglang:v0.5.17 is torch==2.11.0+cu130; an unpinned torchvision
# previously pulled torch 2.13 and broke sglang-kernel 0.4.5 on sm_120.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
    && rm -rf /tmp/* /root/.cache/pip

# Runtime deps for hermetic `pip install --no-deps --no-build-isolation -e python/`.
# Must cover sglang.launch_server start, not a lean subset: the server imports
# gated paths (flashinfer, sglang-kernel, xgrammar) at startup.
COPY images/baseline-sglang/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
    && python3 -c "import torch; v=torch.__version__; assert v.startswith('2.11.0'), v; assert (torch.version.cuda or '').startswith('13'), torch.version.cuda" \
    && rm -rf /tmp/* /root/.cache/pip

WORKDIR /opt/pareton
LABEL org.opencontainers.image.source="https://github.com/Pareton-ai/pareton" \
      org.opencontainers.image.description="Pareton Stage 0 hermetic baseline (SGLang v0.5.17 pin)"
