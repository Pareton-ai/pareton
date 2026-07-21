# Pareton Stage 0 baseline (build) image.
# Pre-bake CUDA/Python/torch + vLLM build-system deps so hermetic builds can
# run with --network=none and `pip install --no-deps --no-build-isolation -e .`.
# Aligned to vLLM pin ee0da84 (v0.24.0): CUDA 13.0.2, torch 2.11.0, Python 3.12.
# Pin and publish; put the resulting digest into the campaign manifest.

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

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
        pkg-config \
        libssl-dev \
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
        | sh -s -- -y --default-toolchain stable \
    && . "$HOME/.cargo/env" \
    && rustc --version

ENV PATH="/root/.cargo/bin:${PATH}"

# Build-system requires from vLLM pyproject.toml at ee0da84 + CUDA torch wheels.
# Runtime deps are a practical subset so `import vllm` / api_server can start once
# the hermetic step does `pip install --no-deps --no-build-isolation -e .`.
RUN pip install --upgrade pip \
    && pip install \
        "cmake>=3.26.1" \
        ninja \
        "packaging>=24.2" \
        "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8.0" \
        "setuptools-rust>=1.9.0" \
        wheel \
        jinja2 \
        "torch==2.11.0" \
        "torchaudio==2.11.0" \
        "torchvision==0.26.0" \
        numpy \
        "typing_extensions>=4.10" \
        "pydantic>=2.12.0" \
        "fastapi[standard]>=0.133.0,<0.137.0" \
        "uvicorn[standard]" \
        sentencepiece \
        "protobuf>=5.29.6" \
        "transformers>=5.5.3" \
        "tokenizers>=0.21.1" \
        "safetensors>=0.6.2" \
        "aiohttp>=3.13.3" \
        "openai>=2.0.0" \
        "prometheus_client>=0.18.0" \
        pillow \
        "tiktoken>=0.6.0" \
        "filelock>=3.16.1" \
        pyyaml \
        einops \
        regex \
        psutil \
        requests \
        tqdm \
        msgspec \
        "pyzmq>=25.0.0"

WORKDIR /opt/pareton
LABEL org.opencontainers.image.source="https://github.com/Pareton-ai/pareton" \
      org.opencontainers.image.description="Pareton Stage 0 hermetic baseline (vLLM v0.24.0 pin)"
