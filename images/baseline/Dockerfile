# Pareton Stage 0 baseline (build) image.
# Pre-bake CUDA/Python/torch + vLLM build-system deps so hermetic builds can
# run with --network=none and `pip install --no-deps --no-build-isolation -e .`.
# Aligned to vLLM pin ee0da84 (v0.24.0): CUDA 13.0.2, torch 2.11.0, Python 3.12.
# Pin and publish; put the resulting digest into the campaign manifest.
#
# Keep layers lean for GitHub-hosted runners (~disk-limited): install torch
# alone, drop optional audio/vision wheels and fastapi[standard] extras.

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

# Build-system requires from vLLM pyproject.toml at ee0da84 (except torch).
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        "cmake>=3.26.1" \
        ninja \
        "packaging>=24.2" \
        "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8.0" \
        "setuptools-rust>=1.9.0" \
        wheel \
        jinja2 \
    && rm -rf /tmp/* /root/.cache/pip

# Torch alone (pulls CUDA 13 wheels). Separate layer so rebuilds of lighter
# deps do not re-download ~2GB of NVIDIA packages.
RUN pip install --no-cache-dir "torch==2.11.0" \
    && rm -rf /tmp/* /root/.cache/pip

# Runtime deps for hermetic `pip install --no-deps --no-build-isolation -e .`.
# Must cover vLLM api_server + EngineCore start (common.txt + cuda FlashInfer).
# Ad-hoc one-package patches caused multi-hour A2b rebuilds per missing module.
COPY images/baseline/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
    && rm -rf /tmp/* /root/.cache/pip

WORKDIR /opt/pareton
LABEL org.opencontainers.image.source="https://github.com/Pareton-ai/pareton" \
      org.opencontainers.image.description="Pareton Stage 0 hermetic baseline (vLLM v0.24.0 pin)"
