# Pareton Stage 0 baseline image.
# Pre-bake CUDA/Python/vLLM build deps so hermetic builds can run with --network=none.
# Pin and publish; put the resulting digest into the campaign manifest.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        build-essential \
        ninja-build \
        cmake \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Pre-install common vLLM build/runtime deps (exact pins are ops-owned).
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install \
        numpy \
        typing-extensions \
        pydantic \
        fastapi \
        uvicorn \
        sentencepiece \
        protobuf

WORKDIR /opt/pareton
LABEL org.opencontainers.image.source="https://github.com/Pareton-ai/pareton" \
      org.opencontainers.image.description="Pareton Stage 0 hermetic baseline"
