# Trusted SGLang build base for 4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc.
# The final image digest pins the resolved OS, Python, CUDA and kernel wheels.
# Build on linux/amd64, then use python -m builder --engine sglang for
# the offline empty-patch baseline and the non-empty miner build check.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        curl ca-certificates git build-essential ninja-build cmake ccache \
        pkg-config libssl-dev libnuma-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal

ENV PATH="/opt/venv/bin:/root/.cargo/bin:${PATH}"

RUN pip install --upgrade pip \
    && pip install "setuptools>=61.0" "setuptools-rust>=1.11" \
        "setuptools-scm>=8.0" wheel ninja "cmake>=3.26.1" "packaging>=24.2"

# Match the source commit's Torch ABI and its official CUDA 13 recipe.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"
COPY images/baseline-sglang/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install -r /tmp/requirements-runtime.txt \
    && python -c "import torch; assert torch.__version__.startswith('2.13.0'), torch.__version__; assert (torch.version.cuda or '').startswith('13'), torch.version.cuda"

# Rust is outside the campaign's Python patch surface. Compile it once in
# this trusted, network-enabled build at the same /src path used by miners.
# COPY baseline/ in the hermetic builder retains these .so files. Disabling
# subsequent Rust builds avoids Cargo downloads during editable installs.
# No shared writable cache is introduced for miner code.
WORKDIR /src
RUN git init \
    && git remote add origin https://github.com/sgl-project/sglang.git \
    && git fetch --depth=1 origin 4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = 4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc \
    && CARGO_BUILD_JOBS=4 pip install --no-deps --no-build-isolation -e python/ \
    && python -c "from pathlib import Path; assert list(Path('/src/python/sglang').rglob('*.so')), 'Rust extensions missing'" \
    && kernels lock python \
    && kernels download python \
    && pip check \
    && pip freeze > /opt/sglang-dependencies.lock \
    && rm -rf /src/.git /src/rust/target /src/rust/sglang-radix-tree/target /tmp/pip-* /root/.cache/pip

ENV SGLANG_BUILD_RUST_EXTS=none \
    CARGO_NET_OFFLINE=true

LABEL org.opencontainers.image.source="https://github.com/Pareton-ai/pareton" \
      org.opencontainers.image.description="Pareton hermetic SGLang baseline" \
      ai.pareton.sglang.commit="4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc"
