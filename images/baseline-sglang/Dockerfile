# Trusted SGLang build base for 11972e520709a76766f91744a672e92727fdcb09.
# The final image digest pins the resolved OS, Python, CUDA and kernel wheels.
# Build on linux/amd64, then use python -m builder --engine sglang for
# the offline empty-patch baseline and the non-empty miner build check.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04 AS bootstrap

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
        "torch==2.14.1" torchvision "torchaudio==2.11.0"
COPY images/baseline-sglang/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install -r /tmp/requirements-runtime.txt \
    && python -c "import torch; assert torch.__version__.startswith('2.14.1'), torch.__version__; assert (torch.version.cuda or '').startswith('13'), torch.version.cuda"

# Bootstrap Rust and retain Cargo's resolved dependency sources. The native
# stage below reenables Rust rebuilds against this offline dependency cache.
WORKDIR /src
RUN git init \
    && git remote add origin https://github.com/sgl-project/sglang.git \
    && git fetch --depth=1 origin 11972e520709a76766f91744a672e92727fdcb09 \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = 11972e520709a76766f91744a672e92727fdcb09 \
    && CARGO_BUILD_JOBS=4 pip install --no-deps --no-build-isolation -e python/ \
    && python -c "from pathlib import Path; assert list(Path('/src/python/sglang').rglob('*.so')), 'Rust extensions missing'" \
    && pip check \
    && pip freeze > /opt/sglang-dependencies.lock \
    && rm -rf /src/.git /src/rust/target /src/rust/sglang-radix-tree/target /tmp/pip-* /root/.cache/pip

ENV SGLANG_BUILD_RUST_EXTS=none \
    CARGO_NET_OFFLINE=true \
    SGLANG_USE_SGL_FA3_KERNEL=1

# This pin's community FA3 repository has no Torch 2.13 build. Use the
# implementation bundled in the pinned sglang-kernel wheel directly, as
# supported by sglang/kernels/ops/attention/flash_attention_v3.py. This also
# avoids a Hugging Face lookup from the offline evaluation container.

LABEL org.opencontainers.image.source="https://github.com/Pareton-ai/pareton" \
      org.opencontainers.image.description="Pareton hermetic SGLang baseline" \
      ai.pareton.sglang.commit="11972e520709a76766f91744a672e92727fdcb09"

# Reuse the verified dependency/Rust bootstrap by digest. The subsequent trusted
# empty-patch build replaces the prebuilt AOT package with the pinned source build.
FROM ghcr.io/pareton-ai/pareton-baseline@sha256:8119dfe64ec0d342a988ff3f3627b0a81e7000aef2b792c2a5ddf11dbb3f1f72 AS native
RUN pip install 'scikit-build-core==0.11.6' 'cmake==3.31.10'
# FlashInfer otherwise fetches trtllm-gen cubins from NVIDIA's artifactory at
# first use, which fails in the offline evaluation container. Upstream installs
# the matching cubin package from FlashInfer's index; PyPI stops at 0.6.13.
RUN pip install --no-deps --index-url https://flashinfer.ai/whl 'flashinfer-cubin==0.7.0.post1' \
    && python -c "import flashinfer, flashinfer_cubin; assert flashinfer.__version__ == flashinfer_cubin.__version__, (flashinfer.__version__, flashinfer_cubin.__version__)"
COPY images/baseline-sglang/prepare-deps.py /opt/pareton/prepare-sglang-deps.py
RUN python /opt/pareton/prepare-sglang-deps.py
COPY images/baseline-sglang/install.sh /usr/local/bin/pareton-install-sglang
RUN chmod 755 /usr/local/bin/pareton-install-sglang
ENV CARGO_NET_OFFLINE=true \
    SGLANG_BUILD_RUST_EXTS=all \
    CARGO_TARGET_DIR=/opt/sglang-rust-target
LABEL ai.pareton.sglang.native-build="1"
