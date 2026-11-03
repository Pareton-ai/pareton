# Trusted SGLang build base that matches the Open Kernel Challenge (OKC) engine.
# It starts from OKC's own image, so sglang-kernel 0.4.9, NCCL 2.30.7, the Muse
# DeepGEMM and FlashInfer 0.7 are exactly what OKC runs. Only the Python
# package and its Rust extensions are rebuilt, by the sglang-okc engine preset:
#   pip install --no-deps --no-build-isolation -e python/
# Every SGLang build runs with --network=none, so this image stages what that
# command needs offline: the build backend, the Rust toolchain and the Cargo
# registry for every Cargo workspace under rust/ (sglang-radix-tree is its
# own), plus a warm Cargo target directory for the a55ab7d5 workspace.
# Build on linux/amd64, then use python -m builder --engine sglang-okc.
FROM lmsysorg/sglang:dev-muse-spark-open@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a

ARG SGLANG_COMMIT=a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
ARG MAX_JOBS=8

# OKC installed SGLang with build isolation, so the backend from
# python/pyproject.toml may be missing. --no-build-isolation needs it here.
RUN python3 -m pip install --no-cache-dir \
        "setuptools>=61.0" "setuptools-rust>=1.11" "setuptools-scm>=8.0" wheel \
    && python3 -c "import torch; assert torch.__version__.startswith('2.14.1'), torch.__version__"

# The workspace pins its toolchain in rust/rust-toolchain.toml; install that
# channel so no Cargo call has to download one in the offline build.
RUN if ! command -v rustup >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --default-toolchain none --profile minimal; \
    fi
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /src
RUN git init -q \
    && git remote add origin https://github.com/sgl-project/sglang.git \
    && git fetch -q --depth=1 origin "$SGLANG_COMMIT" \
    && git checkout -q --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$SGLANG_COMMIT" \
    && toolchain=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust/rust-toolchain.toml) \
    && rustup toolchain install "$toolchain" --profile minimal \
    && rustup default "$toolchain" \
    && for lock in $(find rust -name Cargo.lock); do \
        cargo fetch --locked --manifest-path "$(dirname "$lock")/Cargo.toml"; \
    done

# Cargo builds dependencies outside /src so they survive into miner builds;
# miners can't change rust/**, so their builds mostly reuse this target dir.
ENV CARGO_NET_OFFLINE=true \
    CARGO_TARGET_DIR=/opt/sglang-rust-target

# Warm build, offline, with the same command miners get. This proves the
# staged inputs are enough and fills the Cargo target directory. No pip check:
# OKC's image installs several packages with --no-deps, so it may already fail.
RUN --network=none CARGO_BUILD_JOBS="$MAX_JOBS" \
        pip install --no-deps --no-build-isolation -e python/ \
    && python3 -c "from pathlib import Path; assert list(Path('/src/python/sglang').rglob('*.so')), 'Rust extensions missing'" \
    && python3 -c "import torch; assert torch.__version__.startswith('2.14.1'), torch.__version__" \
    && python3 -c "import sglang; from sglang.srt.layers.moe.muse_spark_megamoe import require_muse_megamoe_patch; require_muse_megamoe_patch()" \
    && rm -rf /src /tmp/pip-* /root/.cache/pip

# OKC's image sets SGLANG_RUST_BUILD_MODE=never: the runtime trusts the
# extensions pip built and never invokes Cargo in the offline bench container.
RUN test "$SGLANG_RUST_BUILD_MODE" = never

LABEL org.opencontainers.image.source="https://github.com/Pareton-ai/pareton" \
      org.opencontainers.image.description="Pareton hermetic SGLang base matching the OKC engine" \
      ai.pareton.sglang.commit="a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
