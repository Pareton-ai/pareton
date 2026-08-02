"""Guard: A2 runtime must cover vLLM ee0da84 cuda.txt (minus torch)."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = _ROOT / "images" / "baseline" / "requirements-runtime.txt"
_A2B = _ROOT / "ops" / "a2b-build.sh"

# From https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/requirements/cuda.txt
# torch is installed in the Dockerfile; common.txt is vendored separately.
_CUDA_TXT_PACKAGES = (
    "numba",
    "torchaudio",
    "torchvision",
    "flashinfer-python",
    "flashinfer-cubin",
    "apache-tvm-ffi",
    "tilelang",
    "nvidia-cudnn-frontend",
    "fastsafetensors",
    "nvidia-cutlass-dsl",
    "quack-kernels",
    "tokenspeed-mla",
    "humming-kernels",
)


def test_requirements_runtime_covers_cuda_txt() -> None:
    text = _RUNTIME.read_text(encoding="utf-8")
    missing = [name for name in _CUDA_TXT_PACKAGES if name not in text]
    assert not missing, f"requirements-runtime.txt missing cuda.txt pkgs: {missing}"


def test_a2b_build_requires_explicit_base_digest() -> None:
    text = _A2B.read_text(encoding="utf-8")
    assert "need_env BASE" in text
    assert "@sha256:" in text
    # Stale hardcoded default caused A2b FROM pre-dep-fix images.
    assert (
        "72b601e4314fa3c5e522e814305fad3a10f06eb174a5785e2729e655cb490986" not in text
    )
