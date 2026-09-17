"""Generate a miner patch that exercises CUDA, CMake, JIT and Rust builds."""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

CUDA_SOURCE = r"""#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void pareton_add_seven_kernel(const float* x, float* y, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 7.0f;
}

at::Tensor pareton_add_seven(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous());
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty_like(x);
  const auto n = x.numel();
  if (n > 0) {
    pareton_add_seven_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}
"""

JIT_SOURCE = r"""#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

namespace sglang::pareton {
__global__ void add_eleven(float* dst, const float* src, size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i] + 11.0f;
}

void run(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;
  SymbolicSize n = {"num_elements"};
  SymbolicDevice device;
  TensorMatcher({n}).with_dtype<float>().with_device<kDLGPU>(device).verify(dst).verify(src);
  RuntimeCheck(n.unwrap() > 0, "probe requires nonempty tensors");
  LaunchKernel(div_ceil(n.unwrap(), size_t(256)), 256, device.unwrap())(
      add_eleven, static_cast<float*>(dst.data_ptr()),
      static_cast<const float*>(src.data_ptr()), n.unwrap());
}
}  // namespace sglang::pareton
"""

PYTHON_SOURCE = '''"""Run with python -m sglang.pareton_build_probe on the validation GPU."""
import json


def main():
    import torch
    import sgl_kernel
    from sglang.kernels.jit.utils.compile import load_jit
    from sglang.srt.mem_cache.rust_tree_core import mem_cache

    assert mem_cache.PARETON_NATIVE_PROBE == 41
    x = torch.arange(1027, device="cuda", dtype=torch.float32)
    actual = torch.ops.sgl_kernel.pareton_add_seven(x)
    torch.testing.assert_close(actual, x + 7, rtol=0, atol=0)
    module = load_jit(
        "pareton_native_probe",
        cuda_files=["pareton_probe.cuh"],
        cuda_wrappers=[("run", "pareton::run")],
    )
    result = torch.empty_like(x)
    module.run(result, x)
    torch.testing.assert_close(result, x + 11, rtol=0, atol=0)
    torch.cuda.synchronize()
    print(json.dumps({
        "cuda_registered_kernel": "passed",
        "sglang_jit_kernel": "passed",
        "rust_extension": "passed",
        "elements": x.numel(),
        "gpu": torch.cuda.get_device_name(),
        "sgl_kernel_path": sgl_kernel.__file__,
    }), flush=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    image_ref, output = sys.argv[1:]
    changes: dict[str, tuple[str, str]] = {}

    def replace(path: str, old: str, new: str) -> None:
        source = subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "cat",
                image_ref,
                "/src/" + path,
            ],
            text=True,
        )
        if source.count(old) != 1:
            raise RuntimeError(f"Expected one pinned probe insertion point in {path}")
        changes[path] = (source, source.replace(old, new))

    aot = "python/sglang/kernels/aot/"
    replace(
        aot + "CMakeLists.txt",
        "set(SOURCES\n",
        "set(SOURCES\n    csrc/pareton_probe.cu\n",
    )
    replace(
        aot + "csrc/common_extension.cc",
        "TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {\n",
        "at::Tensor pareton_add_seven(const at::Tensor& x);\n\n"
        "TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {\n"
        '  m.def("pareton_add_seven(Tensor x) -> Tensor");\n'
        '  m.impl("pareton_add_seven", torch::kCUDA, &pareton_add_seven);\n',
    )
    replace(
        "rust/sglang-radix-tree/src/python_bindings.rs",
        "fn mem_cache(m: &Bound<'_, PyModule>) -> PyResult<()> {\n",
        "fn mem_cache(m: &Bound<'_, PyModule>) -> PyResult<()> {\n"
        '    m.add("PARETON_NATIVE_PROBE", 41)?;\n',
    )
    changes[aot + "csrc/pareton_probe.cu"] = ("", CUDA_SOURCE)
    changes["python/sglang/kernels/jit/csrc/pareton_probe.cuh"] = ("", JIT_SOURCE)
    changes["python/sglang/pareton_build_probe.py"] = ("", PYTHON_SOURCE)
    patch = []
    for path, (before, after) in sorted(changes.items()):
        patch.append(f"diff --git a/{path} b/{path}\n")
        if not before:
            patch.append("new file mode 100644\n")
        patch.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}" if before else "/dev/null",
                tofile=f"b/{path}",
            )
        )
    Path(output).write_text("".join(patch))


if __name__ == "__main__":
    main()
