import torch
import os
import glob

package_dir = os.path.dirname(os.path.abspath(__file__))

lib_pattern = os.path.join(package_dir, "gemm_int8_CUDA*.so")
lib_files = glob.glob(lib_pattern)
if not lib_files:
    raise ImportError(f"Could not find compiled CUDA extension in {package_dir}")
    
for lib_file in lib_files:
    torch.ops.load_library(lib_file)


@torch.library.register_fake("gemm_int8_CUDA::int8_matmul")
def _(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    torch._check(y.device.type == "cuda", "y must be a CUDA tensor")
    torch._check(x.dtype == torch.int8, "x must be an int8 tensor")
    torch._check(y.dtype == torch.int8, "y must be an int8 tensor")
    torch._check(len(x.shape) == 2, "x must be a 2D tensor")
    torch._check(len(y.shape) == 2, "y must be a 2D tensor")
    torch._check(x.shape[1] == y.shape[1], "x.shape[1] must be equal to y.shape[1]")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


def matmul(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Matrix-Matrix Multiplication for INT8 data type in the form of (x @ y.t())*alpha.
    The output is BF16 data type. todo: support arbitrary output dtype!
    Argumengs:
        x: torch.Tensor, shape (M, K)
        y: torch.Tensor, shape (K, N)
        alpha: float, which is multiplied by the output (default=1.0)
    """
    return torch.ops.gemm_int8_CUDA.int8_matmul(x, y, alpha)


__all__ = ["matmul"]
