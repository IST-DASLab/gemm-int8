from setuptools import setup
import os
import shutil
import pathlib
import torch

setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent

min_cuda_version = (11, 8)

def check_cuda_version():
    """Verify CUDA compatibility before building."""
    print(f"CUDA version: {torch.version.cuda}")
    cuda_version = tuple(map(int, torch.version.cuda.split(".")))
    assert cuda_version >= min_cuda_version, (
        f"CUDA version must be >= {min_cuda_version}, yours is {torch.version.cuda}"
    )

if __name__ == "__main__":
    # Read README for the long description
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    check_cuda_version()

    # The actual setup call without ext_modules
    setup(
        # All package configuration is now in pyproject.toml
        package_data={"gemm_int8": ["*.so"]},  # Include compiled libraries
    )
