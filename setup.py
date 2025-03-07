from setuptools import setup
import os
import platform
import pathlib
import torch
import sys
from setuptools.command.bdist_wheel import bdist_wheel

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

def get_platform_tag(architecture=None):
    """Determine the platform tag for the wheel."""
    if architecture is None:
        architecture = platform.machine()
        
    system = platform.system()

    if system == "Linux":
        tag = "manylinux_2_24_x86_64" if architecture == "x86_64" else "manylinux_2_24_aarch64"
    elif system == "Darwin":
        tag = "macosx_13_1_x86_64" if architecture == "x86_64" else "macosx_13_1_arm64"
    elif system == "Windows":
        tag = "win_amd64" if architecture == "x86_64" else "win_arm64"
    else:
        raise ValueError(f"Unsupported system: {system}")

    return tag

class BdistWheelCommand(bdist_wheel):
    """Custom wheel building command to set platform tags correctly."""
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # Mark the wheel as platform-specific (not "any")
        self.root_is_pure = False
        
    def get_tag(self):
        python_tag = "py3"
        
        platform_tag = get_platform_tag()
        
        # Force the ABI tag to be 'none' since we're not using Python C API directly
        # (PyTorch's C++ extensions handle this for us)
        abi_tag = 'none'
        
        return python_tag, abi_tag, platform_tag

if __name__ == "__main__":
    # Read README for the long description
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    check_cuda_version()
    
    print(f"Building wheel with platform tag: {get_platform_tag()}")

    # The actual setup call without ext_modules
    setup(
        # All package configuration is now in pyproject.toml
        package_data={"gemm_int8": ["*.so"]},  # Include compiled libraries
        cmdclass={
            'bdist_wheel': BdistWheelCommand,
        },
    )
