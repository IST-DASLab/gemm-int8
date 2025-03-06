from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import subprocess
import sys
import shutil
import pathlib
import torch

setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent

min_cuda_version = (11, 8)


# Read README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


def get_cuda_arch_flags():
    return [
        "-gencode", "arch=compute_80,code=sm_80",  # Ampere
        "-gencode", "arch=compute_86,code=sm_86",  # Ampere
        "-gencode", "arch=compute_89,code=sm_89",  # Ada
        "-gencode", "arch=compute_90,code=sm_90",  # Hopper
        "--expt-relaxed-constexpr",
    ]


def third_party_cmake():
    cmake = shutil.which("cmake")
    print(f"cmake: {cmake}")
    if cmake is None:
        raise RuntimeError("Cannot find CMake executable. Please install CMake (pip install cmake).")
    
    ninja = shutil.which("ninja")
    print(f"ninja: {ninja}")
    if ninja is None:
        raise RuntimeError("Cannot find Ninja executable. Please install Ninja (pip install ninja).")

    build_dir = HERE / "build"
    build_dir.mkdir(exist_ok=True)

    retcode = subprocess.call([
        cmake,
        "-B", str(build_dir),
        "-S", str(HERE),
        "-G", "Ninja",
    ])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)


class CustomBuildExtension(BuildExtension):
    """Custom build extension that verifies CUDA compatibility before building."""

    def run(self):
        print(f"CUDA version: {torch.version.cuda}")
        cuda_version = tuple(map(int, torch.version.cuda.split(".")))
        assert cuda_version >= min_cuda_version, (
            f"CUDA version must be >= {min_cuda_version}, yours is {torch.version.cuda}"
        )
        third_party_cmake()
        remove_unwanted_pytorch_nvcc_flags()
        super().run()


# The actual setup call, using the ext_modules for CUDA extensions
setup(
    ext_modules=[
        CUDAExtension(
            name="gemm_int8.gemm_int8_CUDA",
            sources=[
                "gemm_int8/kernels/bindings.cpp",
                "gemm_int8/kernels/gemm.cu",
            ],
            include_dirs=[
                os.path.join(setup_dir, "gemm_int8/kernels/include"),
                os.path.join(setup_dir, "cutlass/include"),
                os.path.join(setup_dir, "cutlass/tools/util/include"),
            ],
            extra_compile_args={
                "cxx": [],
                "nvcc": get_cuda_arch_flags(),
            },
        )
    ],
    cmdclass={"build_ext": CustomBuildExtension},
)
