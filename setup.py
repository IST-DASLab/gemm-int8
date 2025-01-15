from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import pathlib, torch
setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent

def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

def get_cuda_arch_flags():
    return [
        # '-gencode', 'arch=compute_75,code=sm_75',  # Turing
        '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
        # '-gencode', 'arch=compute_86,code=sm_86',  # Ampere
        '-gencode', 'arch=compute_89,code=sm_89',  # Ada
        '--expt-relaxed-constexpr'
    ]
    
def third_party_cmake():
    import subprocess, sys, shutil
    
    cmake = shutil.which('cmake')
    if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

    retcode = subprocess.call([cmake, HERE])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)

if __name__ == '__main__':

    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.cuda.current_device()
    print(f"Current device: {torch.cuda.get_device_name(device)}")
    print(f"Current CUDA capability: {torch.cuda.get_device_capability(device)}")
    assert torch.cuda.get_device_capability(device)[0] >= 8, f"CUDA capability must be >= 8.0, yours is {torch.cuda.get_device_capability(device)}"

    third_party_cmake()
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name='gemm_int8',
        ext_modules=[
            CUDAExtension(
                name='gemm_int8._CUDA',
                sources=[
                    'gemm_int8/kernels/bindings.cpp',
                    'gemm_int8/kernels/gemm.cu',
                ],
                include_dirs=[
                    os.path.join(setup_dir, 'gemm_int8/kernels/include'),
                    os.path.join(setup_dir, 'cutlass/include'),
                    os.path.join(setup_dir, 'cutlass/tools/util/include')
                ],
                extra_compile_args={
                    'cxx': [],
                    'nvcc': get_cuda_arch_flags(),
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        version='0.1.0',
    )