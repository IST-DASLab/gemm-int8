# INT8 GEMM with PyTorch Interface

<!-- [![PyPI version](https://badge.fury.io/py/gemm-int8.svg)](https://badge.fury.io/py/gemm-int8) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
<!-- [![GitHub stars](https://img.shields.io/github/stars/IST-DASLab/gemm-int8.svg)](https://github.com/IST-DASLab/gemm-int8/stargazers) -->
<!-- [![GitHub issues](https://img.shields.io/github/issues/IST-DASLab/gemm-int8.svg)](https://github.com/IST-DASLab/gemm-int8/issues) -->

A PyTorch CUDA extension providing high-performance INT8 matrix multiplication operations utilizing CUTLASS iterators. Specifically optimized for modern NVIDIA GPUs including Ada Lovelace and Hopper architectures, this library offers measurable performance improvements over standard BF16 matrix multiplication in deep learning applications. (It was originally used in [HALO: Hadamard-Assisted Low-Precision Optimization and Training method for finetuning LLMs](https://github.com/IST-DASLab/HALO))

## Features

- INT8 matrix multiplication with PyTorch integration, providing up to 4x speedup on RTX 4090 GPUs
- Compatible with PyTorch's torch.compile (autograd not supported)
- Optimized CUDA kernels for compute capabilities 89-100 (Ada Lovelace, Hopper)
- Tuned kernel configurations for common matrix dimensions in transformer models
- Direct integration with existing PyTorch workflows

## Quick Start

```bash
# Install from GitHub releases
pip install https://github.com/IST-DASLab/gemm-int8/releases/download/latest/gemm_int8-1.0.0-py3-none-manylinux_2_24_x86_64.whl
```

```python
import torch
import gemm_int8

# Create input tensors
a = torch.randint(-128, 127, (1024, 4096), device='cuda', dtype=torch.int8)
b = torch.randint(-128, 127, (4096, 4096), device='cuda', dtype=torch.int8)

# Perform INT8 matrix multiplication (compute a @ b.t())
result = gemm_int8.matmul(a, b, alpha=1.0)  # Returns bfloat16 tensor of (a @ b.t()) * alpha
```

Performs matrix multiplication in the form of `(x @ y.t()) * alpha`.

**Parameters:**
- `x` (torch.Tensor): Input matrix of shape (M, K) with dtype torch.int8
- `y` (torch.Tensor): Input matrix of shape (N, K) with dtype torch.int8
- `alpha` (float, optional): Scaling factor applied to the output. Default: 1.0

**Returns:**
- torch.Tensor: Result matrix of shape (M, N) with dtype torch.bfloat16

## Requirements

- Python 3.9+
- PyTorch 2.0.0+
- CUDA 11.8+
- NVIDIA GPU with Compute Capability 70 or higher
- Linux with x86_64 architecture (primary platform)

## Installation

### Option 1: From PyPI (Coming Soon)

```bash
pip install gemm-int8
```

### Option 2: From GitHub Release

Download pre-built wheels directly from the GitHub releases page:

```bash
pip install https://github.com/IST-DASLab/gemm-int8/releases/download/v$(VERSION)/gemm_int8-$(VERSION)-py3-none-$(PLATFORM_TAG).whl
```

Where:
- `$(VERSION)` is the package version (e.g., "1.0.0")
- `$(PLATFORM_TAG)` is your platform tag (e.g., "manylinux_2_24_x86_64")

Or to install the latest build from the main branch:

```bash
pip install https://github.com/IST-DASLab/gemm-int8/releases/download/latest/gemm_int8-$(VERSION)-py3-none-$(PLATFORM_TAG).whl
```

### Option 3: Build From Source

Building from source requires additional development tools:

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/IST-DASLab/gemm-int8.git
cd gemm-int8

# Make sure CUDA toolkit is properly installed and CUDA_HOME is set
echo $CUDA_HOME  # Should point to your CUDA installation directory
# If not set, you may need to run: export CUDA_HOME=/usr/local/cuda

# Also make sure you hace cmake and ninja installed in your environment.
pip install cmake ninja

# Build and install
./build.sh
pip install .

# Alternatively, for development installation
pip install -e .
```


### Integration with torch.compile

The library is compatible with PyTorch's `torch.compile` i.e. if this code is used within a compiled scope:

```python
import torch
import gemm_int8

@torch.compile(dynamic=True)
def compiled_matmul_routine(x, y, alpha):
    # ... some pytorch operations
    res = gemm_int8.matmul(x, y, alpha)
    # ... some pytorch operations
    return res

# Use the compiled function
result = compiled_matmul_routine(a, b, 1.0)
```

Note that compile won't optimize this kernel and it's only compatible in the sense that torch compile backend will recognize it as an operator and can be compiled along other operations in a routine.

## Benchmarks

You can run the benchmark script to compare performance:

```bash
python benchmark.py
```

This will generate a benchmark report and a visualization showing the speedup compared to BF16 matrix multiplication across different matrix sizes and token dimensions.

Typical speedups range from 2x to 4x depending on the matrix dimensions and hardware.

## Performance Tips

- For best performance, ensure your tensors are contiguous in memory
- The library is optimized for large matrix sizes commonly found in transformer models
- Performance benefits are most significant for matrix dimensions commonly used in LLM inference

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{gemm_int8,
  author = {Roberto L. Castro and Saleh Ashkboos and Soroush Tabesh},
  title = {INT8 GEMM with PyTorch Interface},
  url = {https://github.com/IST-DASLab/gemm-int8},
  year = {2024},
}
```

```bibtex
@article{halo2025,
      title={HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs}, 
      author={Saleh Ashkboos and Mahdi Nikdan and Soroush Tabesh and Roberto L. Castro and Torsten Hoefler and Dan Alistarh},
      year={2025},
      eprint={2501.02625},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.02625}, 
}
```

## Acknowledgements

This project uses [CUTLASS](https://github.com/NVIDIA/cutlass) for optimized CUDA kernels.
