# INT8 GEMM with PyTorch Interface

<!-- [![PyPI version](https://badge.fury.io/py/gemm-int8.svg)](https://badge.fury.io/py/gemm-int8) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
<!-- [![GitHub stars](https://img.shields.io/github/stars/IST-DASLab/gemm-int8.svg)](https://github.com/IST-DASLab/gemm-int8/stargazers) -->
<!-- [![GitHub issues](https://img.shields.io/github/issues/IST-DASLab/gemm-int8.svg)](https://github.com/IST-DASLab/gemm-int8/issues) -->

A high-performance CUDA extension for PyTorch that provides optimized INT8 matrix multiplication operations. This library offers significant speedups over standard BF16 matrix multiplication in PyTorch, making it ideal for accelerating large matrix operations in deep learning applications.

## Features

- Fast INT8 matrix multiplication with PyTorch integration, up to 4x on RTX4090 
- Compatible with PyTorch's torch.compile (autograd is not supported on this operator)
- Supports CUDA-enabled GPUs (requires CUDA 11.8+ and SM>80)
- Provides significant speedups over standard BF16 matrix multiplication using int8 specific kernels
- Simple API that integrates seamlessly with existing PyTorch code

## Requirements

- Python 3.9+
- PyTorch 2.0.0+
- CUDA 11.8+ (supported versions: 11.8, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.8)
- NVIDIA GPU with Compute Capability > 80
- CMake 3.18.0+
- Linux with x86_64 architecture

## Installation

### From PyPI (Coming Soon)

```bash
pip install gemm-int8
```

### From GitHub Release

```bash
# Install the latest release wheel
pip install https://github.com/IST-DASLab/gemm-int8/releases/download/continuous-release_main/gemm_int8-1.0.0-py3-none-linux_x86_64.whl
```

### From Source

If you're installing from source, you'll need additional build dependencies:

```bash
# Clone the repository
git clone https://github.com/IST-DASLab/gemm-int8.git
cd gemm-int8

# Make sure cmake and ninja are installed in your environment
pip install -r requirements-build.txt

# Install the package
pip install -e .  # For development installation
# OR
pip install .     # For regular installation
```


## Usage

The library provides a simple API for INT8 matrix multiplication:

```python
import torch
import gemm_int8

# Create INT8 tensors
a = torch.randn(1024, 4096, device='cuda').clamp(-128, 127).to(torch.int8)
b = torch.randn(4096, 4096, device='cuda').clamp(-128, 127).to(torch.int8)

# Perform INT8 matrix multiplication
result = gemm_int8.matmul(a, b, alpha=1.0)

# The result is in BF16 format
print(result.dtype)  # torch.bfloat16
```

### API Reference

```python
gemm_int8.matmul(x, y, alpha=1.0)
```

Performs matrix multiplication in the form of `(x @ y.t()) * alpha`.

**Parameters:**
- `x` (torch.Tensor): Input matrix of shape (M, K) with dtype torch.int8
- `y` (torch.Tensor): Input matrix of shape (N, K) with dtype torch.int8
- `alpha` (float, optional): Scaling factor applied to the output. Default: 1.0

**Returns:**
- torch.Tensor: Result matrix of shape (M, N) with dtype torch.bfloat16

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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
