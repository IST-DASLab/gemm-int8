# INT8 GEMM with PyTorch Interface

## Usage

Insall the kernels using the following commands:

```bash
git clone https://github.com/IST-DASLab/gemm_int8.git
cd gemm_int8
pip install -e .  # or pip install .
```

Then, the kernel can be used as follows:

```python
import torch
import gemm_int8
y = gemm_int8.matmul(a, b, alpha=1.0)
```

where `a` and `b` are the input matrices (in `torch.int8` format) and `alpha` is the scaling factor (in `float`).

## Benchmark

Run the following command to benchmark the kernel:

```bash
python benchmark.py
```