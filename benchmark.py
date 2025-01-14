import torch
import int8_ada
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Iterable, List, Tuple

matplotlib.rcParams["lines.linewidth"] = 2 * matplotlib.rcParams["lines.linewidth"]
matplotlib.rcParams["lines.markersize"] = 2 * matplotlib.rcParams["lines.markersize"]
matplotlib.rcParams.update({"font.size": 2 * matplotlib.rcParams["font.size"]})

iters = 100
warmup = 10

def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda').contiguous() * 5
    b = torch.randn((n, k), device='cuda').contiguous() * 5

    if dtype == torch.int8:
        return to_int8(a), to_int8(b)
    if dtype == torch.bfloat16:
        return a.to(torch.bfloat16), b.to(torch.bfloat16)
    if dtype == torch.float16:
        return a.half(), b.half()
    if dtype == torch.float32:
        return a.float(), b.float()

    raise ValueError("unsupported dtype")



# bench
def bench_fn(fn: Callable, *args, **kwargs) -> Tuple:

    times_ = []
    for i in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    for _ in range(10):
        start = time.time()
        for i in range(iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        times_.append((time.time() - start) * 1000 / iters)

    return np.mean(np.array(times_)), np.std(np.array(times_))


matrix_sizes = [
    (4096, 4096),
    (14336, 4096),
    (4096, 14336),
]

tokens = [512, 1024, 2048]

x_labels = []
bf16_runtimes = []
int8_runtimes = []
int8_quik_runtimes = []

for token in tokens:
    print('------------------')
    print(f"Token: {token}")
    for (n, k) in matrix_sizes:
        print(f"Matrix size: {k}x{n}")
        x_labels.append(f"{k}x{n}")
        a, b = make_rand_tensors(torch.bfloat16, token, n, k)
        a_int8, b_int8 = make_rand_tensors(torch.int8, token, n, k)

        bf16_times, bf16_times_std = bench_fn(torch.matmul, a, b.t())
        v_1_times, v_1_times_std = bench_fn(int8_ada.int8_matmul_v1, a_int8, b_int8, 1.0)
        quik_times, quik_times_std = bench_fn(int8_ada.int8_matmul_quik, a_int8, b_int8)

        print(f'Speedup (V1): {bf16_times/v_1_times:.2f}x')
        #print(f'Speedup (Quik): {bf16_times/quik_times:.2f}x')

        int8_runtimes.append(v_1_times.item())
        int8_quik_runtimes.append(quik_times.item())
        bf16_runtimes.append(bf16_times.item())

print(bf16_runtimes)
print(int8_runtimes)

""" for layer in range(len(matrix_sizes)):
    plt.plot(
        x_labels[(layer*len(tokens)):(layer*len(tokens))+len(tokens)],
        np.array(bf16_runtimes[(layer*len(tokens)):(layer*len(tokens))+len(tokens)])/np.array(int8_runtimes[(layer*len(tokens)):(layer*len(tokens))+len(tokens)]),
        'o-', label=f"Layer shape: {matrix_sizes[layer]}")

plt.axhline(1, color='black', linestyle='--')
plt.ylabel("Speedup (over BF16)")
plt.xlabel("M-dim")
plt.title(f'{torch.cuda.get_device_name()}')
plt.legend()
plt.savefig("int8_bf16_benchmark.png") """

sns.set()
plt.figure(figsize=(15, 10))
for token_id in range(len(tokens)):
    plt.plot(
        x_labels[(token_id*len(matrix_sizes)):(token_id*len(matrix_sizes))+len(matrix_sizes)],
        np.array(bf16_runtimes[(token_id*len(matrix_sizes)):(token_id*len(matrix_sizes))+len(matrix_sizes)])/np.array(int8_runtimes[(token_id*len(matrix_sizes)):(token_id*len(matrix_sizes))+len(matrix_sizes)]),
        'o-', label=f"Token Dim: {tokens[token_id]}")
plt.plot(x_labels, np.ones(len(x_labels))*4, "k")
plt.axhline(1, color='black', linestyle='--')
plt.ylabel("Speedup (over BF16)")
plt.xlabel("Matrix Dimensions (k x n)")
plt.title(f'{torch.cuda.get_device_name()}')
plt.legend()
plt.yticks(np.arange(1, 4.1, 0.25))

plt.tight_layout()
plt.savefig("int8_bf16_benchmark.png")