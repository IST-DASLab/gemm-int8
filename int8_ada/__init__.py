import torch
import int8_ada._CUDA

__all__ =  [ 
           "int8_matmul_v1",
           "int8_matmul_quik"
           ]
def int8_matmul_quik(x: torch.Tensor, y: torch.Tensor):
    return int8_ada._CUDA.int8_matmul_quik(x, y)

def int8_matmul_v1(x: torch.Tensor,
               y: torch.Tensor,
               alpha: float = 1.0):
    return int8_ada._CUDA.int8_matmul_v1(x, y, alpha)