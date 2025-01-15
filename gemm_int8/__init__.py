import torch
import gemm_int8._CUDA


def matmul(x: torch.Tensor,
           y: torch.Tensor,
           alpha: float = 1.0):
    '''
    Matrix-Matrix Multiplication for INT8 data type in the form of (x @ y.t())*alpha.
    The output is BF16 data type. todo: support arbitrary output dtype!
    Argumengs:
        x: torch.Tensor, shape (M, K)
        y: torch.Tensor, shape (K, N)
        alpha: float, which is multiplied by the output (default=1.0)
        fastAcc: bool, (default=True)
    '''
    return gemm_int8._CUDA.int8_matmul(x, y, alpha)


__all__ =  ["matmul"]