import torch
import gemm_int8

x = torch.rand(32, 32).cuda()*100
x = x.to(torch.int8)

y_torch = x.float()@x.float().t()
y_int8 = gemm_int8.matmul(x, x, 1.0)
print(y_torch)
print(y_int8)
print(torch.allclose(y_torch.to(torch.bfloat16), y_int8, atol=1e-2))