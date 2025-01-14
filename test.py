import torch
import int8_ada

x = torch.rand(32, 32).cuda()*100
x = x.to(torch.int8)

y_torch = x.float()@x.float().t()
y_int8 = int8_ada.int8_matmul_v1(x, x, 1.0)
print(y_torch)
print(y_int8)
print(torch.allclose(y_torch.to(torch.bfloat16), y_int8, atol=1e-2))