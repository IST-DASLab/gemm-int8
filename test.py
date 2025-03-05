import torch
import gemm_int8


x = torch.rand(32, 32).cuda() * 100
x = x.to(torch.int8)

y_torch = x.bfloat16() @ x.bfloat16().t()
y_int8 = gemm_int8.matmul(x, x, 1.0)

print("Testing opcheck...")
torch.library.opcheck(torch.ops.gemm_int8_CUDA.int8_matmul.default, (x, x, 1.0))

print("Testing assert_close of torch matmul vs gemm_int8...")
torch.testing.assert_close(y_torch, y_int8)


@torch.compile(dynamic=True)
def test_gemm_int8(x, y, alpha):
    return gemm_int8.matmul(x, y, alpha)


y_int8 = test_gemm_int8(x, x, 1.0)
print("Testing compile of gemm_int8...")
torch.testing.assert_close(y_torch, y_int8)
