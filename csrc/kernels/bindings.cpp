#include <torch/extension.h>
#include <gemm.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <utility> // For std::pair

torch::Tensor int8_matmul(const torch::Tensor &A,
                          const torch::Tensor &B,
                          double alpha)
{
    float alpha_f = static_cast<float>(alpha);
    torch::checkAllContiguous("int8_matmul", {{A, "A", 0},
                                              {B, "B", 1}});
    torch::checkDeviceType("int8_matmul", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("int8_matmul", {{A, "A", 0},
                                           {B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    return int8_matmul_host(A, B, C, alpha_f);
}

//====== pybind ======

TORCH_LIBRARY(gemm_int8_CUDA, m)
{
    m.def("int8_matmul(Tensor A, Tensor B, float alpha) -> Tensor");
}

TORCH_LIBRARY_IMPL(gemm_int8_CUDA, CUDA, m)
{
    m.impl("int8_matmul", &int8_matmul);
}
