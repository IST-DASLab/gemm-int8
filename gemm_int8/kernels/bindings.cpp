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
                             float alpha)
{
    torch::checkAllContiguous("int8_matmul", {{A, "A",       0},
                                                {B, "B", 1}});
    torch::checkDeviceType("int8_matmul", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("int8_matmul", {{A, "A",       0},
                                          {   B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    return int8_matmul_host(A, B, C, alpha);
}




//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{
    m.def("int8_matmul", &int8_matmul,
        "int8_matmul");

}