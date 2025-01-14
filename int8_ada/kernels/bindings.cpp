#include <torch/extension.h>
#include <gemm.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <utility> // For std::pair




torch::Tensor int8_matmul_v1(const torch::Tensor &A,
                                                const torch::Tensor &B,
                                                float alpha)
{
    torch::checkAllContiguous("int8_matmul_v1", {{A, "A",       0},
                                                {B, "B", 1}});
    torch::checkDeviceType("int8_matmul_v1", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("int8_matmul_v1", {{A, "A",       0},
                                          {   B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

    return int8_matmul_host_v1(A, B, C, alpha);
}

torch::Tensor int8_matmul_quik(const torch::Tensor &A,
                                                const torch::Tensor &B)
{
    return int8MatmulQuikCUDA(A, B);
}



//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{
    m.def("int8_matmul_v1", &int8_matmul_v1,
        "int8_matmul_v1 (same as QLLMT)");

    m.def("int8_matmul_quik", &int8_matmul_quik,
        "int8_matmul_quik (same as QUIK)");

}