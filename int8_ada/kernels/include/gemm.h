#pragma once
#include <common.h>
#include <torch/types.h>


// used by out_proj and fc2, return FP32
torch::Tensor int8_matmul_host_v1(torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  torch::Tensor out,   // INT32
                                  float alpha          // FP32
);

torch::Tensor int8MatmulQuikCUDA(const torch::Tensor &A, const torch::Tensor &B);