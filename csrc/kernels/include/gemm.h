#pragma once
#include <common.h>
#include <torch/types.h>


torch::Tensor int8_matmul_host(torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  torch::Tensor out,   // BF16
                                  float alpha          // FP32
);

