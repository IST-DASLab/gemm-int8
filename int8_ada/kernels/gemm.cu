#include <gemm.h>
#include <cutlass/float8.h>
#include "cutlass/float8.h"


#include <stddef.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>


#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>


torch::Tensor int8MatmulQuikCUDA(const torch::Tensor &A, const torch::Tensor &B) {
    torch::checkAllSameGPU("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int32_t,                         // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {C.data_ptr<int32_t>(), N},
      {C.data_ptr<int32_t>(), N},
      {2.0, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor int8_matmul_v1(
                                  torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  torch::Tensor out,   // INT32
                                  float alpha          // FP32
){
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TileShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();

  cutlass::gemm::GemmCoord problem_size(M, N, K);


  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      static_cast<ElementInputA*>(input.data_ptr()),
      LayoutInputA::packed(input_size));

  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      //weight.data_ptr<ElementInputB>(),
      static_cast<ElementInputB*>(weight.data_ptr()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      //out.data_ptr<ElementOutput>(),
      static_cast<ElementOutput*>(out.data_ptr()),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, 0.0}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}

torch::Tensor int8_matmul_host_v1(
                                  torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  torch::Tensor out,   // INT32
                                  float alpha          // FP32
){
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  if (M==512 && N==4096 && K==4096){
    using TileShape = typename cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 128>;
    static const int kStages = 3;
    return int8_matmul_v1<TileShape, WarpShape, kStages>(input, weight, out, alpha);
  } else if (M==512 && N==4096 && K==14336){
    using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    static const int kStages = 4;
    return int8_matmul_v1<TileShape, WarpShape, kStages>(input, weight, out, alpha);
  } else if (K==4096 && N==4096){
    using TileShape = typename cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    static const int kStages = 3;
    return int8_matmul_v1<TileShape, WarpShape, kStages>(input, weight, out, alpha);
  } else if (M==1024 && N==14336 && K==4096){
    using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    static const int kStages = 3;
    return int8_matmul_v1<TileShape, WarpShape, kStages>(input, weight, out, alpha);
  } else {
    using TileShape = typename cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    static const int kStages = 3;
    return int8_matmul_v1<TileShape, WarpShape, kStages>(input, weight, out, alpha);
  }
}