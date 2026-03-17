#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::kernels::cuda {

Result<void> matmul_cuda_launch(const void* input1,
                                const void* input2,
                                void* output,
                                i32 m,
                                i32 n,
                                i32 k,
                                DataType dtype,
                                bool transpose_input1 = false,
                                bool transpose_input2 = false,
                                cudaStream_t stream = nullptr);

}  // namespace pulse::kernels::cuda
