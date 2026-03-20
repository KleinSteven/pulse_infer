#pragma once

#include <cuda_runtime.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::kernels::cuda {

Result<void> mul_cuda_launch(const void* input1,
                             const void* input2,
                             void* output,
                             i64 size,
                             DataType dtype,
                             cudaStream_t stream = nullptr);

}  // namespace pulse::kernels::cuda
