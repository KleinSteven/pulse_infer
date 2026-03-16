#pragma once

#include <cuda_runtime.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::kernels::cuda {

Result<void> silu_cuda_launch(const void* input,
                              void* output,
                              i64 size,
                              DataType dtype,
                              cudaStream_t stream = nullptr);

}  // namespace pulse::kernels::cuda
