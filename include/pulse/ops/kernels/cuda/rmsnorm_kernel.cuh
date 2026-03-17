#pragma once

#include <cuda_runtime.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::kernels::cuda {

Result<void> rms_norm_cuda_launch(const void* input,
                                  const void* weight,
                                  void* output,
                                  i32 rows,
                                  i32 normalized_size,
                                  f64 eps,
                                  DataType dtype,
                                  cudaStream_t stream = nullptr);

}  // namespace pulse::kernels::cuda
