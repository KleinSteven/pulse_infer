#pragma once

#include <cuda_runtime.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::kernels::cuda {

Result<void> embedding_cuda_launch(const void* input,
                                   const void* weight,
                                   void* output,
                                   i32 num_tokens,
                                   i32 vocab_size,
                                   i32 embedding_dim,
                                   DataType dtype,
                                   cudaStream_t stream = nullptr);

}  // namespace pulse::kernels::cuda
