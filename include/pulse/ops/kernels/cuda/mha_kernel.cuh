#pragma once

#include <cuda_runtime.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::kernels::cuda {

Result<void> mha_cuda_launch(const void* query,
                             const void* key_cache,
                             const void* value_cache,
                             void* score,
                             void* output,
                             i32 pos,
                             i32 seq_len,
                             i32 kv_dim,
                             i32 head_num,
                             i32 head_size,
                             i32 kv_mul,
                             DataType dtype,
                             cudaStream_t stream = nullptr);

}  // namespace pulse::kernels::cuda
