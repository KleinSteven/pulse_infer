#pragma once

#include <cuda_runtime.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::kernels::cuda {

Result<void> paged_kv_write_cuda_launch(const void* key,
                                        const void* value,
                                        void* key_cache,
                                        void* value_cache,
                                        const i32* block_table,
                                        const i32* positions,
                                        i32 total_tokens,
                                        i32 num_kv_heads,
                                        i32 head_size,
                                        i32 block_size,
                                        i32 max_blocks_per_seq,
                                        DataType dtype,
                                        cudaStream_t stream = nullptr);

}  // namespace pulse::kernels::cuda
