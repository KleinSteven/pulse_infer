#pragma once

#include <cuda_runtime.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::kernels::cuda {

Result<void> rope_cuda_launch(const void* input,
                              void* output,
                              i32 rows,
                              i32 seq_len,
                              i32 head_dim,
                              i32 rotary_dim,
                              i32 position_offset,
                              f32 theta,
                              DataType dtype,
                              cudaStream_t stream = nullptr);

Result<void> rope_positions_cuda_launch(const void* input,
                                        void* output,
                                        const i32* positions,
                                        i32 batch_size,
                                        i32 rows_per_batch,
                                        i32 head_dim,
                                        i32 rotary_dim,
                                        f32 theta,
                                        DataType dtype,
                                        cudaStream_t stream = nullptr);

}  // namespace pulse::kernels::cuda
