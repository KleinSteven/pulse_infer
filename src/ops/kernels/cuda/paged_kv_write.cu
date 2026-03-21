#include "pulse/ops/kernels/cuda/paged_kv_write.cuh"

#include <type_traits>

#include "pulse/logging.hpp"

#define LAUNCH_KERNEL(kernel, block_num, thread_num, stream, ...) \
  do { \
    if (stream) { \
      kernel<<<block_num, thread_num, 0, stream>>>(__VA_ARGS__); \
    } else { \
      kernel<<<block_num, thread_num>>>(__VA_ARGS__); \
    } \
  } while (0)

namespace pulse::kernels::cuda {

namespace {

template<typename T>
__global__ void paged_kv_write_kernel(const T* key,
                                      const T* value,
                                      T* key_cache,
                                      T* value_cache,
                                      const i32* block_table,
                                      const i32* positions,
                                      i32 num_kv_heads,
                                      i32 head_size,
                                      i32 block_size,
                                      i32 max_blocks_per_seq) {
    const i32 token_idx = blockIdx.x;
    const i32 kv_head_idx = blockIdx.y;
    const i32 tid = threadIdx.x;

    const i32 token_pos = positions[token_idx];
    const i32 logical_block = token_pos / block_size;
    const i32 block_offset = token_pos % block_size;
    const i32 physical_block = block_table[token_idx * max_blocks_per_seq + logical_block];
    if (physical_block < 0) {
        return;
    }

    const i64 src_offset =
        (static_cast<i64>(token_idx) * num_kv_heads + kv_head_idx) * static_cast<i64>(head_size);
    const i64 dst_offset =
        (((static_cast<i64>(physical_block) * num_kv_heads + kv_head_idx) * block_size) + block_offset) *
        static_cast<i64>(head_size);

    for (i32 dim = tid; dim < head_size; dim += blockDim.x) {
        key_cache[dst_offset + dim] = key[src_offset + dim];
        value_cache[dst_offset + dim] = value[src_offset + dim];
    }
}

}  // namespace

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
                                        cudaStream_t stream) {
    if (total_tokens <= 0 || num_kv_heads <= 0 || head_size <= 0 || block_size <= 0 || max_blocks_per_seq <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "Invalid paged KV write launch dimensions");
    }

    dim3 grid(total_tokens, num_kv_heads);
    const i32 thread_num = std::min(head_size, 256);

    switch (dtype) {
        case DataType::Float16:
            LAUNCH_KERNEL(paged_kv_write_kernel<f16>,
                          grid,
                          thread_num,
                          stream,
                          static_cast<const f16*>(key),
                          static_cast<const f16*>(value),
                          static_cast<f16*>(key_cache),
                          static_cast<f16*>(value_cache),
                          block_table,
                          positions,
                          num_kv_heads,
                          head_size,
                          block_size,
                          max_blocks_per_seq);
            break;
        case DataType::BFloat16:
            LAUNCH_KERNEL(paged_kv_write_kernel<bf16>,
                          grid,
                          thread_num,
                          stream,
                          static_cast<const bf16*>(key),
                          static_cast<const bf16*>(value),
                          static_cast<bf16*>(key_cache),
                          static_cast<bf16*>(value_cache),
                          block_table,
                          positions,
                          num_kv_heads,
                          head_size,
                          block_size,
                          max_blocks_per_seq);
            break;
        case DataType::Float32:
            LAUNCH_KERNEL(paged_kv_write_kernel<f32>,
                          grid,
                          thread_num,
                          stream,
                          static_cast<const f32*>(key),
                          static_cast<const f32*>(value),
                          static_cast<f32*>(key_cache),
                          static_cast<f32*>(value_cache),
                          block_table,
                          positions,
                          num_kv_heads,
                          head_size,
                          block_size,
                          max_blocks_per_seq);
            break;
        default:
            return Err<void>(ErrorCode::NotImplemented,
                             "Paged KV write only supports Float16/BFloat16/Float32");
    }

    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        pulse::error("CUDA paged_kv_write launch failed: {}", cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA paged_kv_write launch failed: {}", cudaGetErrorString(err)));
    }

    return Ok();
}

}  // namespace pulse::kernels::cuda
