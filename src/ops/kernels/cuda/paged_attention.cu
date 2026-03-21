#include "pulse/ops/kernels/cuda/paged_attention.cuh"

#include <algorithm>
#include <cfloat>
#include <type_traits>

#include <cub/cub.cuh>

#include "pulse/logging.hpp"

namespace pulse::kernels::cuda {

namespace {

template<typename T>
__device__ float to_float(T value) {
    if constexpr (std::is_same_v<T, f16>) {
        return __half2float(value);
    } else if constexpr (std::is_same_v<T, bf16>) {
        return __bfloat162float(value);
    } else {
        return static_cast<float>(value);
    }
}

template<typename T>
__device__ T from_float(float value) {
    if constexpr (std::is_same_v<T, f16>) {
        return __float2half(value);
    } else if constexpr (std::is_same_v<T, bf16>) {
        return __float2bfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

template<typename T, int THREADS>
__global__ void paged_attention_kernel(const T* query,
                                       const T* key_cache,
                                       const T* value_cache,
                                       const i32* block_table,
                                       const i32* context_lens,
                                       T* output,
                                       i32 num_heads,
                                       i32 num_kv_heads,
                                       i32 head_size,
                                       i32 block_size,
                                       i32 max_blocks_per_seq) {
    using BlockReduce = cub::BlockReduce<float, THREADS>;

    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ float shared_dot;
    __shared__ float shared_running_max;
    __shared__ float shared_running_sum;
    __shared__ float shared_alpha;
    __shared__ float shared_beta;
    extern __shared__ float shared_storage[];
    float* shared_query = shared_storage;
    float* shared_acc = shared_storage + head_size;

    const i32 head_idx = blockIdx.x;
    const i32 token_idx = blockIdx.y;
    const i32 tid = threadIdx.x;

    const i32 context_len = context_lens[token_idx];
    if (context_len <= 0) {
        return;
    }

    const i32 kv_mul = num_heads / num_kv_heads;
    const i32 kv_head_idx = head_idx / kv_mul;

    const i64 query_offset =
        (static_cast<i64>(token_idx) * num_heads + head_idx) * static_cast<i64>(head_size);
    for (i32 dim = tid; dim < head_size; dim += THREADS) {
        shared_query[dim] = to_float(query[query_offset + dim]);
        shared_acc[dim] = 0.0f;
    }
    if (tid == 0) {
        shared_dot = 0.0f;
        shared_running_max = -FLT_MAX;
        shared_running_sum = 0.0f;
        shared_alpha = 0.0f;
        shared_beta = 0.0f;
    }
    __syncthreads();

    for (i32 step = 0; step < context_len; ++step) {
        const i32 logical_block = step / block_size;
        const i32 block_offset = step % block_size;
        const i32 physical_block = block_table[token_idx * max_blocks_per_seq + logical_block];
        if (physical_block < 0) {
            continue;
        }

        const i64 cache_offset =
            (((static_cast<i64>(physical_block) * num_kv_heads + kv_head_idx) * block_size) + block_offset) *
            static_cast<i64>(head_size);

        float local_dot = 0.0f;
        for (i32 dim = tid; dim < head_size; dim += THREADS) {
            local_dot += shared_query[dim] * to_float(key_cache[cache_offset + dim]);
        }

        const float dot = BlockReduce(reduce_storage).Sum(local_dot);
        if (tid == 0) {
            shared_dot = dot;
            const float score = shared_dot * rsqrtf(static_cast<float>(head_size));
            const float new_max = fmaxf(shared_running_max, score);
            shared_alpha = shared_running_max == -FLT_MAX ? 0.0f : expf(shared_running_max - new_max);
            shared_beta = expf(score - new_max);
            shared_running_sum = shared_running_sum * shared_alpha + shared_beta;
            shared_running_max = new_max;
        }
        __syncthreads();

        for (i32 dim = tid; dim < head_size; dim += THREADS) {
            shared_acc[dim] =
                shared_acc[dim] * shared_alpha + shared_beta * to_float(value_cache[cache_offset + dim]);
        }
        __syncthreads();
    }

    if (shared_running_sum <= 0.0f) {
        for (i32 dim = tid; dim < head_size; dim += THREADS) {
            const i64 output_offset =
                (static_cast<i64>(token_idx) * num_heads + head_idx) * static_cast<i64>(head_size);
            output[output_offset + dim] = from_float<T>(0.0f);
        }
        return;
    }

    const i64 output_offset =
        (static_cast<i64>(token_idx) * num_heads + head_idx) * static_cast<i64>(head_size);
    for (i32 dim = tid; dim < head_size; dim += THREADS) {
        output[output_offset + dim] = from_float<T>(shared_acc[dim] / shared_running_sum);
    }
}

}  // namespace

Result<void> paged_attention_cuda_launch(const void* query,
                                         const void* key_cache,
                                         const void* value_cache,
                                         const i32* block_table,
                                         const i32* context_lens,
                                         void* output,
                                         i32 total_tokens,
                                         i32 num_heads,
                                         i32 num_kv_heads,
                                         i32 head_size,
                                         i32 block_size,
                                         i32 max_blocks_per_seq,
                                         DataType dtype,
                                         cudaStream_t stream) {
    if (total_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_size <= 0 || block_size <= 0 ||
        max_blocks_per_seq <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "Invalid paged attention launch dimensions");
    }

    if (num_heads % num_kv_heads != 0) {
        return Err<void>(ErrorCode::InvalidArgument, "Paged attention requires num_heads divisible by num_kv_heads");
    }

    constexpr i32 thread_num = 128;
    dim3 grid(num_heads, total_tokens);
    const usize shared_mem_bytes = static_cast<usize>(head_size) * sizeof(float) * 2;

    switch (dtype) {
        case DataType::Float16:
            if (stream != nullptr) {
                paged_attention_kernel<f16, thread_num><<<grid, thread_num, shared_mem_bytes, stream>>>(
                    static_cast<const f16*>(query),
                    static_cast<const f16*>(key_cache),
                    static_cast<const f16*>(value_cache),
                    block_table,
                    context_lens,
                    static_cast<f16*>(output),
                    num_heads,
                    num_kv_heads,
                    head_size,
                    block_size,
                    max_blocks_per_seq);
            } else {
                paged_attention_kernel<f16, thread_num><<<grid, thread_num, shared_mem_bytes>>>(
                    static_cast<const f16*>(query),
                    static_cast<const f16*>(key_cache),
                    static_cast<const f16*>(value_cache),
                    block_table,
                    context_lens,
                    static_cast<f16*>(output),
                    num_heads,
                    num_kv_heads,
                    head_size,
                    block_size,
                    max_blocks_per_seq);
            }
            break;
        case DataType::BFloat16:
            if (stream != nullptr) {
                paged_attention_kernel<bf16, thread_num><<<grid, thread_num, shared_mem_bytes, stream>>>(
                    static_cast<const bf16*>(query),
                    static_cast<const bf16*>(key_cache),
                    static_cast<const bf16*>(value_cache),
                    block_table,
                    context_lens,
                    static_cast<bf16*>(output),
                    num_heads,
                    num_kv_heads,
                    head_size,
                    block_size,
                    max_blocks_per_seq);
            } else {
                paged_attention_kernel<bf16, thread_num><<<grid, thread_num, shared_mem_bytes>>>(
                    static_cast<const bf16*>(query),
                    static_cast<const bf16*>(key_cache),
                    static_cast<const bf16*>(value_cache),
                    block_table,
                    context_lens,
                    static_cast<bf16*>(output),
                    num_heads,
                    num_kv_heads,
                    head_size,
                    block_size,
                    max_blocks_per_seq);
            }
            break;
        case DataType::Float32:
            if (stream != nullptr) {
                paged_attention_kernel<f32, thread_num><<<grid, thread_num, shared_mem_bytes, stream>>>(
                    static_cast<const f32*>(query),
                    static_cast<const f32*>(key_cache),
                    static_cast<const f32*>(value_cache),
                    block_table,
                    context_lens,
                    static_cast<f32*>(output),
                    num_heads,
                    num_kv_heads,
                    head_size,
                    block_size,
                    max_blocks_per_seq);
            } else {
                paged_attention_kernel<f32, thread_num><<<grid, thread_num, shared_mem_bytes>>>(
                    static_cast<const f32*>(query),
                    static_cast<const f32*>(key_cache),
                    static_cast<const f32*>(value_cache),
                    block_table,
                    context_lens,
                    static_cast<f32*>(output),
                    num_heads,
                    num_kv_heads,
                    head_size,
                    block_size,
                    max_blocks_per_seq);
            }
            break;
        default:
            return Err<void>(ErrorCode::NotImplemented,
                             "Paged attention only supports Float16/BFloat16/Float32");
    }

    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        pulse::error("CUDA paged_attention launch failed: {}", cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA paged_attention launch failed: {}", cudaGetErrorString(err)));
    }
    return Ok();
}

}  // namespace pulse::kernels::cuda
