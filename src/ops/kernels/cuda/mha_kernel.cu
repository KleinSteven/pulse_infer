#include "pulse/ops/kernels/cuda/mha_kernel.cuh"

#include <cfloat>
#include <type_traits>
#include <cub/cub.cuh>

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

constexpr i32 kThreadNum = 128;

__device__ void softmax_inplace(f32* score, i32 len) {
    using BlockReduce = cub::BlockReduce<f32, kThreadNum>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ f32 shared_value;

    const i32 tid = threadIdx.x;

    f32 local_max = tid < len ? score[tid] : -FLT_MAX;
    for (i32 index = tid + blockDim.x; index < len; index += blockDim.x) {
        local_max = max(local_max, score[index]);
    }

    const f32 max_value = BlockReduce(reduce_storage).Reduce(local_max, cub::Max());
    if (tid == 0) {
        shared_value = max_value;
    }
    __syncthreads();

    f32 local_sum = 0.0f;
    for (i32 index = tid; index < len; index += blockDim.x) {
        score[index] = __expf(score[index] - shared_value);
        local_sum += score[index];
    }
    __syncthreads();

    const f32 sum_value = BlockReduce(reduce_storage).Sum(local_sum);
    if (tid == 0) {
        shared_value = sum_value;
    }
    __syncthreads();

    for (i32 index = tid; index < len; index += blockDim.x) {
        score[index] /= shared_value;
    }
}

template<typename T>
__global__ void mha_kernel(const T* query,
                           const T* key_cache,
                           const T* value_cache,
                           f32* score,
                           T* output,
                           i32 pos,
                           i32 seq_len,
                           i32 kv_dim,
                           i32 kv_mul,
                           i32 head_num,
                           i32 head_size) {
    const i32 head = blockIdx.x;
    if (head >= head_num) {
        return;
    }

    const T* query_head = query + static_cast<i64>(head) * head_size;
    f32* score_head = score + static_cast<i64>(head) * seq_len;
    T* output_head = output + static_cast<i64>(head) * head_size;
    const i32 kv_offset = (head / kv_mul) * head_size;
    const f32 scale = rsqrtf(static_cast<f32>(head_size));

    for (i32 token = threadIdx.x; token <= pos; token += blockDim.x) {
        const T* key_head = key_cache + static_cast<i64>(token) * kv_dim + kv_offset;
        f32 dot = 0.0f;
        for (i32 dim = 0; dim < head_size; ++dim) {
            if constexpr (std::is_same_v<T, f32>) {
                dot += query_head[dim] * key_head[dim];
            } else if constexpr (std::is_same_v<T, f16>) {
                dot += __half2float(query_head[dim]) * __half2float(key_head[dim]);
            } else if constexpr (std::is_same_v<T, bf16>) {
                dot += __bfloat162float(query_head[dim]) * __bfloat162float(key_head[dim]);
            }
        }
        score_head[token] = dot * scale;
    }
    __syncthreads();

    softmax_inplace(score_head, pos + 1);
    __syncthreads();

    for (i32 dim = threadIdx.x; dim < head_size; dim += blockDim.x) {
        f32 sum = 0.0f;
        for (i32 token = 0; token <= pos; ++token) {
            const T* value_head = value_cache + static_cast<i64>(token) * kv_dim + kv_offset;
            if constexpr (std::is_same_v<T, f32>) {
                sum += score_head[token] * value_head[dim];
            } else if constexpr (std::is_same_v<T, f16>) {
                sum += score_head[token] * __half2float(value_head[dim]);
            } else if constexpr (std::is_same_v<T, bf16>) {
                sum += score_head[token] * __bfloat162float(value_head[dim]);
            }
        }

        if constexpr (std::is_same_v<T, f32>) {
            output_head[dim] = sum;
        } else if constexpr (std::is_same_v<T, f16>) {
            output_head[dim] = __float2half(sum);
        } else if constexpr (std::is_same_v<T, bf16>) {
            output_head[dim] = __float2bfloat16(sum);
        }
    }
}

}  // namespace

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
                             cudaStream_t stream) {
    if (pos < 0 || seq_len <= 0 || kv_dim <= 0 || head_num <= 0 || head_size <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "Invalid CUDA MHA launch dimensions");
    }

    switch (dtype) {
        case DataType::Float32:
            LAUNCH_KERNEL(mha_kernel<f32>,
                          head_num,
                          kThreadNum,
                          stream,
                          static_cast<const f32*>(query),
                          static_cast<const f32*>(key_cache),
                          static_cast<const f32*>(value_cache),
                          static_cast<f32*>(score),
                          static_cast<f32*>(output),
                          pos,
                          seq_len,
                          kv_dim,
                          kv_mul,
                          head_num,
                          head_size);
            break;
        case DataType::Float16:
            LAUNCH_KERNEL(mha_kernel<f16>,
                          head_num,
                          kThreadNum,
                          stream,
                          static_cast<const f16*>(query),
                          static_cast<const f16*>(key_cache),
                          static_cast<const f16*>(value_cache),
                          static_cast<f32*>(score),
                          static_cast<f16*>(output),
                          pos,
                          seq_len,
                          kv_dim,
                          kv_mul,
                          head_num,
                          head_size);
            break;
        case DataType::BFloat16:
            LAUNCH_KERNEL(mha_kernel<bf16>,
                          head_num,
                          kThreadNum,
                          stream,
                          static_cast<const bf16*>(query),
                          static_cast<const bf16*>(key_cache),
                          static_cast<const bf16*>(value_cache),
                          static_cast<f32*>(score),
                          static_cast<bf16*>(output),
                          pos,
                          seq_len,
                          kv_dim,
                          kv_mul,
                          head_num,
                          head_size);
            break;
        default:
            return Err<void>(ErrorCode::NotImplemented, "CUDA MHA only supports Float16/BFloat16/Float32");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        pulse::error("CUDA MHA kernel launch failed for {}: {}",
                     data_type_str(dtype),
                     cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA MHA kernel launch failed for {}: {}",
                                     data_type_str(dtype),
                                     cudaGetErrorString(err)));
    }

    return Ok();
}

}  // namespace pulse::kernels::cuda
