#include "pulse/ops/kernels/cuda/rope_kernel.cuh"

#include <cmath>

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
__global__ void rope_kernel(const T* input,
                            T* output,
                            i32 rows,
                            i32 seq_len,
                            i32 head_dim,
                            i32 rotary_dim,
                            i32 position_offset,
                            float theta) {
    const i64 total = static_cast<i64>(rows) * static_cast<i64>(seq_len) * static_cast<i64>(head_dim);
    const i64 idx = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const i32 dim = static_cast<i32>(idx % head_dim);
    const i64 token_idx = idx / head_dim;
    const i32 pos = static_cast<i32>(token_idx % seq_len);

    if (dim >= rotary_dim) {
        output[idx] = input[idx];
        return;
    }

    const i32 pair_base_dim = dim & ~1;
    const i64 pair_base_idx = idx - dim + pair_base_dim;
    float x0;
    float x1;

    if constexpr (std::is_same_v<T, f16>) {
        x0 = __half2float(input[pair_base_idx]);
        x1 = __half2float(input[pair_base_idx + 1]);
    } else if constexpr (std::is_same_v<T, bf16>) {
        x0 = __bfloat162float(input[pair_base_idx]);
        x1 = __bfloat162float(input[pair_base_idx + 1]);
    } else {
        x0 = static_cast<float>(input[pair_base_idx]);
        x1 = static_cast<float>(input[pair_base_idx + 1]);
    }
    
    const float exponent = static_cast<float>(pair_base_dim) / static_cast<float>(rotary_dim);
    const float angle = static_cast<float>(position_offset + pos) / powf(theta, exponent);
    const float cos_value = cosf(angle);
    const float sin_value = sinf(angle);

    const float rotated =
        (dim & 1) == 0 ? (x0 * cos_value - x1 * sin_value) : (x0 * sin_value + x1 * cos_value);
    if constexpr (std::is_same_v<T, f16>) {
        output[idx] = __float2half(rotated);
    } else if constexpr (std::is_same_v<T, bf16>) {
        output[idx] = __float2bfloat16(rotated);
    } else {
        output[idx] = static_cast<T>(rotated);
    }
}

__global__ void rope_f64_kernel(const f64* input,
                                f64* output,
                                i32 rows,
                                i32 seq_len,
                                i32 head_dim,
                                i32 rotary_dim,
                                i32 position_offset,
                                double theta) {
    const i64 total = static_cast<i64>(rows) * static_cast<i64>(seq_len) * static_cast<i64>(head_dim);
    const i64 idx = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const i32 dim = static_cast<i32>(idx % head_dim);
    const i64 token_idx = idx / head_dim;
    const i32 pos = static_cast<i32>(token_idx % seq_len);

    if (dim >= rotary_dim) {
        output[idx] = input[idx];
        return;
    }

    const i32 pair_base_dim = dim & ~1;
    const i64 pair_base_idx = idx - dim + pair_base_dim;
    const double x0 = input[pair_base_idx];
    const double x1 = input[pair_base_idx + 1];
    const double exponent = static_cast<double>(pair_base_dim) / static_cast<double>(rotary_dim);
    const double angle = static_cast<double>(position_offset + pos) / pow(theta, exponent);
    const double cos_value = cos(angle);
    const double sin_value = sin(angle);

    output[idx] =
        (dim & 1) == 0 ? (x0 * cos_value - x1 * sin_value) : (x0 * sin_value + x1 * cos_value);
}

}  // namespace

Result<void> rope_cuda_launch(const void* input,
                              void* output,
                              i32 rows,
                              i32 seq_len,
                              i32 head_dim,
                              i32 rotary_dim,
                              i32 position_offset,
                              f32 theta,
                              DataType dtype,
                              cudaStream_t stream) {
    if (rows <= 0 || seq_len <= 0 || head_dim <= 0) {
        return Ok();
    }

    const i64 total = static_cast<i64>(rows) * static_cast<i64>(seq_len) * static_cast<i64>(head_dim);
    constexpr i32 thread_num = 256;
    const i32 block_num = static_cast<i32>((total + thread_num - 1) / thread_num);

    switch (dtype) {
        case DataType::Float16:
            LAUNCH_KERNEL(rope_kernel<f16>,
                          block_num,
                          thread_num,
                          stream,
                          static_cast<const f16*>(input),
                          static_cast<f16*>(output),
                          rows,
                          seq_len,
                          head_dim,
                          rotary_dim,
                          position_offset,
                          theta);
            break;
        case DataType::BFloat16:
            LAUNCH_KERNEL(rope_kernel<bf16>,
                          block_num,
                          thread_num,
                          stream,
                          static_cast<const bf16*>(input),
                          static_cast<bf16*>(output),
                          rows,
                          seq_len,
                          head_dim,
                          rotary_dim,
                          position_offset,
                          theta);
            break;
        case DataType::Float32:
            LAUNCH_KERNEL(rope_kernel<f32>,
                          block_num,
                          thread_num,
                          stream,
                          static_cast<const f32*>(input),
                          static_cast<f32*>(output),
                          rows,
                          seq_len,
                          head_dim,
                          rotary_dim,
                          position_offset,
                          theta);
            break;
        case DataType::Float64:
            LAUNCH_KERNEL(rope_f64_kernel,
                          block_num,
                          thread_num,
                          stream,
                          static_cast<const f64*>(input),
                          static_cast<f64*>(output),
                          rows,
                          seq_len,
                          head_dim,
                          rotary_dim,
                          position_offset,
                          static_cast<double>(theta));
            break;
        default:
            return Err<void>(ErrorCode::NotImplemented,
                             "CUDA RoPE only supports Float16/BFloat16/Float32/Float64");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        pulse::error("CUDA rope kernel launch failed for {}: {}",
                     data_type_str(dtype),
                     cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA rope kernel launch failed for {}: {}",
                                     data_type_str(dtype),
                                     cudaGetErrorString(err)));
    }

    // TODO: support true multi-stream/asynchronous execution without forcing stream sync here.
    if (stream == nullptr) {
        err = cudaDeviceSynchronize();
    } else {
        err = cudaStreamSynchronize(stream);
    }

    if (err != cudaSuccess) {
        pulse::error("CUDA rope kernel execution failed for {}: {}",
                     data_type_str(dtype),
                     cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA rope kernel execution failed for {}: {}",
                                     data_type_str(dtype),
                                     cudaGetErrorString(err)));
    }

    return Ok();
}

}  // namespace pulse::kernels::cuda
