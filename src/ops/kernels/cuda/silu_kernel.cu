#include "pulse/ops/kernels/cuda/silu_kernel.cuh"

#include <cmath>

#include "pulse/logging.hpp"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

#define LAUNCH_KERNEL(kernel, block_num, thread_num, stream, ...) \
  do { \
    if (stream) { \
      kernel<<<block_num, thread_num, 0, stream>>>(__VA_ARGS__); \
    } else { \
      kernel<<<block_num, thread_num>>>(__VA_ARGS__); \
    } \
  } while (0)

namespace pulse::kernels::cuda {

__global__ void silu_f32_kernel(f32* input, f32* output, i64 size) {
  const i32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

  if (idx + 3 < size) {
    const float4 reg_input = FLOAT4(input[idx]);
    float4 reg_output;

    reg_output.x = reg_input.x / (1.0f + __expf(-reg_input.x));
    reg_output.y = reg_input.y / (1.0f + __expf(-reg_input.y));
    reg_output.z = reg_input.z / (1.0f + __expf(-reg_input.z));
    reg_output.w = reg_input.w / (1.0f + __expf(-reg_input.w));

    FLOAT4(output[idx]) = reg_output;
    return;
  }

  for (i64 i = idx; i < size; ++i) {
    output[i] = input[i] / (1.0f + __expf(-input[i]));
  }
}

__global__ void silu_f64_kernel(f64* input, f64* output, i64 size) {
  const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) {
    return;
  }

  output[idx] = input[idx] / (1.0 + exp(-input[idx]));
}

__global__ void silu_f16x8_pack_kernel(f16* input, f16* output, i64 size) {
  const i32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

  if (idx + 7 < size) {
    output[idx] = __float2half(__half2float(input[idx]) / (1.0f + __expf(-__half2float(input[idx]))));
    output[idx + 1] =
        __float2half(__half2float(input[idx + 1]) / (1.0f + __expf(-__half2float(input[idx + 1]))));
    output[idx + 2] =
        __float2half(__half2float(input[idx + 2]) / (1.0f + __expf(-__half2float(input[idx + 2]))));
    output[idx + 3] =
        __float2half(__half2float(input[idx + 3]) / (1.0f + __expf(-__half2float(input[idx + 3]))));
    output[idx + 4] =
        __float2half(__half2float(input[idx + 4]) / (1.0f + __expf(-__half2float(input[idx + 4]))));
    output[idx + 5] =
        __float2half(__half2float(input[idx + 5]) / (1.0f + __expf(-__half2float(input[idx + 5]))));
    output[idx + 6] =
        __float2half(__half2float(input[idx + 6]) / (1.0f + __expf(-__half2float(input[idx + 6]))));
    output[idx + 7] =
        __float2half(__half2float(input[idx + 7]) / (1.0f + __expf(-__half2float(input[idx + 7]))));
    return;
  }

  for (i64 i = 0; idx + i < size; ++i) {
    output[idx + i] =
        __float2half(__half2float(input[idx + i]) / (1.0f + __expf(-__half2float(input[idx + i]))));
  }
}

__global__ void silu_bf16x8_pack_kernel(bf16* input, bf16* output, i64 size) {
  const i32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

  if (idx + 7 < size) {
    output[idx] =
        __float2bfloat16(__bfloat162float(input[idx]) / (1.0f + __expf(-__bfloat162float(input[idx]))));
    output[idx + 1] = __float2bfloat16(__bfloat162float(input[idx + 1]) /
                                       (1.0f + __expf(-__bfloat162float(input[idx + 1]))));
    output[idx + 2] = __float2bfloat16(__bfloat162float(input[idx + 2]) /
                                       (1.0f + __expf(-__bfloat162float(input[idx + 2]))));
    output[idx + 3] = __float2bfloat16(__bfloat162float(input[idx + 3]) /
                                       (1.0f + __expf(-__bfloat162float(input[idx + 3]))));
    output[idx + 4] = __float2bfloat16(__bfloat162float(input[idx + 4]) /
                                       (1.0f + __expf(-__bfloat162float(input[idx + 4]))));
    output[idx + 5] = __float2bfloat16(__bfloat162float(input[idx + 5]) /
                                       (1.0f + __expf(-__bfloat162float(input[idx + 5]))));
    output[idx + 6] = __float2bfloat16(__bfloat162float(input[idx + 6]) /
                                       (1.0f + __expf(-__bfloat162float(input[idx + 6]))));
    output[idx + 7] = __float2bfloat16(__bfloat162float(input[idx + 7]) /
                                       (1.0f + __expf(-__bfloat162float(input[idx + 7]))));
    return;
  }

  for (i64 i = 0; idx + i < size; ++i) {
    output[idx + i] = __float2bfloat16(__bfloat162float(input[idx + i]) /
                                       (1.0f + __expf(-__bfloat162float(input[idx + i]))));
  }
}

Result<void> silu_cuda_launch(const void* input,
                              void* output,
                              i64 size,
                              DataType dtype,
                              cudaStream_t stream) {
  if (size <= 0) {
    return Ok();
  }

  cudaError_t err = cudaSuccess;

  switch (dtype) {
    case DataType::Float32: {
      constexpr i32 thread_num = 256;
      const i32 block_num = static_cast<i32>((size + static_cast<i64>(thread_num) * 4 - 1) /
                                             (static_cast<i64>(thread_num) * 4));
      LAUNCH_KERNEL(silu_f32_kernel, block_num, thread_num, stream, (f32*)input, (f32*)output, size);
      break;
    }
    case DataType::Float64: {
      constexpr i32 thread_num = 256;
      const i32 block_num = static_cast<i32>((size + thread_num - 1) / thread_num);
      LAUNCH_KERNEL(silu_f64_kernel, block_num, thread_num, stream, (f64*)input, (f64*)output, size);
      break;
    }
    case DataType::Float16: {
      constexpr i32 thread_num = 256;
      const i32 block_num = static_cast<i32>((size + static_cast<i64>(thread_num) * 8 - 1) /
                                             (static_cast<i64>(thread_num) * 8));
      LAUNCH_KERNEL(silu_f16x8_pack_kernel,
                    block_num,
                    thread_num,
                    stream,
                    (f16*)input,
                    (f16*)output,
                    size);
      break;
    }
    case DataType::BFloat16: {
      constexpr i32 thread_num = 256;
      const i32 block_num = static_cast<i32>((size + static_cast<i64>(thread_num) * 8 - 1) /
                                             (static_cast<i64>(thread_num) * 8));
      LAUNCH_KERNEL(silu_bf16x8_pack_kernel,
                    block_num,
                    thread_num,
                    stream,
                    (bf16*)input,
                    (bf16*)output,
                    size);
      break;
    }
    default:
      return Err<void>(ErrorCode::NotImplemented,
                       "CUDA SiLU only supports Float16/BFloat16/Float32/Float64");
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    pulse::error("CUDA silu kernel launch failed for {}: {}",
                 data_type_str(dtype),
                 cudaGetErrorString(err));
    return Err<void>(ErrorCode::CudaError,
                     std::format("CUDA silu kernel launch failed for {}: {}",
                                 data_type_str(dtype),
                                 cudaGetErrorString(err)));
  }

  if (stream == nullptr) {
    err = cudaDeviceSynchronize();
  } else {
    err = cudaStreamSynchronize(stream);
  }

  if (err != cudaSuccess) {
    pulse::error("CUDA silu kernel execution failed for {}: {}",
                 data_type_str(dtype),
                 cudaGetErrorString(err));
    return Err<void>(ErrorCode::CudaError,
                     std::format("CUDA silu kernel execution failed for {}: {}",
                                 data_type_str(dtype),
                                 cudaGetErrorString(err)));
  }

  return Ok();
}

}  // namespace pulse::kernels::cuda
