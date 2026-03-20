#include "pulse/ops/kernels/cuda/add_kernel.cuh"

#include "pulse/logging.hpp"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

#define LAUNCH_KERNEL(kernel, block_num, thread_num, stream, ...) \
  do { \
    if (stream) { \
      kernel<<<block_num, thread_num, 0, stream>>>(__VA_ARGS__); \
    } else { \
      kernel<<<block_num, thread_num>>>(__VA_ARGS__); \
    } \
  } while (0)

namespace pulse::kernels::cuda {

__global__ void add_f32_kernel(f32* input1, f32* input2, f32* output, i64 size) {
  const i32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

  if (idx + 3 < size) {
    float4 reg_a = FLOAT4(input1[idx]);
    float4 reg_b = FLOAT4(input2[idx]);
    float4 reg_c;

    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;

    FLOAT4(output[idx]) = reg_c;
    return;
  }

  for (i64 i = idx; i < size; ++i) {
    output[i] = input1[i] + input2[i];
  }
}

__global__ void add_f64_kernel(f64* input1, f64* input2, f64* output, i64 size) {
  const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) {
    return;
  }

  output[idx] = input1[idx] + input2[idx];
}

__global__ void add_f16x8_pack_kernel(f16* input1, f16* input2, f16* output, i64 size) {
  const i32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

  if (idx + 7 < size) {
    half2 reg_c_0 = __hadd2(HALF2(input1[idx]), HALF2(input2[idx]));
    half2 reg_c_1 = __hadd2(HALF2(input1[idx + 2]), HALF2(input2[idx + 2]));
    half2 reg_c_2 = __hadd2(HALF2(input1[idx + 4]), HALF2(input2[idx + 4]));
    half2 reg_c_3 = __hadd2(HALF2(input1[idx + 6]), HALF2(input2[idx + 6]));

    HALF2(output[idx]) = reg_c_0;
    HALF2(output[idx + 2]) = reg_c_1;
    HALF2(output[idx + 4]) = reg_c_2;
    HALF2(output[idx + 6]) = reg_c_3;
    return;
  }

  for (i64 i = 0; idx + i < size; ++i) {
    output[idx + i] = __hadd(input1[idx + i], input2[idx + i]);
  }
}


__global__ void add_bf16x8_pack_kernel(bf16* input1, bf16* input2, bf16* output, i64 size) {
  const i32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

  if (idx + 7 < size) {
    __nv_bfloat162 reg_c_0 = __hadd2(BFLOAT2(input1[idx]), BFLOAT2(input2[idx]));
    __nv_bfloat162 reg_c_1 = __hadd2(BFLOAT2(input1[idx + 2]), BFLOAT2(input2[idx + 2]));
    __nv_bfloat162 reg_c_2 = __hadd2(BFLOAT2(input1[idx + 4]), BFLOAT2(input2[idx + 4]));
    __nv_bfloat162 reg_c_3 = __hadd2(BFLOAT2(input1[idx + 6]), BFLOAT2(input2[idx + 6]));

    BFLOAT2(output[idx]) = reg_c_0;
    BFLOAT2(output[idx + 2]) = reg_c_1;
    BFLOAT2(output[idx + 4]) = reg_c_2;
    BFLOAT2(output[idx + 6]) = reg_c_3;
    return;
  }

  for (i64 i = 0; idx + i < size; ++i) {
    output[idx + i] = __hadd(input1[idx + i], input2[idx + i]);
  }
}

__global__ void add_bias_f32_kernel(const f32* bias, f32* output, i64 total_size, i32 cols) {
  const i64 idx = threadIdx.x + static_cast<i64>(blockIdx.x) * blockDim.x;
  if (idx >= total_size) {
    return;
  }

  const i32 col = static_cast<i32>(idx % cols);
  output[idx] += bias[col];
}

__global__ void add_bias_f64_kernel(const f64* bias, f64* output, i64 total_size, i32 cols) {
  const i64 idx = threadIdx.x + static_cast<i64>(blockIdx.x) * blockDim.x;
  if (idx >= total_size) {
    return;
  }

  const i32 col = static_cast<i32>(idx % cols);
  output[idx] += bias[col];
}

__global__ void add_bias_f16_kernel(const f16* bias, f16* output, i64 total_size, i32 cols) {
  const i64 idx = threadIdx.x + static_cast<i64>(blockIdx.x) * blockDim.x;
  if (idx >= total_size) {
    return;
  }

  const i32 col = static_cast<i32>(idx % cols);
  output[idx] = __float2half(__half2float(output[idx]) + __half2float(bias[col]));
}

__global__ void add_bias_bf16_kernel(const bf16* bias, bf16* output, i64 total_size, i32 cols) {
  const i64 idx = threadIdx.x + static_cast<i64>(blockIdx.x) * blockDim.x;
  if (idx >= total_size) {
    return;
  }

  const i32 col = static_cast<i32>(idx % cols);
  output[idx] = __float2bfloat16(__bfloat162float(output[idx]) + __bfloat162float(bias[col]));
}


Result<void> add_cuda_launch(
    const void* input1,
    const void* input2,
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
      LAUNCH_KERNEL(add_f32_kernel,
                    block_num,
                    thread_num,
                    stream,
                    (f32*)input1,
                    (f32*)input2,
                    (f32*)output,
                    size);
      break;
    }
    case DataType::Float64: {
      constexpr i32 thread_num = 256;
      const i32 block_num = static_cast<i32>((size + thread_num - 1) / thread_num);
      LAUNCH_KERNEL(add_f64_kernel,
                    block_num,
                    thread_num,
                    stream,
                    (f64*)input1,
                    (f64*)input2,
                    (f64*)output,
                    size);
      break;
    }
    case DataType::Float16: {
      constexpr i32 thread_num = 256;
      const i32 block_num = (size + thread_num * 8 - 1) / (thread_num * 8);
      LAUNCH_KERNEL(add_f16x8_pack_kernel,
                    block_num,
                    thread_num,
                    stream,
                    (f16*)input1,
                    (f16*)input2,
                    (f16*)output,
                    size);
      break;
    }
    case DataType::BFloat16: {
      constexpr i32 thread_num = 256;
      const i32 block_num = (size + thread_num * 8 - 1) / (thread_num * 8);
      LAUNCH_KERNEL(add_bf16x8_pack_kernel,
                    block_num,
                    thread_num,
                    stream,
                    (bf16*)input1,
                    (bf16*)input2,
                    (bf16*)output,
                    size);
      break;
    }
    default:
      return Err<void>(ErrorCode::NotImplemented,
                       "CUDA Add only supports Float16/BFloat16/Float32/Float64");
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    pulse::error("CUDA add kernel launch failed for {}: {}",
                 data_type_str(dtype),
                 cudaGetErrorString(err));
    return Err<void>(ErrorCode::CudaError,
      std::format("CUDA add kernel launch failed for {}: {}", 
        data_type_str(dtype), cudaGetErrorString(err)));
  }

  return Ok();
}

Result<void> add_bias_cuda_launch(const void* bias,
                                  void* output,
                                  i32 rows,
                                  i32 cols,
                                  DataType dtype,
                                  cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return Ok();
  }

  const i64 total_size = static_cast<i64>(rows) * static_cast<i64>(cols);
  constexpr i32 thread_num = 256;
  const i32 block_num = static_cast<i32>((total_size + thread_num - 1) / thread_num);

  switch (dtype) {
    case DataType::Float32:
      LAUNCH_KERNEL(
          add_bias_f32_kernel, block_num, thread_num, stream, (const f32*)bias, (f32*)output, total_size, cols);
      break;
    case DataType::Float64:
      LAUNCH_KERNEL(
          add_bias_f64_kernel, block_num, thread_num, stream, (const f64*)bias, (f64*)output, total_size, cols);
      break;
    case DataType::Float16:
      LAUNCH_KERNEL(
          add_bias_f16_kernel, block_num, thread_num, stream, (const f16*)bias, (f16*)output, total_size, cols);
      break;
    case DataType::BFloat16:
      LAUNCH_KERNEL(add_bias_bf16_kernel,
                    block_num,
                    thread_num,
                    stream,
                    (const bf16*)bias,
                    (bf16*)output,
                    total_size,
                    cols);
      break;
    default:
      return Err<void>(ErrorCode::NotImplemented,
                       "CUDA Linear bias only supports Float16/BFloat16/Float32/Float64");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    pulse::error("CUDA linear bias kernel launch failed for {}: {}",
                 data_type_str(dtype),
                 cudaGetErrorString(err));
    return Err<void>(ErrorCode::CudaError,
                     std::format("CUDA linear bias kernel launch failed for {}: {}",
                                 data_type_str(dtype),
                                 cudaGetErrorString(err)));
  }

  return Ok();
}

}  // namespace pulse::kernels::cuda
