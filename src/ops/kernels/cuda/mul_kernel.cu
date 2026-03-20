#include "pulse/ops/kernels/cuda/mul_kernel.cuh"

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
__global__ void mul_kernel(const T* input1, const T* input2, T* output, i64 size) {
    const i64 idx = threadIdx.x + static_cast<i64>(blockIdx.x) * blockDim.x;
    if (idx >= size) {
        return;
    }

    output[idx] = input1[idx] * input2[idx];
}

template<>
__global__ void mul_kernel<f16>(const f16* input1, const f16* input2, f16* output, i64 size) {
    const i64 idx = threadIdx.x + static_cast<i64>(blockIdx.x) * blockDim.x;
    if (idx >= size) {
        return;
    }

    output[idx] = __float2half(__half2float(input1[idx]) * __half2float(input2[idx]));
}

template<>
__global__ void mul_kernel<bf16>(const bf16* input1, const bf16* input2, bf16* output, i64 size) {
    const i64 idx = threadIdx.x + static_cast<i64>(blockIdx.x) * blockDim.x;
    if (idx >= size) {
        return;
    }

    output[idx] = __float2bfloat16(__bfloat162float(input1[idx]) * __bfloat162float(input2[idx]));
}

}  // namespace

Result<void> mul_cuda_launch(
    const void* input1, const void* input2, void* output, i64 size, DataType dtype, cudaStream_t stream) {
    if (size <= 0) {
        return Ok();
    }

    constexpr i32 thread_num = 256;
    const i32 block_num = static_cast<i32>((size + thread_num - 1) / thread_num);

    switch (dtype) {
        case DataType::Float32:
            LAUNCH_KERNEL(
                mul_kernel<f32>, block_num, thread_num, stream, (const f32*)input1, (const f32*)input2, (f32*)output, size);
            break;
        case DataType::Float64:
            LAUNCH_KERNEL(
                mul_kernel<f64>, block_num, thread_num, stream, (const f64*)input1, (const f64*)input2, (f64*)output, size);
            break;
        case DataType::Float16:
            LAUNCH_KERNEL(
                mul_kernel<f16>, block_num, thread_num, stream, (const f16*)input1, (const f16*)input2, (f16*)output, size);
            break;
        case DataType::BFloat16:
            LAUNCH_KERNEL(mul_kernel<bf16>,
                          block_num,
                          thread_num,
                          stream,
                          (const bf16*)input1,
                          (const bf16*)input2,
                          (bf16*)output,
                          size);
            break;
        default:
            return Err<void>(ErrorCode::NotImplemented,
                             "CUDA Mul only supports Float16/BFloat16/Float32/Float64");
    }

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        pulse::error("CUDA mul kernel launch failed for {}: {}",
                     data_type_str(dtype),
                     cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA mul kernel launch failed for {}: {}",
                                     data_type_str(dtype),
                                     cudaGetErrorString(err)));
    }

    return Ok();
}

}  // namespace pulse::kernels::cuda
