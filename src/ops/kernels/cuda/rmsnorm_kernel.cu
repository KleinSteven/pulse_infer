#include "pulse/ops/kernels/cuda/rmsnorm_kernel.cuh"

#include <cmath>
#include <type_traits>

#include "pulse/logging.hpp"

namespace pulse::kernels::cuda {

namespace {

constexpr int kWarpSize = 32;

template<int WarpSize = kWarpSize>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<int NumThreads = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    constexpr int num_warps = (NumThreads + kWarpSize - 1) / kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    static __shared__ float shared[num_warps];

    val = warp_reduce_sum<kWarpSize>(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    val = lane < num_warps ? shared[lane] : 0.0f;
    val = warp_reduce_sum<num_warps>(val);
    return val;
}

template<int NumThreads = 256>
__device__ __forceinline__ double block_reduce_sum_f64(double val) {
    static __shared__ double shared[NumThreads];
    shared[threadIdx.x] = val;
    __syncthreads();

    for (int stride = NumThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    return shared[0];
}

template<typename T, int NumThreads = 256>
struct BlockReduceSum;

template<int NumThreads>
struct BlockReduceSum<float, NumThreads> {
    __device__ __forceinline__ static float run(float val) {
        return block_reduce_sum_f32<NumThreads>(val);
    }
};

template<int NumThreads>
struct BlockReduceSum<double, NumThreads> {
    __device__ __forceinline__ static double run(double val) {
        return block_reduce_sum_f64<NumThreads>(val);
    }
};

template<typename AccT>
__device__ __forceinline__ AccT inverse_rms(AccT variance_sum, i32 normalized_size, AccT eps) {
    if constexpr (std::is_same_v<AccT, float>) {
        return rsqrtf(variance_sum / static_cast<float>(normalized_size) + eps);
    } else {
        return static_cast<AccT>(1) /
               sqrt(variance_sum / static_cast<AccT>(normalized_size) + eps);
    }
}

template<typename T, typename AccT, int Pack, int NumThreads = 256>
__global__ void rms_norm_vec_kernel(const T* input,
                                    const T* weight,
                                    T* output,
                                    i32 normalized_size,
                                    AccT eps) {
    const i32 row = blockIdx.x;
    const i32 tid = threadIdx.x;
    const i64 row_base = static_cast<i64>(row) * static_cast<i64>(normalized_size);
    const i32 pack_stride = NumThreads * Pack;

    __shared__ AccT inv_rms_shared;

    AccT variance = static_cast<AccT>(0);
    for (i32 col = tid * Pack; col < normalized_size; col += pack_stride) {
#pragma unroll
        for (int i = 0; i < Pack; ++i) {
            const i32 idx = col + i;
            if (idx < normalized_size) {
                AccT value;
                if constexpr (std::is_same_v<T, f16>) {
                    value = static_cast<AccT>(__half2float(input[row_base + idx]));
                } else if constexpr (std::is_same_v<T, bf16>) {
                    value = static_cast<AccT>(__bfloat162float(input[row_base + idx]));
                } else {
                    value = static_cast<AccT>(input[row_base + idx]);
                }
                variance += value * value;
            }
        }
    }

    variance = BlockReduceSum<AccT, NumThreads>::run(variance);
    if (tid == 0) {
        inv_rms_shared = inverse_rms<AccT>(variance, normalized_size, eps);
    }
    __syncthreads();

    for (i32 col = tid * Pack; col < normalized_size; col += pack_stride) {
#pragma unroll
        for (int i = 0; i < Pack; ++i) {
            const i32 idx = col + i;
            if (idx < normalized_size) {
                AccT value;
                if constexpr (std::is_same_v<T, f16>) {
                    value = static_cast<AccT>(__half2float(input[row_base + idx]));
                } else if constexpr (std::is_same_v<T, bf16>) {
                    value = static_cast<AccT>(__bfloat162float(input[row_base + idx]));
                } else {
                    value = static_cast<AccT>(input[row_base + idx]);
                }

                AccT scale;
                if (weight == nullptr) {
                    scale = static_cast<AccT>(1);
                } else if constexpr (std::is_same_v<T, f16>) {
                    scale = static_cast<AccT>(__half2float(weight[idx]));
                } else if constexpr (std::is_same_v<T, bf16>) {
                    scale = static_cast<AccT>(__bfloat162float(weight[idx]));
                } else {
                    scale = static_cast<AccT>(weight[idx]);
                }

                const AccT output_value = value * inv_rms_shared * scale;
                if constexpr (std::is_same_v<T, f16>) {
                    output[row_base + idx] = __float2half(output_value);
                } else if constexpr (std::is_same_v<T, bf16>) {
                    output[row_base + idx] = __float2bfloat16(output_value);
                } else {
                    output[row_base + idx] = static_cast<T>(output_value);
                }
            }
        }
    }
}

template<typename T, typename AccT, int Pack>
void launch_rms_norm_kernel(const void* input,
                            const void* weight,
                            void* output,
                            i32 rows,
                            i32 normalized_size,
                            AccT eps,
                            cudaStream_t stream) {
    constexpr i32 thread_num = 256;
    if (stream != nullptr) {
        rms_norm_vec_kernel<T, AccT, Pack, thread_num><<<rows, thread_num, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<const T*>(weight),
            static_cast<T*>(output),
            normalized_size,
            eps);
    } else {
        rms_norm_vec_kernel<T, AccT, Pack, thread_num><<<rows, thread_num>>>(
            static_cast<const T*>(input),
            static_cast<const T*>(weight),
            static_cast<T*>(output),
            normalized_size,
            eps);
    }
}

}  // namespace

Result<void> rms_norm_cuda_launch(const void* input,
                                  const void* weight,
                                  void* output,
                                  i32 rows,
                                  i32 normalized_size,
                                  f64 eps,
                                  DataType dtype,
                                  cudaStream_t stream) {
    if (rows <= 0 || normalized_size <= 0) {
        return Ok();
    }

    switch (dtype) {
        case DataType::Float16:
            launch_rms_norm_kernel<f16, float, 8>(
                input, weight, output, rows, normalized_size, static_cast<float>(eps), stream);
            break;
        case DataType::BFloat16:
            launch_rms_norm_kernel<bf16, float, 8>(
                input, weight, output, rows, normalized_size, static_cast<float>(eps), stream);
            break;
        case DataType::Float32:
            launch_rms_norm_kernel<f32, float, 4>(
                input, weight, output, rows, normalized_size, static_cast<float>(eps), stream);
            break;
        case DataType::Float64:
            launch_rms_norm_kernel<f64, double, 2>(
                input, weight, output, rows, normalized_size, eps, stream);
            break;
        default:
            return Err<void>(ErrorCode::NotImplemented,
                             "CUDA RMSNorm only supports Float16/BFloat16/Float32/Float64");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        pulse::error("CUDA rmsnorm kernel launch failed for {}: {}",
                     data_type_str(dtype),
                     cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA rmsnorm kernel launch failed for {}: {}",
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
        pulse::error("CUDA rmsnorm kernel execution failed for {}: {}",
                     data_type_str(dtype),
                     cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA rmsnorm kernel execution failed for {}: {}",
                                     data_type_str(dtype),
                                     cudaGetErrorString(err)));
    }

    return Ok();
}

}  // namespace pulse::kernels::cuda
