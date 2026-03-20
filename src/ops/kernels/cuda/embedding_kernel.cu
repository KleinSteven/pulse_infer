#include "pulse/ops/kernels/cuda/embedding_kernel.cuh"

#include "pulse/logging.hpp"

namespace pulse::kernels::cuda {

template<typename T>
__global__ void embedding_kernel(const i32* input,
                                 const T* weight,
                                 T* output,
                                 i32 num_tokens,
                                 i32 vocab_size,
                                 i32 embedding_dim) {
    const i32 token_index = blockIdx.x;
    const i32 dim_index = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_index >= num_tokens || dim_index >= embedding_dim) {
        return;
    }

    const i32 token = input[token_index];
    if (token < 0 || token >= vocab_size) {
        return;
    }

    output[static_cast<usize>(token_index) * static_cast<usize>(embedding_dim) + static_cast<usize>(dim_index)] =
        weight[static_cast<usize>(token) * static_cast<usize>(embedding_dim) + static_cast<usize>(dim_index)];
}

Result<void> embedding_cuda_launch(const void* input,
                                   const void* weight,
                                   void* output,
                                   i32 num_tokens,
                                   i32 vocab_size,
                                   i32 embedding_dim,
                                   DataType dtype,
                                   cudaStream_t stream) {
    if (num_tokens <= 0 || embedding_dim <= 0) {
        return Ok();
    }

    constexpr i32 thread_num = 256;
    const dim3 block(thread_num);
    const dim3 grid(num_tokens, static_cast<u32>((embedding_dim + thread_num - 1) / thread_num));

    switch (dtype) {
        case DataType::Float32:
            embedding_kernel<<<grid, block, 0, stream>>>(
                static_cast<const i32*>(input),
                static_cast<const f32*>(weight),
                static_cast<f32*>(output),
                num_tokens,
                vocab_size,
                embedding_dim);
            break;
        case DataType::Float64:
            embedding_kernel<<<grid, block, 0, stream>>>(
                static_cast<const i32*>(input),
                static_cast<const f64*>(weight),
                static_cast<f64*>(output),
                num_tokens,
                vocab_size,
                embedding_dim);
            break;
        case DataType::Float16:
            embedding_kernel<<<grid, block, 0, stream>>>(
                static_cast<const i32*>(input),
                static_cast<const f16*>(weight),
                static_cast<f16*>(output),
                num_tokens,
                vocab_size,
                embedding_dim);
            break;
        case DataType::BFloat16:
            embedding_kernel<<<grid, block, 0, stream>>>(
                static_cast<const i32*>(input),
                static_cast<const bf16*>(weight),
                static_cast<bf16*>(output),
                num_tokens,
                vocab_size,
                embedding_dim);
            break;
        default:
            return Err<void>(ErrorCode::NotImplemented,
                             "CUDA Embedding only supports Float16/BFloat16/Float32/Float64");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        pulse::error("CUDA embedding kernel launch failed for {}: {}",
                     data_type_str(dtype),
                     cudaGetErrorString(err));
        return Err<void>(ErrorCode::CudaError,
                         std::format("CUDA embedding kernel launch failed for {}: {}",
                                     data_type_str(dtype),
                                     cudaGetErrorString(err)));
    }

    return Ok();
}

}  // namespace pulse::kernels::cuda
