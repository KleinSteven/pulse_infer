#include "pulse/ops/kernels/cuda/matmul_kernel.cuh"

#include "pulse/logging.hpp"

namespace pulse::kernels::cuda {

namespace {

Result<void> cublas_error(cublasStatus_t status, std::string_view action) {
  pulse::error("cuBLAS {} failed: {} ({})",
               action,
               cublasGetStatusName(status),
               cublasGetStatusString(status));
  return Err<void>(ErrorCode::CudaError,
                   std::format("cuBLAS {} failed: {} ({})",
                               action,
                               cublasGetStatusName(status),
                               cublasGetStatusString(status)));
}

}  // namespace

Result<void> matmul_cuda_launch(const void* input1,
                                const void* input2,
                                void* output,
                                i32 m,
                                i32 n,
                                i32 k,
                                DataType dtype,
                                bool transpose_input1,
                                bool transpose_input2,
                                cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return Ok();
  }

  cublasHandle_t handle = nullptr;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_error(status, "create");
  }

  status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    return cublas_error(status, "set stream");
  }

  // Row-major C = op(A) * op(B) maps to column-major C^T = op(B)^T * op(A)^T on the same buffers.
  const cublasOperation_t input2_op = transpose_input2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t input1_op = transpose_input1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  const i32 input2_leading_dim = transpose_input2 ? k : n;
  const i32 input1_leading_dim = transpose_input1 ? m : k;

  switch (dtype) {
    case DataType::Float32: {
      const f32 alpha = 1.0f;
      const f32 beta = 0.0f;
      status = cublasSgemm(handle,
                           input2_op,
                           input1_op,
                           n,
                           m,
                           k,
                           &alpha,
                           static_cast<const f32*>(input2),
                           input2_leading_dim,
                           static_cast<const f32*>(input1),
                           input1_leading_dim,
                           &beta,
                           static_cast<f32*>(output),
                           n);
      break;
    }
    case DataType::Float64: {
      const f64 alpha = 1.0;
      const f64 beta = 0.0;
      status = cublasDgemm(handle,
                           input2_op,
                           input1_op,
                           n,
                           m,
                           k,
                           &alpha,
                           static_cast<const f64*>(input2),
                           input2_leading_dim,
                           static_cast<const f64*>(input1),
                           input1_leading_dim,
                           &beta,
                           static_cast<f64*>(output),
                           n);
      break;
    }
    case DataType::Float16: {
      const f32 alpha = 1.0f;
      const f32 beta = 0.0f;
      status = cublasGemmEx(handle,
                            input2_op,
                            input1_op,
                            n,
                            m,
                            k,
                            &alpha,
                            static_cast<const f16*>(input2),
                            CUDA_R_16F,
                            input2_leading_dim,
                            static_cast<const f16*>(input1),
                            CUDA_R_16F,
                            input1_leading_dim,
                            &beta,
                            static_cast<f16*>(output),
                            CUDA_R_16F,
                            n,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT);
      break;
    }
    case DataType::BFloat16: {
      const f32 alpha = 1.0f;
      const f32 beta = 0.0f;
      status = cublasGemmEx(handle,
                            input2_op,
                            input1_op,
                            n,
                            m,
                            k,
                            &alpha,
                            static_cast<const bf16*>(input2),
                            CUDA_R_16BF,
                            input2_leading_dim,
                            static_cast<const bf16*>(input1),
                            CUDA_R_16BF,
                            input1_leading_dim,
                            &beta,
                            static_cast<bf16*>(output),
                            CUDA_R_16BF,
                            n,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT);
      break;
    }
    default:
      cublasDestroy(handle);
      return Err<void>(ErrorCode::NotImplemented,
                       "CUDA Matmul only supports Float16/BFloat16/Float32/Float64");
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    return cublas_error(status, "gemm");
  }

  cudaError_t err = stream == nullptr ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    cublasDestroy(handle);
    pulse::error("CUDA matmul synchronization failed for {}: {}",
                 data_type_str(dtype),
                 cudaGetErrorString(err));
    return Err<void>(ErrorCode::CudaError,
                     std::format("CUDA matmul synchronization failed for {}: {}",
                                 data_type_str(dtype),
                                 cudaGetErrorString(err)));
  }

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_error(status, "destroy");
  }

  return Ok();
}

}  // namespace pulse::kernels::cuda
