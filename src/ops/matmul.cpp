#include "pulse/ops/matmul.hpp"

#include <numeric>
#include <vector>

#include "pulse/ops/kernels/matmul_kernel.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/matmul_kernel.cuh"
#endif

namespace pulse::ops {

[[nodiscard]] Result<std::vector<i32>> infer_matmul_output_dims(const Tensor& input1,
                                                                const Tensor& input2,
                                                                bool transpose_input1,
                                                                bool transpose_input2) {
    const std::vector<i32>& input1_dims = input1.dims();
    const std::vector<i32>& input2_dims = input2.dims();

    if (input1_dims.size() != 2 || input2_dims.size() != 2) {
        return Err<std::vector<i32>>(ErrorCode::InvalidArgument, "Matmul only supports 2D tensors");
    }

    const i32 k = transpose_input1 ? input1.dim(0) : input1.dim(1);
    const i32 rhs_k = transpose_input2 ? input2.dim(1) : input2.dim(0);
    if (k != rhs_k) {
        return Err<std::vector<i32>>(ErrorCode::ShapeMismatch, "Matmul input tensors shape mismatch");
    }

    return Ok(std::vector<i32>{transpose_input1 ? input1.dim(1) : input1.dim(0),
                               transpose_input2 ? input2.dim(0) : input2.dim(1)});
}

[[nodiscard]] Result<void> matmul(const Tensor& input1,
                                  const Tensor& input2,
                                  Tensor& output,
                                  bool transpose_input1,
                                  bool transpose_input2) {
    if (input1.device() != input2.device() || input1.device() != output.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "All tensors must be on the same device");
    }

    if (input1.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Input1 tensor is empty");
    }

    if (input2.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Input2 tensor is empty");
    }

    if (output.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
    }

    if (input1.dtype() != input2.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Input tensors dtype mismatch");
    }

    if (input1.dtype() != output.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Output tensor dtype mismatch");
    }

    if (input1.ndim() != 2 || input2.ndim() != 2) {
        return Err<void>(ErrorCode::InvalidArgument, "Matmul only supports 2D tensors");
    }

    const i32 m = transpose_input1 ? input1.dim(1) : input1.dim(0);
    const i32 k = transpose_input1 ? input1.dim(0) : input1.dim(1);
    const i32 rhs_k = transpose_input2 ? input2.dim(1) : input2.dim(0);
    const i32 n = transpose_input2 ? input2.dim(0) : input2.dim(1);

    if (k != rhs_k) {
        return Err<void>(ErrorCode::ShapeMismatch, "Matmul input tensors shape mismatch");
    }

    if (output.dims() != std::vector<i32>{m, n}) {
        return Err<void>(ErrorCode::ShapeMismatch, "Output tensor shape mismatch");
    }

    if (input1.device() == DeviceType::CPU) {
        if (input1.dtype() == DataType::Float32) {
            std::span<const f32> input1_span(input1.ptr<f32>(), input1.size());
            std::span<const f32> input2_span(input2.ptr<f32>(), input2.size());
            std::span<f32> output_span(output.ptr<f32>(), output.size());
            kernels::matmul<f32>(
                input1_span, input2_span, output_span, m, n, k, transpose_input1, transpose_input2);

            return Ok();
        } else if (input1.dtype() == DataType::Float64) {
            std::span<const f64> input1_span(input1.ptr<f64>(), input1.size());
            std::span<const f64> input2_span(input2.ptr<f64>(), input2.size());
            std::span<f64> output_span(output.ptr<f64>(), output.size());
            kernels::matmul<f64>(
                input1_span, input2_span, output_span, m, n, k, transpose_input1, transpose_input2);

            return Ok();
        } else {
            return Err<void>(ErrorCode::NotImplemented, "NotSupport this dtype");
        }
    }

#ifdef PULSE_USE_CUDA
    if (input1.device() == DeviceType::CUDA) {
        return kernels::cuda::matmul_cuda_launch(
            input1.data(),
            input2.data(),
            output.data(),
            m,
            n,
            k,
            input1.dtype(),
            transpose_input1,
            transpose_input2,
            nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "Matmul operation not implemented for this device");
}

}  // namespace pulse::ops
