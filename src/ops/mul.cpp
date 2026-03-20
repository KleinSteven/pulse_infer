#include "pulse/ops/mul.hpp"

#include "pulse/ops/kernels/mul_kernel.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/mul_kernel.cuh"
#endif

namespace pulse::ops {

Result<void> mul(const Tensor& input1, const Tensor& input2, Tensor& output) {
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

    if (input1.dims() != input2.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Input tensors shape mismatch");
    }

    if (input1.dims() != output.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Output tensor shape mismatch");
    }

    if (input1.dtype() != input2.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Input tensors dtype mismatch");
    }

    if (input1.dtype() != output.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Output tensor dtype mismatch");
    }

    const i64 size = static_cast<i64>(input1.size());

    if (input1.device() == DeviceType::CPU) {
        if (input1.dtype() == DataType::Float32) {
            std::span<const f32> input1_span(input1.ptr<f32>(), input1.size());
            std::span<const f32> input2_span(input2.ptr<f32>(), input2.size());
            std::span<f32> output_span(output.ptr<f32>(), output.size());
            kernels::mul<f32>(input1_span, input2_span, output_span, size);
            return Ok();
        }

        if (input1.dtype() == DataType::Float64) {
            std::span<const f64> input1_span(input1.ptr<f64>(), input1.size());
            std::span<const f64> input2_span(input2.ptr<f64>(), input2.size());
            std::span<f64> output_span(output.ptr<f64>(), output.size());
            kernels::mul<f64>(input1_span, input2_span, output_span, size);
            return Ok();
        }

        return Err<void>(ErrorCode::NotImplemented, "NotSupport this dtype");
    }

#ifdef PULSE_USE_CUDA
    if (input1.device() == DeviceType::CUDA) {
        return kernels::cuda::mul_cuda_launch(
            input1.data(), input2.data(), output.data(), size, input1.dtype(), nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "Mul operation not implemented for this device");
}

}  // namespace pulse::ops
