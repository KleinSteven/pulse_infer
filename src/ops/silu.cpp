#include "pulse/ops/silu.hpp"

#include "pulse/ops/kernels/silu_kernel.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/silu_kernel.cuh"
#endif

namespace pulse::ops {

[[nodiscard]] Result<void> silu(const Tensor& input, Tensor& output) {
    if (input.device() != output.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "Input and output tensors must be on the same device");
    }

    if (input.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Input tensor is empty");
    }

    if (output.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
    }

    if (input.dims() != output.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Output tensor shape mismatch");
    }

    if (input.dtype() != output.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Output tensor dtype mismatch");
    }

    const i64 size = static_cast<i64>(input.size());

    if (input.device() == DeviceType::CPU) {
        if (input.dtype() == DataType::Float32) {
            std::span<const f32> input_span(input.ptr<f32>(), input.size());
            std::span<f32> output_span(output.ptr<f32>(), output.size());
            kernels::silu<f32>(input_span, output_span, size);

            return Ok();
        } else if (input.dtype() == DataType::Float64) {
            std::span<const f64> input_span(input.ptr<f64>(), input.size());
            std::span<f64> output_span(output.ptr<f64>(), output.size());
            kernels::silu<f64>(input_span, output_span, size);

            return Ok();
        } else {
            return Err<void>(ErrorCode::NotImplemented, "NotSupport this dtype");
        }
    }

#ifdef PULSE_USE_CUDA
    if (input.device() == DeviceType::CUDA) {
        return kernels::cuda::silu_cuda_launch(input.data(), output.data(), size, input.dtype(), nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "SiLU operation not implemented for this device");
}

}  // namespace pulse::ops
