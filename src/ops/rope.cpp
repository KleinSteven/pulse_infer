#include "pulse/ops/rope.hpp"

#include <span>

#include "pulse/ops/kernels/rope_kernel.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/rope_kernel.cuh"
#endif

namespace pulse::ops {

namespace {

template<typename T>
Result<void> rope_cpu(const Tensor& input,
                      Tensor& output,
                      i32 rows,
                      i32 seq_len,
                      i32 head_dim,
                      i32 rotary_dim,
                      i32 position_offset,
                      f32 theta) {
    std::span<const T> input_span(input.ptr<T>(), input.size());
    std::span<T> output_span(output.ptr<T>(), output.size());
    kernels::rope<T>(
        input_span, output_span, rows, seq_len, head_dim, rotary_dim, position_offset, theta);
    return Ok();
}

}  // namespace

Result<void> rope(const Tensor& input, Tensor& output, i32 position_offset, f32 theta, i32 rotary_dim) {
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

    if (input.ndim() < 2) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE requires input tensor ndim >= 2");
    }

    if (position_offset < 0) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE position_offset must be non-negative");
    }

    if (!(theta > 0.0f)) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE theta must be positive");
    }

    const i32 seq_len = input.dim(input.ndim() - 2);
    const i32 head_dim = input.dim(input.ndim() - 1);
    if (head_dim <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE head_dim must be positive");
    }

    if (rotary_dim < 0) {
        rotary_dim = head_dim;
    }

    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE rotary_dim must be in range (0, head_dim]");
    }

    if ((rotary_dim % 2) != 0) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE rotary_dim must be even");
    }

    const i64 rows_i64 = static_cast<i64>(input.size()) /
                         (static_cast<i64>(seq_len) * static_cast<i64>(head_dim));
    const i32 rows = static_cast<i32>(rows_i64);

    if (input.device() == DeviceType::CPU) {
        if (input.dtype() == DataType::Float32) {
            return rope_cpu<f32>(
                input, output, rows, seq_len, head_dim, rotary_dim, position_offset, theta);
        }

        if (input.dtype() == DataType::Float64) {
            return rope_cpu<f64>(
                input, output, rows, seq_len, head_dim, rotary_dim, position_offset, theta);
        }

        return Err<void>(ErrorCode::NotImplemented, "CPU RoPE only supports Float32/Float64");
    }

#ifdef PULSE_USE_CUDA
    if (input.device() == DeviceType::CUDA) {
        return kernels::cuda::rope_cuda_launch(input.data(),
                                               output.data(),
                                               rows,
                                               seq_len,
                                               head_dim,
                                               rotary_dim,
                                               position_offset,
                                               theta,
                                               input.dtype(),
                                               nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "RoPE operation not implemented for this device");
}

}  // namespace pulse::ops
