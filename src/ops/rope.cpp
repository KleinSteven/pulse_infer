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

template<typename T>
Result<void> rope_positions_cpu(const Tensor& input,
                                Tensor& output,
                                const Tensor& positions,
                                i32 batch_size,
                                i32 rows_per_batch,
                                i32 head_dim,
                                i32 rotary_dim,
                                f32 theta) {
    const auto* input_ptr = input.ptr<T>();
    auto* output_ptr = output.ptr<T>();
    const auto* positions_ptr = positions.ptr<i32>();
    if (input_ptr == nullptr || output_ptr == nullptr || positions_ptr == nullptr) {
        return Err<void>(ErrorCode::NullPointer, "RoPE positions CPU pointer is null");
    }

    for (i32 batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const i32 position = positions_ptr[batch_idx];
        for (i32 row = 0; row < rows_per_batch; ++row) {
            const i64 row_base =
                (static_cast<i64>(batch_idx) * rows_per_batch + row) * static_cast<i64>(head_dim);
            for (i32 dim = 0; dim < head_dim; ++dim) {
                if (dim >= rotary_dim) {
                    output_ptr[row_base + dim] = input_ptr[row_base + dim];
                    continue;
                }

                const i32 pair_base_dim = dim & ~1;
                const i64 pair_base_idx = row_base + pair_base_dim;
                const f32 x0 = static_cast<f32>(input_ptr[pair_base_idx]);
                const f32 x1 = static_cast<f32>(input_ptr[pair_base_idx + 1]);
                const f32 exponent = static_cast<f32>(pair_base_dim) / static_cast<f32>(rotary_dim);
                const f32 angle = static_cast<f32>(position) / std::pow(theta, exponent);
                const f32 cos_value = std::cos(angle);
                const f32 sin_value = std::sin(angle);
                const f32 rotated =
                    (dim & 1) == 0 ? (x0 * cos_value - x1 * sin_value) : (x0 * sin_value + x1 * cos_value);
                output_ptr[row_base + dim] = static_cast<T>(rotated);
            }
        }
    }

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

Result<void> rope(const Tensor& input, Tensor& output, const Tensor& positions, f32 theta, i32 rotary_dim) {
    if (input.device() != output.device() || input.device() != positions.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "RoPE position tensors must be on the same device");
    }

    if (input.dims() != output.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "RoPE output tensor shape mismatch");
    }

    if (input.dtype() != output.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "RoPE output tensor dtype mismatch");
    }

    if (positions.dtype() != DataType::Int32) {
        return Err<void>(ErrorCode::DtypeMismatch, "RoPE positions must be Int32");
    }

    if (input.ndim() != 3) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE positions overload requires a 3D input tensor");
    }

    if (positions.ndim() != 1 || positions.dim(0) != input.dim(0)) {
        return Err<void>(ErrorCode::ShapeMismatch, "RoPE positions tensor shape mismatch");
    }

    if (!(theta > 0.0f)) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE theta must be positive");
    }

    const i32 batch_size = input.dim(0);
    const i32 rows_per_batch = input.dim(1);
    const i32 head_dim = input.dim(2);

    if (rotary_dim < 0) {
        rotary_dim = head_dim;
    }

    if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
        return Err<void>(ErrorCode::InvalidArgument, "RoPE rotary_dim must be even and in range (0, head_dim]");
    }

    if (input.device() == DeviceType::CPU) {
        if (input.dtype() == DataType::Float32) {
            return rope_positions_cpu<f32>(
                input, output, positions, batch_size, rows_per_batch, head_dim, rotary_dim, theta);
        }

        if (input.dtype() == DataType::Float64) {
            return rope_positions_cpu<f64>(
                input, output, positions, batch_size, rows_per_batch, head_dim, rotary_dim, theta);
        }

        return Err<void>(ErrorCode::NotImplemented, "CPU RoPE positions only supports Float32/Float64");
    }

#ifdef PULSE_USE_CUDA
    if (input.device() == DeviceType::CUDA) {
        return kernels::cuda::rope_positions_cuda_launch(input.data(),
                                                         output.data(),
                                                         positions.ptr<i32>(),
                                                         batch_size,
                                                         rows_per_batch,
                                                         head_dim,
                                                         rotary_dim,
                                                         theta,
                                                         input.dtype(),
                                                         nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "RoPE positions operation not implemented for this device");
}

}  // namespace pulse::ops
