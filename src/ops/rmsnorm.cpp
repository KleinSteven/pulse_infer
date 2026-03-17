#include "pulse/ops/rmsnorm.hpp"

#include <span>

#include "pulse/ops/kernels/rmsnorm_kernel.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/rmsnorm_kernel.cuh"
#endif

namespace pulse::ops {

namespace {

f64 default_rms_norm_eps(DataType dtype) {
    switch (dtype) {
        case DataType::Float16:
            return 9.765625e-4;
        case DataType::BFloat16:
            return 7.8125e-3;
        case DataType::Float32:
            return 1.1920928955078125e-7;
        case DataType::Float64:
            return 2.2204460492503131e-16;
        default:
            return 1.0e-6;
    }
}

usize normalized_size_from_shape(const std::vector<i32>& normalized_shape) noexcept {
    usize size = 1;
    for (i32 dim : normalized_shape) {
        size *= static_cast<usize>(dim);
    }
    return size;
}

template<typename T, typename AccT>
Result<void> rms_norm_cpu(
    const Tensor& input, const Tensor* weight, Tensor& output, i32 rows, i32 normalized_size, AccT eps) {
    std::span<const T> input_span(input.ptr<T>(), input.size());
    std::span<T> output_span(output.ptr<T>(), output.size());
    const T* weight_ptr = weight == nullptr ? nullptr : weight->ptr<T>();
    kernels::rms_norm<T, AccT>(input_span, weight_ptr, output_span, rows, normalized_size, eps);
    return Ok();
}

}  // namespace

Result<void> rms_norm(const Tensor& input,
                      const std::vector<i32>& normalized_shape,
                      const Tensor* weight,
                      Tensor& output,
                      f64 eps) {
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

    if (normalized_shape.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "RMSNorm normalized_shape cannot be empty");
    }

    if (input.ndim() < static_cast<i32>(normalized_shape.size())) {
        return Err<void>(ErrorCode::ShapeMismatch, "RMSNorm normalized_shape rank exceeds input rank");
    }

    const auto input_rank = static_cast<usize>(input.ndim());
    const auto norm_rank = normalized_shape.size();
    for (usize i = 0; i < norm_rank; ++i) {
        const i32 expected_dim = normalized_shape[i];
        if (expected_dim <= 0) {
            return Err<void>(ErrorCode::InvalidArgument, "RMSNorm normalized_shape must be positive");
        }

        const i32 input_dim = input.dim(static_cast<i32>(input_rank - norm_rank + i));
        if (input_dim != expected_dim) {
            return Err<void>(ErrorCode::ShapeMismatch, "RMSNorm normalized_shape must match input suffix");
        }
    }

    if (weight != nullptr) {
        if (weight->device() != input.device()) {
            return Err<void>(ErrorCode::DeviceMismatch, "RMSNorm weight device mismatch");
        }

        if (weight->dtype() != input.dtype()) {
            return Err<void>(ErrorCode::DtypeMismatch, "RMSNorm weight dtype mismatch");
        }

        if (weight->dims() != normalized_shape) {
            return Err<void>(ErrorCode::ShapeMismatch, "RMSNorm weight shape mismatch");
        }
    }

    const f64 resolved_eps = eps < 0.0 ? default_rms_norm_eps(input.dtype()) : eps;
    if (resolved_eps < 0.0) {
        return Err<void>(ErrorCode::InvalidArgument, "RMSNorm eps must be non-negative");
    }

    const usize normalized_size_usize = normalized_size_from_shape(normalized_shape);
    const i32 normalized_size = static_cast<i32>(normalized_size_usize);
    const i32 rows = static_cast<i32>(input.size() / normalized_size_usize);

    if (input.device() == DeviceType::CPU) {
        if (input.dtype() == DataType::Float32) {
            return rms_norm_cpu<f32, f64>(input, weight, output, rows, normalized_size, resolved_eps);
        }

        if (input.dtype() == DataType::Float64) {
            return rms_norm_cpu<f64, f64>(input, weight, output, rows, normalized_size, resolved_eps);
        }

        return Err<void>(ErrorCode::NotImplemented, "CPU RMSNorm only supports Float32/Float64");
    }

#ifdef PULSE_USE_CUDA
    if (input.device() == DeviceType::CUDA) {
        return kernels::cuda::rms_norm_cuda_launch(input.data(),
                                                   weight == nullptr ? nullptr : weight->data(),
                                                   output.data(),
                                                   rows,
                                                   normalized_size,
                                                   resolved_eps,
                                                   input.dtype(),
                                                   nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "RMSNorm operation not implemented for this device");
}

}  // namespace pulse::ops
