#include "pulse/layer/rmsnorm.hpp"

#include <format>

#include "pulse/ops/rmsnorm.hpp"

namespace pulse::layer {

RMSNorm::RMSNorm(std::vector<i32> normalized_shape,
                 f64 eps,
                 bool elementwise_affine,
                 DeviceType device,
                 DataType dtype) noexcept
    : normalized_shape_(std::move(normalized_shape)),
      eps_(eps < 0.0 ? resolve_default_eps(dtype) : eps),
      elementwise_affine_(elementwise_affine),
      device_(device),
      dtype_(dtype) {}

f64 RMSNorm::resolve_default_eps(DataType dtype) noexcept {
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

Result<void> RMSNorm::init(const VarBuilder& builder) {
    if (normalized_shape_.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "RMSNorm normalized_shape cannot be empty");
    }

    for (i32 dim : normalized_shape_) {
        if (dim <= 0) {
            return Err<void>(ErrorCode::InvalidArgument, "RMSNorm normalized_shape must be positive");
        }
    }

    if (eps_ < 0.0) {
        return Err<void>(ErrorCode::InvalidArgument, "RMSNorm eps must be non-negative");
    }

    if (device_ == DeviceType::CPU) {
        if (dtype_ != DataType::Float32 && dtype_ != DataType::Float64) {
            return Err<void>(ErrorCode::NotImplemented, "CPU RMSNorm only supports Float32/Float64");
        }
    } else if (device_ == DeviceType::CUDA) {
#ifdef PULSE_USE_CUDA
        if (dtype_ != DataType::Float16 && dtype_ != DataType::BFloat16 && dtype_ != DataType::Float32 &&
            dtype_ != DataType::Float64) {
            return Err<void>(ErrorCode::NotImplemented,
                             "CUDA RMSNorm only supports Float16/BFloat16/Float32/Float64");
        }
#else
        return Err<void>(ErrorCode::NotImplemented, "CUDA support not enabled");
#endif
    } else {
        return Err<void>(ErrorCode::InvalidArgument, "Unsupported RMSNorm device");
    }

    if (!elementwise_affine_) {
        weight_ = nullptr;
        return Ok();
    }

    auto weight_result = builder.get("weight", normalized_shape_, dtype_);
    if (!weight_result) {
        return Err<void>(std::move(weight_result.error()));
    }

    const Tensor* weight = weight_result.value();
    if (weight->device() != device_) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("RMSNorm weight device mismatch: expected {}, got {}",
                                     device_type_str(device_),
                                     device_type_str(weight->device())));
    }

    weight_ = weight;
    return Ok();
}

Result<Tensor> RMSNorm::forward(const Tensor& input) const {
    auto output_result = Tensor::create(input.dims(), input.dtype(), input.device());
    if (!output_result) {
        return Err<Tensor>(std::move(output_result.error()));
    }

    Tensor output(std::move(output_result.value()));
    auto forward_result = forward(input, output);
    if (!forward_result) {
        return Err<Tensor>(std::move(forward_result.error()));
    }

    return Ok(std::move(output));
}

Result<void> RMSNorm::forward(const Tensor& input, Tensor& output) const {
    if (normalized_shape_.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "RMSNorm normalized_shape cannot be empty");
    }

    if (input.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "RMSNorm input tensor is empty");
    }

    if (input.device() != device_) {
        return Err<void>(ErrorCode::DeviceMismatch, "RMSNorm input device mismatch");
    }

    if (input.dtype() != dtype_) {
        return Err<void>(ErrorCode::DtypeMismatch, "RMSNorm input dtype mismatch");
    }

    if (input.ndim() < static_cast<i32>(normalized_shape_.size())) {
        return Err<void>(ErrorCode::ShapeMismatch, "RMSNorm input rank is smaller than normalized_shape");
    }

    const auto input_rank = static_cast<usize>(input.ndim());
    const auto norm_rank = normalized_shape_.size();
    for (usize i = 0; i < norm_rank; ++i) {
        if (input.dim(static_cast<i32>(input_rank - norm_rank + i)) != normalized_shape_[i]) {
            return Err<void>(ErrorCode::ShapeMismatch, "RMSNorm input suffix shape mismatch");
        }
    }

    if (elementwise_affine_ && weight_ == nullptr) {
        return Err<void>(ErrorCode::InvalidArgument, "RMSNorm layer is not initialized");
    }

    if (output.dims() != input.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "RMSNorm output shape mismatch");
    }

    if (output.device() != input.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "RMSNorm output device mismatch");
    }

    if (output.dtype() != input.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "RMSNorm output dtype mismatch");
    }

    auto norm_result = ops::rms_norm(
        input, normalized_shape_, elementwise_affine_ ? weight_ : nullptr, output, eps_);
    if (!norm_result) {
        return Err<void>(std::move(norm_result.error()));
    }

    return Ok();
}

}  // namespace pulse::layer
