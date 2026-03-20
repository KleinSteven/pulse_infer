#include "pulse/layer/linear.hpp"

#include "pulse/ops/matmul.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/add_kernel.cuh"
#endif

namespace pulse::layer {

namespace {
template<typename T>
void add_bias_cpu_impl(const Tensor& bias, Tensor& output) {
    const i32 rows = output.dim(0);
    const i32 cols = output.dim(1);
    const T* bias_ptr = bias.ptr<T>();
    T* output_ptr = output.ptr<T>();

    for (i32 row = 0; row < rows; ++row) {
        for (i32 col = 0; col < cols; ++col) {
            output_ptr[static_cast<usize>(row) * static_cast<usize>(cols) + static_cast<usize>(col)] +=
                bias_ptr[col];
        }
    }
}

Result<void> add_bias_inplace(const Tensor& bias, Tensor& output) {
    if (bias.device() != output.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "Linear bias device mismatch");
    }

    if (bias.dtype() != output.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Linear bias dtype mismatch");
    }

    if (bias.dims() != std::vector<i32>{output.dim(1)}) {
        return Err<void>(ErrorCode::ShapeMismatch, "Linear bias shape mismatch");
    }

    if (output.device() == DeviceType::CPU) {
        if (output.dtype() == DataType::Float32) {
            add_bias_cpu_impl<f32>(bias, output);
            return Ok();
        }

        if (output.dtype() == DataType::Float64) {
            add_bias_cpu_impl<f64>(bias, output);
            return Ok();
        }

        return Err<void>(ErrorCode::NotImplemented, "CPU Linear only supports Float32/Float64");
    }

#ifdef PULSE_USE_CUDA
    if (output.device() == DeviceType::CUDA) {
        return kernels::cuda::add_bias_cuda_launch(bias.data(),
                                                   output.data(),
                                                   output.dim(0),
                                                   output.dim(1),
                                                   output.dtype());
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "Linear bias add not implemented for this device");
}

}  // namespace

Result<Linear> Linear::liner(i32 in_features,
                             i32 out_features,
                             bool bias,
                             DeviceType device,
                             DataType dtype,
                             const VarBuilder& builder) {
    Linear linear(in_features, out_features, bias, device, dtype);
    auto init_result = linear.init(builder);
    if (!init_result) {
        return Err<Linear>(std::move(init_result.error()));
    }

    return Ok(std::move(linear));
}

Result<void> Linear::init(const VarBuilder& builder) {
    if (in_features_ <= 0 || out_features_ <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "Linear features must be positive");
    }

    if (device_ == DeviceType::CPU) {
        if (dtype_ != DataType::Float32 && dtype_ != DataType::Float64) {
            return Err<void>(ErrorCode::NotImplemented, "CPU Linear only supports Float32/Float64");
        }
    } else if (device_ == DeviceType::CUDA) {
#ifdef PULSE_USE_CUDA
        if (dtype_ != DataType::Float16 && dtype_ != DataType::BFloat16 && dtype_ != DataType::Float32 &&
            dtype_ != DataType::Float64) {
            return Err<void>(ErrorCode::NotImplemented,
                             "CUDA Linear only supports Float16/BFloat16/Float32/Float64");
        }
#else
        return Err<void>(ErrorCode::NotImplemented, "CUDA support not enabled");
#endif
    } else {
        return Err<void>(ErrorCode::InvalidArgument, "Unsupported Linear device");
    }

    auto weight_result = builder.get("weight", {out_features_, in_features_}, dtype_);
    if (!weight_result) {
        return Err<void>(std::move(weight_result.error()));
    }

    const Tensor* weight_tensor = weight_result.value();
    if (weight_tensor->device() != device_) {
        return Err<void>(ErrorCode::InvalidArgument, "Linear parameter device mismatch");
    }

    const Tensor* bias_tensor = nullptr;
    if (has_bias_) {
        auto bias_result = builder.get("bias", {out_features_}, dtype_);
        if (!bias_result) {
            return Err<void>(std::move(bias_result.error()));
        }

        bias_tensor = bias_result.value();
        if (bias_tensor->device() != device_) {
            return Err<void>(ErrorCode::InvalidArgument, "Linear parameter device mismatch");
        }
    }

    weight_ = weight_tensor;
    bias_ = bias_tensor;
    return Ok();
}

Result<Tensor> Linear::forward(const Tensor& input) const {
    if (weight_ == nullptr) {
        return Err<Tensor>(ErrorCode::InvalidArgument, "Linear layer is not initialized");
    }

    if (input.empty()) {
        return Err<Tensor>(ErrorCode::InvalidArgument, "Linear input tensor is empty");
    }

    if (input.ndim() != 2) {
        return Err<Tensor>(ErrorCode::InvalidArgument, "Linear only supports 2D input tensors");
    }

    if (input.device() != weight_->device()) {
        return Err<Tensor>(ErrorCode::DeviceMismatch, "Linear input device mismatch");
    }

    if (input.dtype() != weight_->dtype()) {
        return Err<Tensor>(ErrorCode::DtypeMismatch, "Linear input dtype mismatch");
    }

    if (input.dim(1) != weight_->dim(1)) {
        return Err<Tensor>(ErrorCode::ShapeMismatch, "Linear input shape mismatch");
    }

    auto output_result = Tensor::create({input.dim(0), weight_->dim(0)}, input.dtype(), input.device());
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

Result<void> Linear::forward(const Tensor& input, Tensor& output) const {
    if (weight_ == nullptr) {
        return Err<void>(ErrorCode::InvalidArgument, "Linear layer is not initialized");
    }

    if (input.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Linear input tensor is empty");
    }

    if (input.ndim() != 2) {
        return Err<void>(ErrorCode::InvalidArgument, "Linear only supports 2D input tensors");
    }

    if (input.device() != weight_->device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "Linear input device mismatch");
    }

    if (input.dtype() != weight_->dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Linear input dtype mismatch");
    }

    if (input.dim(1) != weight_->dim(1)) {
        return Err<void>(ErrorCode::ShapeMismatch, "Linear input shape mismatch");
    }

    if (output.dims() != std::vector<i32>{input.dim(0), weight_->dim(0)}) {
        return Err<void>(ErrorCode::ShapeMismatch, "Linear output shape mismatch");
    }

    if (output.device() != input.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "Linear output device mismatch");
    }

    if (output.dtype() != input.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Linear output dtype mismatch");
    }

    const Tensor* bias = has_bias_ ? bias_ : nullptr;

    auto matmul_result = ops::matmul(input, *weight_, output, false, true);
    if (!matmul_result) {
        return Err<void>(std::move(matmul_result.error()));
    }

    if (bias != nullptr) {
        auto bias_result = add_bias_inplace(*bias, output);
        if (!bias_result) {
            return Err<void>(std::move(bias_result.error()));
        }
    }

    return Ok();
}

}  // namespace pulse::layer
