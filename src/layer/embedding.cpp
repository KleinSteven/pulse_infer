#include "pulse/layer/embedding.hpp"

#include <span>

#include "pulse/ops/kernels/embedding_kernel.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/embedding_kernel.cuh"
#endif

namespace pulse::layer {

namespace {

Result<void> run_embedding_forward(const Tensor& input, const Tensor& weight, Tensor& output) {
    if (input.device() != weight.device() || input.device() != output.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "Embedding tensors must be on the same device");
    }

    if (input.dtype() != DataType::Int32) {
        return Err<void>(ErrorCode::DtypeMismatch, "Embedding input must be Int32");
    }

    if (weight.dtype() != output.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Embedding output dtype mismatch");
    }

    if (input.ndim() != 1) {
        return Err<void>(ErrorCode::ShapeMismatch, "Embedding input must be 1D");
    }

    if (weight.ndim() != 2 || output.ndim() != 2) {
        return Err<void>(ErrorCode::ShapeMismatch, "Embedding weight/output shape mismatch");
    }

    if (output.dim(0) != static_cast<i32>(input.size()) || output.dim(1) != weight.dim(1)) {
        return Err<void>(ErrorCode::ShapeMismatch, "Embedding output shape mismatch");
    }

    if (weight.dim(0) <= 0 || weight.dim(1) <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "Embedding weight dimensions must be positive");
    }

    if (input.device() == DeviceType::CPU) {
        std::span<const i32> input_span(input.ptr<i32>(), input.size());

        if (weight.dtype() == DataType::Float32) {
            std::span<const f32> weight_span(weight.ptr<f32>(), weight.size());
            std::span<f32> output_span(output.ptr<f32>(), output.size());
            kernels::embedding<f32>(input_span, weight_span, output_span, weight.dim(0), weight.dim(1));
            return Ok();
        }

        if (weight.dtype() == DataType::Float64) {
            std::span<const f64> weight_span(weight.ptr<f64>(), weight.size());
            std::span<f64> output_span(output.ptr<f64>(), output.size());
            kernels::embedding<f64>(input_span, weight_span, output_span, weight.dim(0), weight.dim(1));
            return Ok();
        }

        return Err<void>(ErrorCode::NotImplemented, "CPU Embedding only supports Float32/Float64");
    }

#ifdef PULSE_USE_CUDA
    if (input.device() == DeviceType::CUDA) {
        return kernels::cuda::embedding_cuda_launch(input.data(),
                                                    weight.data(),
                                                    output.data(),
                                                    output.dim(0),
                                                    weight.dim(0),
                                                    weight.dim(1),
                                                    weight.dtype(),
                                                    nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "Embedding layer not implemented for this device");
}

}  // namespace

Result<Embedding> Embedding::embedding(i32 num_embeddings,
                                       i32 embedding_dim,
                                       DeviceType device,
                                       DataType dtype,
                                       const VarBuilder& builder) {
    Embedding embedding_layer(num_embeddings, embedding_dim, device, dtype);
    auto init_result = embedding_layer.init(builder);
    if (!init_result) {
        return Err<Embedding>(std::move(init_result.error()));
    }

    return Ok(std::move(embedding_layer));
}

Result<void> Embedding::init(const VarBuilder& builder) {
    if (num_embeddings_ <= 0 || embedding_dim_ <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "Embedding shape must be positive");
    }

    if (device_ == DeviceType::CPU) {
        if (dtype_ != DataType::Float32 && dtype_ != DataType::Float64) {
            return Err<void>(ErrorCode::NotImplemented, "CPU Embedding only supports Float32/Float64");
        }
    } else if (device_ == DeviceType::CUDA) {
#ifdef PULSE_USE_CUDA
        if (dtype_ != DataType::Float16 && dtype_ != DataType::BFloat16 && dtype_ != DataType::Float32 &&
            dtype_ != DataType::Float64) {
            return Err<void>(ErrorCode::NotImplemented,
                             "CUDA Embedding only supports Float16/BFloat16/Float32/Float64");
        }
#else
        return Err<void>(ErrorCode::NotImplemented, "CUDA support not enabled");
#endif
    } else {
        return Err<void>(ErrorCode::InvalidArgument, "Unsupported Embedding device");
    }

    auto weight_result = builder.get("weight", {num_embeddings_, embedding_dim_}, dtype_);
    if (!weight_result) {
        return Err<void>(std::move(weight_result.error()));
    }

    if (weight_result.value()->device() != device_) {
        return Err<void>(ErrorCode::DeviceMismatch, "Embedding parameter device mismatch");
    }

    weight_ = weight_result.value();
    return Ok();
}

Result<Tensor> Embedding::forward(const Tensor& input) const {
    if (weight_ == nullptr) {
        return Err<Tensor>(ErrorCode::InvalidOperator, "Embedding layer is not initialized");
    }

    if (input.ndim() != 1) {
        return Err<Tensor>(ErrorCode::ShapeMismatch, "Embedding input must be 1D");
    }

    auto output_result = Tensor::zeros({static_cast<i32>(input.size()), embedding_dim_}, dtype_, device_);
    if (!output_result) {
        return Err<Tensor>(std::move(output_result.error()));
    }

    Tensor output(std::move(output_result.value()));
    auto embedding_result = run_embedding_forward(input, *weight_, output);
    if (!embedding_result) {
        return Err<Tensor>(std::move(embedding_result.error()));
    }

    return Ok(std::move(output));
}

}  // namespace pulse::layer
