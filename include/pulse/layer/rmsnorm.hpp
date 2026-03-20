#pragma once

#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/var.hpp"

namespace pulse::layer {

class RMSNorm {
public:
    RMSNorm(std::vector<i32> normalized_shape,
            f64 eps = -1.0,
            bool elementwise_affine = true,
            DeviceType device = DeviceType::CPU,
            DataType dtype = DataType::Float32) noexcept;

    [[nodiscard]] Result<void> init(const VarBuilder& builder);

    [[nodiscard]] Result<void> forward(const Tensor& input, Tensor& output) const;
    [[nodiscard]] Result<Tensor> forward(const Tensor& input) const;

    [[nodiscard]] const std::vector<i32>& normalized_shape() const noexcept {
        return normalized_shape_;
    }

    [[nodiscard]] f64 eps() const noexcept {
        return eps_;
    }

    [[nodiscard]] bool elementwise_affine() const noexcept {
        return elementwise_affine_;
    }

    [[nodiscard]] const Tensor* weight() const noexcept {
        return weight_;
    }

    [[nodiscard]] DeviceType device() const noexcept {
        return device_;
    }

    [[nodiscard]] DataType dtype() const noexcept {
        return dtype_;
    }

private:
    [[nodiscard]] static f64 resolve_default_eps(DataType dtype) noexcept;

    const Tensor* weight_ = nullptr;
    std::vector<i32> normalized_shape_;
    f64 eps_ = 0.0;
    bool elementwise_affine_ = true;
    DeviceType device_ = DeviceType::CPU;
    DataType dtype_ = DataType::Float32;
};

}  // namespace pulse::layer
