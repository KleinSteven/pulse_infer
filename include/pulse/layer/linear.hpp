#pragma once

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/var.hpp"

namespace pulse::layer {

class Linear {
public:
    [[nodiscard]] static Result<Linear> liner(i32 in_features,
                                              i32 out_features,
                                              bool bias,
                                              DeviceType device,
                                              DataType dtype,
                                              const VarBuilder& builder);

    Linear(const Linear&) = delete;
    Linear& operator=(const Linear&) = delete;

    Linear(Linear&&) = default;
    Linear& operator=(Linear&&) = default;

    [[nodiscard]] Result<void> forward(const Tensor& input, Tensor& output) const;
    [[nodiscard]] Result<Tensor> forward(const Tensor& input) const;

    [[nodiscard]] const Tensor* weight() const noexcept {
        return weight_;
    }

    [[nodiscard]] const Tensor* bias() const noexcept {
        return has_bias_ ? bias_ : nullptr;
    }

    [[nodiscard]] bool has_bias() const noexcept {
        return has_bias_;
    }

    [[nodiscard]] i32 in_features() const noexcept {
        return in_features_;
    }

    [[nodiscard]] i32 out_features() const noexcept {
        return out_features_;
    }

private:
    Linear(i32 in_features, i32 out_features, bool bias, DeviceType device, DataType dtype) noexcept
        : in_features_(in_features),
          out_features_(out_features),
          has_bias_(bias),
          dtype_(dtype),
          device_(device) {}

    [[nodiscard]] Result<void> init(const VarBuilder& builder);

    const Tensor* weight_ = nullptr;
    const Tensor* bias_ = nullptr;
    i32 in_features_ = 0;
    i32 out_features_ = 0;
    bool has_bias_ = false;
    DataType dtype_ = DataType::Float32;
    DeviceType device_ = DeviceType::CPU;
};

}  // namespace pulse::layer
