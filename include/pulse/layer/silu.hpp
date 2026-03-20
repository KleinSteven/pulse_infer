#pragma once

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"

namespace pulse::layer {

class SiLU {
public:
    [[nodiscard]] Result<void> forward(const Tensor& input, Tensor& output) const;
    [[nodiscard]] Result<Tensor> forward(const Tensor& input) const;
};

}  // namespace pulse::layer
