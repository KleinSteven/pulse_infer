#pragma once

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"

namespace pulse::layer {

class SiLU {
public:
    [[nodiscard]] Result<Tensor> forward(const Tensor& input) const;

    [[nodiscard]] Result<Tensor> operator()(const Tensor& input) const {
        return forward(input);
    }
};

}  // namespace pulse::layer
