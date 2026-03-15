#pragma once

#include <pulse/core/error.hpp>
#include <pulse/core/tensor.hpp>
#include <pulse/core/types.hpp>

namespace pulse::ops {

[[nodiscard]] Result<void> add(const Tensor& input1, const Tensor& input2, Tensor& output);

}  // namespace pulse::ops
