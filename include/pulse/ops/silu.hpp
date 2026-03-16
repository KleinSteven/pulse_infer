#pragma once

#include <pulse/core/error.hpp>
#include <pulse/core/tensor.hpp>
#include <pulse/core/types.hpp>

namespace pulse::ops {

[[nodiscard]] Result<void> silu(const Tensor& input, Tensor& output);

}  // namespace pulse::ops
