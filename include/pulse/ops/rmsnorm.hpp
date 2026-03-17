#pragma once

#include <vector>

#include <pulse/core/error.hpp>
#include <pulse/core/tensor.hpp>
#include <pulse/core/types.hpp>

namespace pulse::ops {

[[nodiscard]] Result<void> rms_norm(const Tensor& input,
                                    const std::vector<i32>& normalized_shape,
                                    const Tensor* weight,
                                    Tensor& output,
                                    f64 eps = -1.0);

}  // namespace pulse::ops
