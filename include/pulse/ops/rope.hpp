#pragma once

#include <pulse/core/error.hpp>
#include <pulse/core/tensor.hpp>
#include <pulse/core/types.hpp>

namespace pulse::ops {

[[nodiscard]] Result<void> rope(const Tensor& input,
                                Tensor& output,
                                i32 position_offset = 0,
                                f32 theta = 10000.0f,
                                i32 rotary_dim = -1);

}  // namespace pulse::ops
