#pragma once

#include <pulse/core/error.hpp>
#include <pulse/core/tensor.hpp>
#include <pulse/core/types.hpp>

namespace pulse::ops {

[[nodiscard]] Result<void> mha(const Tensor& query,
                               const Tensor& key_cache,
                               const Tensor& value_cache,
                               Tensor& output,
                               i32 pos,
                               i32 head_num,
                               i32 head_size);

}  // namespace pulse::ops
