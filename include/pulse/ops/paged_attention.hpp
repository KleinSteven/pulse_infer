#pragma once

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"

namespace pulse::ops {

[[nodiscard]] Result<void> paged_attention(const Tensor& query,
                                           const Tensor& key_cache,
                                           const Tensor& value_cache,
                                           const Tensor& block_table,
                                           const Tensor& context_lens,
                                           Tensor& output);

}  // namespace pulse::ops
