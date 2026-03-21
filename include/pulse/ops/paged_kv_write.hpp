#pragma once

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"

namespace pulse::ops {

[[nodiscard]] Result<void> paged_kv_write(const Tensor& key,
                                          const Tensor& value,
                                          Tensor& key_cache,
                                          Tensor& value_cache,
                                          const Tensor& block_table,
                                          const Tensor& positions);

}  // namespace pulse::ops
