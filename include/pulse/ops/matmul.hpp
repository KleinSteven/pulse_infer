#pragma once

#include <pulse/core/error.hpp>
#include <pulse/core/tensor.hpp>
#include <pulse/core/types.hpp>

namespace pulse::ops {

[[nodiscard]] Result<std::vector<i32>> infer_matmul_output_dims(const Tensor& input1,
                                                                const Tensor& input2,
                                                                bool transpose_input1 = false,
                                                                bool transpose_input2 = false);

[[nodiscard]] Result<void> matmul(const Tensor& input1,
                                  const Tensor& input2,
                                  Tensor& output,
                                  bool transpose_input1 = false,
                                  bool transpose_input2 = false);

}  // namespace pulse::ops
