#pragma once

#include <cmath>
#include <span>

#include "pulse/core/types.hpp"

namespace pulse::kernels {

template<typename T, typename AccT>
void rms_norm(std::span<const T> input,
              const T* weight,
              std::span<T> output,
              i32 rows,
              i32 normalized_size,
              AccT eps) noexcept {
    for (i32 row = 0; row < rows; ++row) {
        const auto row_base = static_cast<usize>(row) * static_cast<usize>(normalized_size);

        AccT sum_squares = static_cast<AccT>(0);
        for (i32 col = 0; col < normalized_size; ++col) {
            const AccT value = static_cast<AccT>(input[row_base + static_cast<usize>(col)]);
            sum_squares += value * value;
        }

        const AccT inv_rms =
            static_cast<AccT>(1) / std::sqrt(sum_squares / static_cast<AccT>(normalized_size) + eps);

        for (i32 col = 0; col < normalized_size; ++col) {
            const auto idx = row_base + static_cast<usize>(col);
            const AccT value = static_cast<AccT>(input[idx]);
            const AccT scale = weight == nullptr ? static_cast<AccT>(1) : static_cast<AccT>(weight[col]);
            output[idx] = static_cast<T>(value * inv_rms * scale);
        }
    }
}

}  // namespace pulse::kernels
