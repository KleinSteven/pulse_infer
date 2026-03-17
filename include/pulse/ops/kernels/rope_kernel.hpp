#pragma once

#include <cmath>
#include <span>

#include "pulse/core/types.hpp"

namespace pulse::kernels {

template<typename T>
void rope(std::span<const T> input,
          std::span<T> output,
          i32 rows,
          i32 seq_len,
          i32 head_dim,
          i32 rotary_dim,
          i32 position_offset,
          double theta) noexcept {
    const auto row_stride = static_cast<usize>(seq_len) * static_cast<usize>(head_dim);
    const auto rotary_pairs = rotary_dim / 2;

    for (i32 row = 0; row < rows; ++row) {
        const auto row_base = static_cast<usize>(row) * row_stride;
        for (i32 pos = 0; pos < seq_len; ++pos) {
            const auto token_base = row_base + static_cast<usize>(pos) * static_cast<usize>(head_dim);
            const auto absolute_pos = static_cast<double>(position_offset) + static_cast<double>(pos);

            for (i32 pair_idx = 0; pair_idx < rotary_pairs; ++pair_idx) {
                const auto even_idx = token_base + static_cast<usize>(pair_idx * 2);
                const auto odd_idx = even_idx + 1;
                const auto exponent = static_cast<double>(pair_idx * 2) / static_cast<double>(rotary_dim);
                const auto angle = absolute_pos / std::pow(theta, exponent);
                const auto cos_value = std::cos(angle);
                const auto sin_value = std::sin(angle);

                const auto x0 = static_cast<double>(input[even_idx]);
                const auto x1 = static_cast<double>(input[odd_idx]);

                output[even_idx] = static_cast<T>(x0 * cos_value - x1 * sin_value);
                output[odd_idx] = static_cast<T>(x0 * sin_value + x1 * cos_value);
            }

            for (i32 dim = rotary_dim; dim < head_dim; ++dim) {
                const auto idx = token_base + static_cast<usize>(dim);
                output[idx] = input[idx];
            }
        }
    }
}

}  // namespace pulse::kernels
