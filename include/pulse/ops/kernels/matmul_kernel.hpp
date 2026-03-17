#pragma once

#include <span>

#include "pulse/core/types.hpp"

namespace pulse::kernels {

template<typename T>
void matmul(std::span<const T> input1,
            std::span<const T> input2,
            std::span<T> output,
            i32 m,
            i32 n,
            i32 k,
            bool transpose_input1 = false,
            bool transpose_input2 = false) noexcept {
    const i32 input1_stride = transpose_input1 ? m : k;
    const i32 input2_stride = transpose_input2 ? k : n;

    for (i32 row = 0; row < m; ++row) {
        for (i32 col = 0; col < n; ++col) {
            T sum = static_cast<T>(0);
            for (i32 inner = 0; inner < k; ++inner) {
                const i32 input1_row = transpose_input1 ? inner : row;
                const i32 input1_col = transpose_input1 ? row : inner;
                const i32 input2_row = transpose_input2 ? col : inner;
                const i32 input2_col = transpose_input2 ? inner : col;

                sum += input1[static_cast<usize>(input1_row) * static_cast<usize>(input1_stride) +
                              static_cast<usize>(input1_col)] *
                       input2[static_cast<usize>(input2_row) * static_cast<usize>(input2_stride) +
                              static_cast<usize>(input2_col)];
            }
            output[static_cast<usize>(row) * static_cast<usize>(n) + static_cast<usize>(col)] = sum;
        }
    }
}

}  // namespace pulse::kernels
