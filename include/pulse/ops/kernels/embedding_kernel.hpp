#pragma once

#include <cstring>
#include <span>

#include "pulse/core/types.hpp"

namespace pulse::kernels {

template<typename T>
void embedding(std::span<const i32> input,
               std::span<const T> weight,
               std::span<T> output,
               i32 vocab_size,
               i32 embedding_dim) noexcept {
    for (usize token_index = 0; token_index < input.size(); ++token_index) {
        const i32 token = input[token_index];
        if (token < 0 || token >= vocab_size) {
            continue;
        }

        const auto* src = weight.data() + static_cast<usize>(token) * static_cast<usize>(embedding_dim);
        auto* dst = output.data() + token_index * static_cast<usize>(embedding_dim);
        std::memcpy(dst, src, static_cast<usize>(embedding_dim) * sizeof(T));
    }
}

}  // namespace pulse::kernels
