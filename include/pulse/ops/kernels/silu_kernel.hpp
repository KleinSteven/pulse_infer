#pragma once

#include <cmath>
#include <span>

#include "pulse/core/types.hpp"

namespace pulse::kernels {

template<typename T>
void silu(std::span<const T> input, std::span<T> output, i64 len) noexcept {
    for (i64 i = 0; i < len; ++i) {
        output[i] = input[i] / (static_cast<T>(1) + std::exp(-input[i]));
    }
}

}  // namespace pulse::kernels
