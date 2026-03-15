#pragma once

#include <span>

#include "pulse/core/types.hpp"

namespace pulse::kernels {

template<typename T>
void add(std::span<const T> input1, std::span<const T> input2, std::span<T> output, i64 len) noexcept {
    for (i64 i = 0; i < len; i++) {
        output[i] = input1[i] + input2[i];
    }
}

}  // namespace pulse::kernels
