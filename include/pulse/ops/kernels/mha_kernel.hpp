#pragma once

#include <algorithm>
#include <cmath>
#include <span>

#include "pulse/core/types.hpp"

namespace pulse::kernels {

template<typename T>
void softmax(std::span<const T> input, std::span<T> output, i32 len) noexcept {
    T max_val = input[0];
    for (i32 i = 1; i < len; ++i) {
        max_val = std::max(max_val, input[i]);
    }

    T sum = static_cast<T>(0);
    for (i32 i = 0; i < len; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    const T inv_sum = static_cast<T>(1) / sum;
    for (i32 i = 0; i < len; ++i) {
        output[i] *= inv_sum;
    }
}

template<typename T>
T dot_product(std::span<const T> lhs, std::span<const T> rhs, i32 len) noexcept {
    T sum = static_cast<T>(0);
    for (i32 i = 0; i < len; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

template<typename T>
void weighted_sum(std::span<const T> value, T weight, std::span<T> output, i32 len) noexcept {
    for (i32 i = 0; i < len; ++i) {
        output[i] += weight * value[i];
    }
}

template<typename T>
void mha(std::span<const T> query,
         std::span<const T> key_cache,
         std::span<const T> value_cache,
         std::span<T> output,
         std::span<T> score,
         i32 pos,
         i32 kv_dim,
         i32 head_num,
         i32 head_size,
         i32 seq_len,
         i32 kv_mul) noexcept {
    const T scale = static_cast<T>(1) / std::sqrt(static_cast<T>(head_size));
    std::fill(output.begin(), output.end(), static_cast<T>(0));

    for (i32 head = 0; head < head_num; ++head) {
        const i32 q_offset = head * head_size;
        const i32 kv_head = head / kv_mul;
        const i32 kv_offset = kv_head * head_size;

        for (i32 t = 0; t <= pos; ++t) {
            const i32 key_base = t * kv_dim + kv_offset;
            const T dot = dot_product<T>(query.subspan(q_offset, head_size),
                                         key_cache.subspan(key_base, head_size),
                                         head_size);
            score[head * seq_len + t] = dot * scale;
        }

        const i32 score_len = pos + 1;
        auto score_view = score.subspan(head * seq_len, score_len);
        softmax<T>(score_view, score_view, score_len);

        for (i32 t = 0; t <= pos; ++t) {
            const i32 value_base = t * kv_dim + kv_offset;
            const T attn_weight = score[head * seq_len + t];
            weighted_sum<T>(value_cache.subspan(value_base, head_size),
                            attn_weight,
                            output.subspan(q_offset, head_size),
                            head_size);
        }
    }
}

}  // namespace pulse::kernels
