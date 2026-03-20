#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::model {

struct Qwen3Config {
    std::string torch_dtype;
    std::vector<i32> eos_token_ids;
    i32 head_dim = 0;
    i32 hidden_size = 0;
    i32 intermediate_size = 0;
    i32 max_position_embeddings = 0;
    i32 num_attention_heads = 0;
    i32 num_hidden_layers = 0;
    i32 num_key_value_heads = 0;
    f32 rms_norm_eps = 0.0f;
    f64 rope_theta = 0.0;
    bool tie_word_embeddings = false;
    i32 vocab_size = 0;

    f32 temperature = 0.0f;
    i32 top_k = 0;
    f32 top_p = 0.0f;

    [[nodiscard]] static Result<Qwen3Config> load(const std::filesystem::path& path);

    [[nodiscard]] i32 kv_hidden_size() const noexcept {
        return num_key_value_heads * head_dim;
    }
};

}  // namespace pulse::model
