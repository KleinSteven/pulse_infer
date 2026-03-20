#include <filesystem>
#include <format>
#include <iostream>
#include <string>

#include "pulse/model/qwen3/config.hpp"

namespace {

template<typename T>
bool check_equal(std::string_view name, const T& actual, const T& expected) {
    const bool matched = actual == expected;
    std::cout << std::format("check {:<22} actual={} expected={} status={}\n",
                             name,
                             actual,
                             expected,
                             matched ? "pass" : "fail");
    return matched;
}

std::filesystem::path find_default_model_path() {
    namespace fs = std::filesystem;

    auto current = fs::current_path();
    while (true) {
        const auto candidate = current / "model";
        if (fs::exists(candidate / "config.json")) {
            return candidate;
        }

        const auto parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }

    return "../../model";
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    const fs::path model_path = argc > 1 ? fs::path(argv[1]) : find_default_model_path();

    auto config_result = pulse::model::Qwen3Config::load(model_path);
    if (!config_result) {
        std::cerr << std::format("failed to load qwen3 config: {}\n", config_result.error().message());
        return 1;
    }

    const auto& config = config_result.value();

    std::cout << std::format("model path              : {}\n", model_path.string());
    std::cout << std::format("hidden_size             : {}\n", config.hidden_size);
    std::cout << std::format("intermediate_size       : {}\n", config.intermediate_size);
    std::cout << std::format("num_hidden_layers       : {}\n", config.num_hidden_layers);
    std::cout << std::format("num_attention_heads     : {}\n", config.num_attention_heads);
    std::cout << std::format("num_key_value_heads     : {}\n", config.num_key_value_heads);
    std::cout << std::format("head_dim                : {}\n", config.head_dim);
    std::cout << std::format("kv_hidden_size          : {}\n", config.kv_hidden_size());
    std::cout << std::format("max_position_embeddings : {}\n", config.max_position_embeddings);
    std::cout << std::format("vocab_size              : {}\n", config.vocab_size);
    std::cout << std::format("eos_token_ids_count     : {}\n", config.eos_token_ids.size());
    std::cout << std::format("torch_dtype             : {}\n", config.torch_dtype);
    std::cout << std::format("temperature             : {}\n", config.temperature);
    std::cout << std::format("top_k                   : {}\n", config.top_k);
    std::cout << std::format("top_p                   : {}\n\n", config.top_p);

    bool all_passed = true;
    all_passed = check_equal("hidden_size", config.hidden_size, 1024) && all_passed;
    all_passed = check_equal("intermediate_size", config.intermediate_size, 3072) && all_passed;
    all_passed = check_equal("num_hidden_layers", config.num_hidden_layers, 28) && all_passed;
    all_passed = check_equal("num_attention_heads", config.num_attention_heads, 16) && all_passed;
    all_passed = check_equal("num_key_value_heads", config.num_key_value_heads, 8) && all_passed;
    all_passed = check_equal("head_dim", config.head_dim, 128) && all_passed;
    all_passed = check_equal("kv_hidden_size", config.kv_hidden_size(), 1024) && all_passed;
    all_passed = check_equal("max_position_embeddings", config.max_position_embeddings, 40960) && all_passed;
    all_passed = check_equal("vocab_size", config.vocab_size, 151936) && all_passed;
    all_passed = check_equal("eos_token_ids_count", static_cast<int>(config.eos_token_ids.size()), 2) && all_passed;
    all_passed = check_equal("torch_dtype", config.torch_dtype, std::string("bfloat16")) && all_passed;
    all_passed = check_equal("temperature", config.temperature, 0.6f) && all_passed;
    all_passed = check_equal("top_k", config.top_k, 20) && all_passed;
    all_passed = check_equal("top_p", config.top_p, 0.95f) && all_passed;

    std::cout << std::format("\nsummary: {}\n", all_passed ? "pass" : "fail");
    return all_passed ? 0 : 1;
}
