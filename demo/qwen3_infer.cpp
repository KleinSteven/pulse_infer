#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "pulse/core/var.hpp"
#include "pulse/model/qwen3/config.hpp"
#include "pulse/model/qwen3/model.hpp"
#include "pulse/model/qwen3/tokenizer.hpp"

#ifdef PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

using pulse::DataType;
using pulse::DeviceType;
using pulse::Tensor;
using pulse::VarBuilder;
using pulse::VarMap;
using pulse::bf16;
using pulse::f32;
using pulse::i32;
using pulse::model::Qwen3Config;
using pulse::model::Qwen3Model;
using pulse::model::Qwen3Tokenizer;

struct GenerationState {
    i32 current_pos = 0;
    std::vector<i32> all_tokens;
    Tensor logits;
};

std::filesystem::path find_default_model_path() {
    namespace fs = std::filesystem;

    auto current = fs::current_path();
    while (true) {
        const auto candidate = current / "model";
        if (fs::exists(candidate / "config.json") && fs::exists(candidate / "model.safetensors")) {
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

std::vector<std::filesystem::path> find_safetensor_files(const std::filesystem::path& model_dir) {
    namespace fs = std::filesystem;

    std::vector<fs::path> paths;
    if (!fs::exists(model_dir) || !fs::is_directory(model_dir)) {
        return paths;
    }

    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const auto extension = entry.path().extension().string();
        if (extension == ".safetensor" || extension == ".safetensors") {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

pulse::Result<VarMap> load_model_vars(const std::filesystem::path& model_dir) {
    auto safetensor_files = find_safetensor_files(model_dir);
    if (safetensor_files.empty()) {
        return pulse::Err<VarMap>(pulse::ErrorCode::OpenFileError,
                                  std::format("No safetensor file found in '{}'", model_dir.string()));
    }

    VarMap merged;
    for (const auto& path : safetensor_files) {
        auto shard_result = VarMap::from_safetensors(path, DeviceType::CUDA);
        if (!shard_result) {
            return pulse::Err<VarMap>(std::move(shard_result.error()));
        }

        auto append_result = merged.append(std::move(shard_result.value()));
        if (!append_result) {
            return pulse::Err<VarMap>(std::move(append_result.error()));
        }
    }

    return pulse::Ok(std::move(merged));
}

bool is_stop_token(i32 token_id, const Qwen3Config& config, const Qwen3Tokenizer& tokenizer) {
    if (token_id == tokenizer.eos_id()) {
        return true;
    }

    return std::ranges::find(config.eos_token_ids, token_id) != config.eos_token_ids.end();
}

std::size_t valid_utf8_prefix_length(std::string_view text) {
    std::size_t index = 0;
    while (index < text.size()) {
        const auto byte0 = static_cast<unsigned char>(text[index]);
        std::size_t width = 0;
        if ((byte0 & 0x80u) == 0u) {
            width = 1;
        } else if ((byte0 & 0xE0u) == 0xC0u) {
            width = 2;
        } else if ((byte0 & 0xF0u) == 0xE0u) {
            width = 3;
        } else if ((byte0 & 0xF8u) == 0xF0u) {
            width = 4;
        } else {
            // Invalid leading byte. Keep it printable instead of buffering forever.
            width = 1;
        }

        if (index + width > text.size()) {
            break;
        }

        bool valid = true;
        for (std::size_t offset = 1; offset < width; ++offset) {
            const auto continuation = static_cast<unsigned char>(text[index + offset]);
            if ((continuation & 0xC0u) != 0x80u) {
                valid = false;
                width = 1;
                break;
            }
        }

        index += width;
        if (!valid) {
            continue;
        }
    }

    return index;
}

std::string strip_thinking_content(std::string_view text) {
    std::string visible;
    visible.reserve(text.size());

    bool in_think_block = false;
    std::size_t offset = 0;
    while (offset < text.size()) {
        if (!in_think_block) {
            const auto think_start = text.find("<think>", offset);
            if (think_start == std::string_view::npos) {
                visible.append(text.substr(offset));
                break;
            }

            visible.append(text.substr(offset, think_start - offset));
            offset = think_start + std::string_view("<think>").size();
            in_think_block = true;
            continue;
        }

        const auto think_end = text.find("</think>", offset);
        if (think_end == std::string_view::npos) {
            break;
        }

        offset = think_end + std::string_view("</think>").size();
        in_think_block = false;
    }

    return visible;
}

void flush_streaming_answer(const Qwen3Tokenizer& tokenizer,
                            const std::vector<i32>& generated_tokens,
                            std::size_t& printed_bytes) {
    const auto decoded = tokenizer.decode(generated_tokens);
    const auto visible_text = strip_thinking_content(decoded);
    const auto printable_bytes = valid_utf8_prefix_length(visible_text);
    if (printable_bytes <= printed_bytes) {
        return;
    }

    std::cout.write(visible_text.data() + static_cast<std::streamoff>(printed_bytes),
                    static_cast<std::streamsize>(printable_bytes - printed_bytes));
    std::cout.flush();
    printed_bytes = printable_bytes;
}

#ifdef PULSE_USE_CUDA
f32 scalar_to_float(bf16 value) {
    return __bfloat162float(value);
}
#endif

std::vector<f32> logits_to_float_vector(const Tensor& logits) {
    if (logits.device() != DeviceType::CPU) {
        auto cpu_logits_result = logits.to(DeviceType::CPU);
        if (!cpu_logits_result) {
            return {};
        }

        return logits_to_float_vector(cpu_logits_result.value());
    }

    std::vector<f32> values;
    values.reserve(logits.size());

    if (logits.dtype() != DataType::BFloat16) {
        return {};
    }

    const auto* ptr = logits.ptr<bf16>();
    for (pulse::usize index = 0; index < logits.size(); ++index) {
        values.push_back(scalar_to_float(ptr[index]));
    }

    return values;
}

i32 sample_token(const Tensor& logits,
                 f32 temperature,
                 i32 top_k,
                 f32 top_p,
                 const std::vector<i32>& banned_tokens,
                 std::mt19937& rng,
                 bool use_greedy = false,
                 const std::vector<i32>* history_tokens = nullptr,
                 f32 presence_penalty = 0.0f,
                 std::vector<std::pair<i32, f32>>* top_candidates = nullptr) {
    struct TokenCandidate {
        i32 token_id = -1;
        f32 logit = 0.0f;
    };

    auto logits_fp32 = logits_to_float_vector(logits);
    const auto vocab_size = static_cast<i32>(logits_fp32.size());
    if (vocab_size <= 0) {
        return -1;
    }

    if (history_tokens != nullptr && presence_penalty > 0.0f) {
        std::vector<bool> seen(static_cast<std::size_t>(vocab_size), false);
        for (i32 token_id : *history_tokens) {
            if (token_id >= 0 && token_id < vocab_size) {
                seen[static_cast<std::size_t>(token_id)] = true;
            }
        }

        for (i32 token_id = 0; token_id < vocab_size; ++token_id) {
            if (seen[static_cast<std::size_t>(token_id)]) {
                logits_fp32[static_cast<std::size_t>(token_id)] -= presence_penalty;
            }
        }
    }

    const auto is_banned = [&](i32 token_id) {
        return std::binary_search(banned_tokens.begin(), banned_tokens.end(), token_id);
    };

    const auto desc_by_logit = [](const TokenCandidate& lhs, const TokenCandidate& rhs) {
        return lhs.logit > rhs.logit;
    };
    const auto min_heap_by_logit = [](const TokenCandidate& lhs, const TokenCandidate& rhs) {
        return lhs.logit > rhs.logit;
    };

    if (use_greedy || temperature <= 0.0f) {
        TokenCandidate best_candidate{};
        bool has_candidate = false;
        std::vector<TokenCandidate> debug_top_candidates;
        if (top_candidates != nullptr) {
            debug_top_candidates.reserve(5);
        }

        for (i32 token_id = 0; token_id < vocab_size; ++token_id) {
            if (is_banned(token_id)) {
                continue;
            }

            const auto candidate =
                TokenCandidate{token_id, logits_fp32[static_cast<std::size_t>(token_id)]};
            if (!has_candidate || candidate.logit > best_candidate.logit) {
                best_candidate = candidate;
                has_candidate = true;
            }

            if (top_candidates == nullptr) {
                continue;
            }

            if (debug_top_candidates.size() < 5) {
                debug_top_candidates.push_back(candidate);
                std::push_heap(debug_top_candidates.begin(),
                               debug_top_candidates.end(),
                               min_heap_by_logit);
                continue;
            }

            if (candidate.logit > debug_top_candidates.front().logit) {
                std::pop_heap(debug_top_candidates.begin(),
                              debug_top_candidates.end(),
                              min_heap_by_logit);
                debug_top_candidates.back() = candidate;
                std::push_heap(debug_top_candidates.begin(),
                               debug_top_candidates.end(),
                               min_heap_by_logit);
            }
        }

        if (!has_candidate) {
            return -1;
        }

        if (top_candidates != nullptr) {
            std::sort(debug_top_candidates.begin(), debug_top_candidates.end(), desc_by_logit);
            top_candidates->clear();
            for (const auto& candidate : debug_top_candidates) {
                top_candidates->push_back({candidate.token_id, candidate.logit});
            }
        }

        return best_candidate.token_id;
    }

    std::vector<TokenCandidate> candidates;
    if (top_k > 0) {
        const auto keep_count = static_cast<std::size_t>(top_k);
        candidates.reserve(keep_count);

        for (i32 token_id = 0; token_id < vocab_size; ++token_id) {
            if (is_banned(token_id)) {
                continue;
            }

            const auto candidate =
                TokenCandidate{token_id, logits_fp32[static_cast<std::size_t>(token_id)]};
            if (candidates.size() < keep_count) {
                candidates.push_back(candidate);
                std::push_heap(candidates.begin(), candidates.end(), min_heap_by_logit);
                continue;
            }

            if (candidate.logit > candidates.front().logit) {
                std::pop_heap(candidates.begin(), candidates.end(), min_heap_by_logit);
                candidates.back() = candidate;
                std::push_heap(candidates.begin(), candidates.end(), min_heap_by_logit);
            }
        }
    } else {
        candidates.reserve(static_cast<std::size_t>(vocab_size));
        for (i32 token_id = 0; token_id < vocab_size; ++token_id) {
            if (!is_banned(token_id)) {
                candidates.push_back({token_id, logits_fp32[static_cast<std::size_t>(token_id)]});
            }
        }
    }

    if (candidates.empty()) {
        return -1;
    }

    std::sort(candidates.begin(), candidates.end(), desc_by_logit);

    if (top_candidates != nullptr) {
        top_candidates->clear();
        const auto limit = std::min<std::size_t>(5, candidates.size());
        for (std::size_t index = 0; index < limit; ++index) {
            top_candidates->push_back({candidates[index].token_id, candidates[index].logit});
        }
    }

    const auto max_logit = candidates.front().logit;

    std::vector<f32> weights;
    weights.reserve(candidates.size());

    double weight_sum = 0.0;
    for (const auto& candidate : candidates) {
        const auto scaled = (candidate.logit - max_logit) / temperature;
        const auto weight = static_cast<f32>(std::exp(static_cast<double>(scaled)));
        weights.push_back(weight);
        weight_sum += static_cast<double>(weight);
    }

    if (top_p > 0.0f && top_p < 1.0f && !weights.empty()) {
        double cumulative = 0.0;
        std::size_t keep_count = 0;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            cumulative += static_cast<double>(weights[index]) / weight_sum;
            keep_count = index + 1;
            if (cumulative >= top_p) {
                break;
            }
        }

        candidates.resize(keep_count);
        weights.resize(keep_count);
        weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    }

    std::uniform_real_distribution<double> dist(0.0, weight_sum);
    const auto draw = dist(rng);

    double cumulative = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        cumulative += static_cast<double>(weights[index]);
        if (draw <= cumulative) {
            return candidates[index].token_id;
        }
    }

    return candidates.back().token_id;
}

pulse::Result<void> forward_append_token(Qwen3Model& model, GenerationState& state, i32 token_id) {
    state.all_tokens.push_back(token_id);
    auto forward_result = model.forward(token_id, state.current_pos, state.logits);
    if (!forward_result) {
        return forward_result;
    }

    state.current_pos++;
    return pulse::Ok();
}

std::string make_chat_prompt(std::string_view question) {
    return std::format("Question: {}\nAnswer: ", question);
}

pulse::Result<i32> generate_turn(Qwen3Model& model,
                                 const Qwen3Tokenizer& tokenizer,
                                 f32 temperature,
                                 i32 top_k,
                                 f32 top_p,
                                 const Qwen3Config& config,
                                 GenerationState& state,
                                 std::string_view prompt_text,
                                 i32 max_new_tokens,
                                 const std::vector<i32>& banned_tokens,
                                 std::mt19937& rng,
                                 bool use_greedy = false,
                                 bool debug_tokens = false,
                                 bool print_output = true) {
    const auto prompt_tokens = tokenizer.encode(prompt_text);
    if (prompt_tokens.empty()) {
        return pulse::Err<i32>(pulse::ErrorCode::InvalidArgument, "Failed to encode prompt");
    }

    for (i32 token_id : prompt_tokens) {
        auto prompt_result = forward_append_token(model, state, token_id);
        if (!prompt_result) {
            return pulse::Err<i32>(std::move(prompt_result.error()));
        }
    }

    i32 visible_tokens = 0;
    std::vector<i32> generated_tokens;
    std::size_t printed_bytes = 0;
    constexpr f32 kPresencePenalty = 1.5f;
    while (visible_tokens < max_new_tokens) {
        std::vector<std::pair<i32, f32>> top_candidates;
        const auto next_token = sample_token(state.logits,
                                             temperature,
                                             top_k,
                                             top_p,
                                             banned_tokens,
                                             rng,
                                             use_greedy,
                                             &state.all_tokens,
                                             kPresencePenalty,
                                             debug_tokens ? &top_candidates : nullptr);
        if (next_token < 0) {
            return pulse::Err<i32>(pulse::ErrorCode::InvalidArgument, "Failed to sample next token");
        }

        auto next_result = forward_append_token(model, state, next_token);
        if (!next_result) {
            return pulse::Err<i32>(std::move(next_result.error()));
        }

        if (is_stop_token(next_token, config, tokenizer)) {
            break;
        }

        if (print_output) {
            generated_tokens.push_back(next_token);
            flush_streaming_answer(tokenizer, generated_tokens, printed_bytes);
        }

        if (debug_tokens) {
            std::cerr << std::format("\n[debug] sampled token {} text='{}'\n",
                                     next_token,
                                     tokenizer.decode({next_token}));
            for (const auto& [token_id, logit] : top_candidates) {
                std::cerr << std::format("  top token {} logit {:.6f} text='{}'\n",
                                         token_id,
                                         logit,
                                         tokenizer.decode({token_id}));
            }
        }
        visible_tokens++;
    }

    if (print_output) {
        flush_streaming_answer(tokenizer, generated_tokens, printed_bytes);
        std::cout << "\n";
    }

    return pulse::Ok(visible_tokens);
}

std::vector<i32> build_banned_tokens(const Qwen3Tokenizer& tokenizer, const Qwen3Config& config) {
    std::vector<i32> banned_tokens;
    for (i32 token_id = tokenizer.vocab_size(); token_id < config.vocab_size; ++token_id) {
        banned_tokens.push_back(token_id);
    }

    if (tokenizer.pad_id() >= 0) {
        banned_tokens.push_back(tokenizer.pad_id());
    }

    std::sort(banned_tokens.begin(), banned_tokens.end());
    banned_tokens.erase(std::unique(banned_tokens.begin(), banned_tokens.end()), banned_tokens.end());
    return banned_tokens;
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    constexpr i32 kMaxNewTokens = 32;
    constexpr bool kDebugTokens = false;
    const std::vector<std::string_view> questions = {
        "What is 2 + 2? Reply briefly.",
        "Name one programming language.",
        "What color is the sky on a clear day? Reply briefly.",
        "What is the capital of France? Reply briefly.",
    };

    const fs::path model_path = argc > 1 ? fs::path(argv[1]) : find_default_model_path();

    int device_count = 0;
    const auto cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count <= 0) {
        std::cerr << std::format("CUDA is not available: {}\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    std::cout << "PulseInfer - Qwen3 Inference Demo\n";
    std::cout << "=================================\n";
    std::cout << "Device: CUDA\n";
    std::cout << std::format("Model path: {}\n", model_path.string());
    std::cout << std::format("CUDA devices: {}\n\n", device_count);

    std::cout << "Loading tokenizer...\n";
    auto tokenizer_result = Qwen3Tokenizer::load(model_path);
    if (!tokenizer_result) {
        std::cerr << std::format("Failed to load tokenizer: {}\n", tokenizer_result.error().message());
        return 1;
    }
    auto tokenizer = std::move(tokenizer_result.value());
    std::cout << std::format("  Vocab size: {}\n", tokenizer.vocab_size());
    std::cout << std::format("  EOS token : {}\n\n", tokenizer.eos_id());

    std::cout << "Loading config...\n";
    auto config_result = Qwen3Config::load(model_path);
    if (!config_result) {
        std::cerr << std::format("Failed to load config: {}\n", config_result.error().message());
        return 1;
    }
    auto config = std::move(config_result.value());

    if (config.torch_dtype != "bfloat16") {
        std::cerr << std::format("Only bfloat16 models are supported, got '{}'\n", config.torch_dtype);
        return 1;
    }

    constexpr auto model_dtype = DataType::BFloat16;
    const f32 runtime_temperature = config.temperature > 0.0f ? config.temperature : 0.6f;
    const i32 runtime_top_k = config.top_k > 0 ? config.top_k : 20;
    const f32 runtime_top_p = config.top_p > 0.0f ? config.top_p : 0.95f;
    std::cout << std::format("  Layers      : {}\n", config.num_hidden_layers);
    std::cout << std::format("  Hidden size : {}\n", config.hidden_size);
    std::cout << std::format("  DType       : {}\n", pulse::data_type_str(model_dtype));
    std::cout << std::format("  Temperature : {}\n", runtime_temperature);
    std::cout << std::format("  Top-K       : {}\n", runtime_top_k);
    std::cout << std::format("  Top-P       : {}\n\n", runtime_top_p);

    std::cout << "Loading weights to CUDA...\n";
    auto var_map_result = load_model_vars(model_path);
    if (!var_map_result) {
        std::cerr << std::format("Failed to load weights: {}\n", var_map_result.error().message());
        return 1;
    }

    auto var_map = std::move(var_map_result.value());
    const auto builder = VarBuilder::from_var_map(var_map);
    std::cout << std::format("  Tensor count: {}\n\n", var_map.size());

    std::cout << "Creating model...\n";
    Qwen3Model model(config, DeviceType::CUDA, model_dtype, builder);
    auto init_result = model.init();
    if (!init_result) {
        std::cerr << std::format("Failed to initialize model: {}\n", init_result.error().message());
        return 1;
    }

    auto logits_result = Tensor::create({1, config.vocab_size}, model_dtype, DeviceType::CUDA);
    if (!logits_result) {
        std::cerr << std::format("Failed to create logits buffer: {}\n", logits_result.error().message());
        return 1;
    }

    GenerationState state{
        .current_pos = 0,
        .all_tokens = {},
        .logits = std::move(logits_result.value()),
    };

    const auto banned_tokens = build_banned_tokens(tokenizer, config);

    std::random_device rd;
    std::mt19937 rng(rd());

    std::cout << "\nModel ready!\n\n";
    std::cout << "=== QA Demo ===\n";
    for (std::size_t index = 0; index < questions.size(); ++index) {
        model.reset_cache();
        state.current_pos = 0;
        state.all_tokens.clear();

        const auto question = questions[index];
        std::cout << std::format("[{}] Question: {}\n", index + 1, question);
        std::cout << "Answer: ";

        const auto prompt = make_chat_prompt(question);
        const auto begin = std::chrono::steady_clock::now();
        auto result = generate_turn(model,
                                    tokenizer,
                                    runtime_temperature,
                                    runtime_top_k,
                                    runtime_top_p,
                                    config,
                                    state,
                                    prompt,
                                    kMaxNewTokens,
                                    banned_tokens,
                                    rng,
                                    false,
                                    kDebugTokens,
                                    true);
        const auto end = std::chrono::steady_clock::now();
        if (!result) {
            std::cerr << std::format("QA demo failed on case {}: {}\n", index + 1, result.error().message());
            return 1;
        }

        const auto seconds = std::chrono::duration<double>(end - begin).count();
        std::cout << std::format("Generated {} visible tokens in {:.3f}s ({:.3f} tokens/s)\n\n",
                                 result.value(),
                                 seconds,
                                 seconds > 0.0 ? static_cast<double>(result.value()) / seconds : 0.0);
    }
    return 0;
}
