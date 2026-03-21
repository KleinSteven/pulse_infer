#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "pulse/core/var.hpp"
#include "pulse/model/qwen3/config.hpp"
#include "pulse/model/qwen3/model.hpp"
#include "pulse/model/qwen3/tokenizer.hpp"
#include "pulse/scheduler/continuous_batch_engine.hpp"

namespace {

using pulse::DataType;
using pulse::DeviceType;
using pulse::VarBuilder;
using pulse::VarMap;
using pulse::i32;
using pulse::model::Qwen3Config;
using pulse::model::Qwen3Model;
using pulse::model::Qwen3Tokenizer;
using pulse::scheduler::ContinuousBatchEngine;

struct DemoRequest {
    std::string prompt;
    i32 max_new_tokens = 0;
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

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path model_path = argc > 1 ? argv[1] : find_default_model_path();

    auto config_result = Qwen3Config::load(model_path);
    if (!config_result) {
        std::cerr << std::format("Failed to load config: {}\n", config_result.error().message());
        return 1;
    }
    auto config = std::move(config_result.value());

    auto tokenizer_result = Qwen3Tokenizer::load(model_path);
    if (!tokenizer_result) {
        std::cerr << std::format("Failed to load tokenizer: {}\n", tokenizer_result.error().message());
        return 1;
    }
    auto tokenizer = std::move(tokenizer_result.value());

    auto var_map_result = load_model_vars(model_path);
    if (!var_map_result) {
        std::cerr << std::format("Failed to load weights: {}\n", var_map_result.error().message());
        return 1;
    }
    auto var_map = std::move(var_map_result.value());
    const auto builder = VarBuilder::from_var_map(var_map);

    Qwen3Model prototype(config, DeviceType::CUDA, DataType::BFloat16, builder);
    auto init_result = prototype.init();
    if (!init_result) {
        std::cerr << std::format("Failed to initialize model: {}\n", init_result.error().message());
        return 1;
    }

    const std::vector<DemoRequest> requests = {
        {
            .prompt = "请用简短语言解释 continuous batching 在 LLM 推理中的作用。",
            .max_new_tokens = 96,
        },
        {
            .prompt = "用三句话说明 Transformer attention 的核心思想。",
            .max_new_tokens = 72,
        },
        {
            .prompt = "写一个 C++20 版本的 Fibonacci 函数，并说明时间复杂度。",
            .max_new_tokens = 96,
        },
    };

    i32 estimated_total_tokens = 0;
    for (const auto& request : requests) {
        estimated_total_tokens += static_cast<i32>(tokenizer.encode(request.prompt).size()) + request.max_new_tokens + 16;
    }

    auto paged_cache_result = prototype.init_paged_cache(std::max(64, (estimated_total_tokens + 15) / 16), 16);
    if (!paged_cache_result) {
        std::cerr << std::format("Failed to initialize paged cache: {}\n", paged_cache_result.error().message());
        return 1;
    }

    ContinuousBatchEngine engine(prototype, tokenizer, 2, 64);

    std::vector<pulse::i64> request_ids;
    request_ids.reserve(requests.size());
    for (const auto& request : requests) {
        request_ids.push_back(engine.add_request(request.prompt, request.max_new_tokens));
    }

    auto run_result = engine.run_until_complete(true);
    if (!run_result) {
        std::cerr << std::format("Continuous batch run failed: {}\n", run_result.error().message());
        return 1;
    }

    const auto stats = engine.get_stats();
    std::cout << std::format("Finished requests: {}/{}\n\n", stats.num_finished, stats.total_requests);
    for (std::size_t index = 0; index < requests.size(); ++index) {
        std::cout << std::format("[Request {}]\n", index);
        std::cout << std::format("Prompt: {}\n", requests[index].prompt);
        std::cout << std::format("Output: {}\n\n", engine.get_result(request_ids[index]));
    }

    return 0;
}
