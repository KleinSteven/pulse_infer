#include "pulse/model/qwen3/config.hpp"

#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>

namespace pulse::model {

namespace {

using json = nlohmann::json;

[[nodiscard]] Result<std::string> read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return Err<std::string>(ErrorCode::OpenFileError, std::format("Failed to open '{}'", path.string()));
    }

    file.seekg(0, std::ios::end);
    const auto file_size_raw = file.tellg();
    if (file_size_raw < 0) {
        return Err<std::string>(ErrorCode::GetFileSizeError,
                                std::format("Failed to get size of '{}'", path.string()));
    }

    const auto file_size = static_cast<usize>(file_size_raw);
    file.seekg(0, std::ios::beg);

    std::string content(file_size, '\0');
    if (file_size > 0) {
        file.read(content.data(), static_cast<std::streamsize>(file_size));
    }

    return Ok(std::move(content));
}

[[nodiscard]] std::filesystem::path resolve_model_dir(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path;
    }

    return path.parent_path();
}

[[nodiscard]] Result<json> parse_json_file(const std::filesystem::path& path) {
    auto file_result = read_text_file(path);
    if (!file_result) {
        return Err<json>(std::move(file_result.error()));
    }

    auto parsed = json::parse(file_result.value(), nullptr, false, true);
    if (parsed.is_discarded() || !parsed.is_object()) {
        return Err<json>(ErrorCode::InvalidArgument,
                         std::format("Failed to parse json file '{}'", path.string()));
    }

    return Ok(std::move(parsed));
}

[[nodiscard]] const json* find_member(const json& object, std::string_view key) {
    auto it = object.find(std::string(key));
    if (it == object.end()) {
        return nullptr;
    }

    return &(*it);
}

[[nodiscard]] Result<std::string> read_required_string(const json& object,
                                                       std::string_view key,
                                                       const std::filesystem::path& path) {
    const auto* value = find_member(object, key);
    if (value == nullptr || !value->is_string()) {
        return Err<std::string>(ErrorCode::InvalidArgument,
                                std::format("Missing or invalid string key '{}' in '{}'",
                                            key,
                                            path.string()));
    }

    return Ok(value->get<std::string>());
}

[[nodiscard]] Result<bool> read_required_bool(const json& object,
                                              std::string_view key,
                                              const std::filesystem::path& path) {
    const auto* value = find_member(object, key);
    if (value == nullptr || !value->is_boolean()) {
        return Err<bool>(ErrorCode::InvalidArgument,
                         std::format("Missing or invalid bool key '{}' in '{}'",
                                     key,
                                     path.string()));
    }

    return Ok(value->get<bool>());
}

[[nodiscard]] Result<i32> read_required_i32(const json& object,
                                            std::string_view key,
                                            const std::filesystem::path& path) {
    const auto* value = find_member(object, key);
    if (value == nullptr || !value->is_number_integer()) {
        return Err<i32>(ErrorCode::InvalidArgument,
                        std::format("Missing or invalid int key '{}' in '{}'",
                                    key,
                                    path.string()));
    }

    return Ok(value->get<i32>());
}

[[nodiscard]] Result<f32> read_required_f32(const json& object,
                                            std::string_view key,
                                            const std::filesystem::path& path) {
    const auto* value = find_member(object, key);
    if (value == nullptr || !value->is_number()) {
        return Err<f32>(ErrorCode::InvalidArgument,
                        std::format("Missing or invalid float key '{}' in '{}'",
                                    key,
                                    path.string()));
    }

    return Ok(value->get<f32>());
}

[[nodiscard]] Result<f64> read_required_f64(const json& object,
                                            std::string_view key,
                                            const std::filesystem::path& path) {
    const auto* value = find_member(object, key);
    if (value == nullptr || !value->is_number()) {
        return Err<f64>(ErrorCode::InvalidArgument,
                        std::format("Missing or invalid float key '{}' in '{}'",
                                    key,
                                    path.string()));
    }

    return Ok(value->get<f64>());
}

[[nodiscard]] Result<i32> read_optional_i32(const json& object,
                                            std::string_view key,
                                            i32 default_value,
                                            const std::filesystem::path& path) {
    const auto* value = find_member(object, key);
    if (value == nullptr || value->is_null()) {
        return Ok(default_value);
    }

    if (!value->is_number_integer()) {
        return Err<i32>(ErrorCode::InvalidArgument,
                        std::format("Invalid int key '{}' in '{}'", key, path.string()));
    }

    return Ok(value->get<i32>());
}

[[nodiscard]] Result<f32> read_optional_f32(const json& object,
                                            std::string_view key,
                                            f32 default_value,
                                            const std::filesystem::path& path) {
    const auto* value = find_member(object, key);
    if (value == nullptr || value->is_null()) {
        return Ok(default_value);
    }

    if (!value->is_number()) {
        return Err<f32>(ErrorCode::InvalidArgument,
                        std::format("Invalid float key '{}' in '{}'", key, path.string()));
    }

    return Ok(value->get<f32>());
}

[[nodiscard]] Result<std::vector<i32>> read_optional_i32_list(const json& object,
                                                              std::string_view key,
                                                              const std::filesystem::path& path) {
    const auto* value = find_member(object, key);
    if (value == nullptr || value->is_null()) {
        return Ok(std::vector<i32>{});
    }

    std::vector<i32> values;
    if (value->is_number_integer()) {
        values.push_back(value->get<i32>());
        return Ok(std::move(values));
    }

    if (!value->is_array()) {
        return Err<std::vector<i32>>(ErrorCode::InvalidArgument,
                                     std::format("Invalid int list key '{}' in '{}'",
                                                 key,
                                                 path.string()));
    }

    values.reserve(value->size());
    for (const auto& item : *value) {
        if (!item.is_number_integer()) {
            return Err<std::vector<i32>>(ErrorCode::InvalidArgument,
                                         std::format("Invalid int list key '{}' in '{}'",
                                                     key,
                                                     path.string()));
        }
        values.push_back(item.get<i32>());
    }

    return Ok(std::move(values));
}

[[nodiscard]] Result<void> validate_config(const Qwen3Config& config, const std::filesystem::path& path) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.num_hidden_layers <= 0 ||
        config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0 ||
        config.max_position_embeddings <= 0 || config.vocab_size <= 0) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("Invalid model dimensions in '{}'", path.string()));
    }

    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        return Err<void>(
            ErrorCode::InvalidArgument,
            std::format("num_attention_heads({}) is not divisible by num_key_value_heads({}) in '{}'",
                        config.num_attention_heads,
                        config.num_key_value_heads,
                        path.string()));
    }

    return Ok();
}

}  // namespace

Result<Qwen3Config> Qwen3Config::load(const std::filesystem::path& path) {
    const auto model_dir = resolve_model_dir(path);
    const auto config_path = model_dir / "config.json";
    const auto generation_config_path = model_dir / "generation_config.json";

    if (!std::filesystem::exists(config_path)) {
        return Err<Qwen3Config>(ErrorCode::OpenFileError,
                                std::format("Config file '{}' does not exist", config_path.string()));
    }

    auto config_json_result = parse_json_file(config_path);
    if (!config_json_result) {
        return Err<Qwen3Config>(std::move(config_json_result.error()));
    }

    const auto& config_json = config_json_result.value();

    Qwen3Config config;

    auto torch_dtype_result = read_required_string(config_json, "torch_dtype", config_path);
    if (!torch_dtype_result) {
        return Err<Qwen3Config>(std::move(torch_dtype_result.error()));
    }
    config.torch_dtype = std::move(torch_dtype_result.value());

    auto eos_token_ids_result = read_optional_i32_list(config_json, "eos_token_id", config_path);
    if (!eos_token_ids_result) {
        return Err<Qwen3Config>(std::move(eos_token_ids_result.error()));
    }
    config.eos_token_ids = std::move(eos_token_ids_result.value());

    auto head_dim_result = read_required_i32(config_json, "head_dim", config_path);
    if (!head_dim_result) {
        return Err<Qwen3Config>(std::move(head_dim_result.error()));
    }
    config.head_dim = head_dim_result.value();

    auto hidden_size_result = read_required_i32(config_json, "hidden_size", config_path);
    if (!hidden_size_result) {
        return Err<Qwen3Config>(std::move(hidden_size_result.error()));
    }
    config.hidden_size = hidden_size_result.value();

    auto intermediate_size_result = read_required_i32(config_json, "intermediate_size", config_path);
    if (!intermediate_size_result) {
        return Err<Qwen3Config>(std::move(intermediate_size_result.error()));
    }
    config.intermediate_size = intermediate_size_result.value();

    auto max_position_embeddings_result =
        read_required_i32(config_json, "max_position_embeddings", config_path);
    if (!max_position_embeddings_result) {
        return Err<Qwen3Config>(std::move(max_position_embeddings_result.error()));
    }
    config.max_position_embeddings = max_position_embeddings_result.value();

    auto num_attention_heads_result = read_required_i32(config_json, "num_attention_heads", config_path);
    if (!num_attention_heads_result) {
        return Err<Qwen3Config>(std::move(num_attention_heads_result.error()));
    }
    config.num_attention_heads = num_attention_heads_result.value();

    auto num_hidden_layers_result = read_required_i32(config_json, "num_hidden_layers", config_path);
    if (!num_hidden_layers_result) {
        return Err<Qwen3Config>(std::move(num_hidden_layers_result.error()));
    }
    config.num_hidden_layers = num_hidden_layers_result.value();

    auto num_key_value_heads_result =
        read_required_i32(config_json, "num_key_value_heads", config_path);
    if (!num_key_value_heads_result) {
        return Err<Qwen3Config>(std::move(num_key_value_heads_result.error()));
    }
    config.num_key_value_heads = num_key_value_heads_result.value();

    auto rms_norm_eps_result = read_required_f32(config_json, "rms_norm_eps", config_path);
    if (!rms_norm_eps_result) {
        return Err<Qwen3Config>(std::move(rms_norm_eps_result.error()));
    }
    config.rms_norm_eps = rms_norm_eps_result.value();

    auto rope_theta_result = read_required_f64(config_json, "rope_theta", config_path);
    if (!rope_theta_result) {
        return Err<Qwen3Config>(std::move(rope_theta_result.error()));
    }
    config.rope_theta = rope_theta_result.value();

    auto tie_word_embeddings_result =
        read_required_bool(config_json, "tie_word_embeddings", config_path);
    if (!tie_word_embeddings_result) {
        return Err<Qwen3Config>(std::move(tie_word_embeddings_result.error()));
    }
    config.tie_word_embeddings = tie_word_embeddings_result.value();

    auto vocab_size_result = read_required_i32(config_json, "vocab_size", config_path);
    if (!vocab_size_result) {
        return Err<Qwen3Config>(std::move(vocab_size_result.error()));
    }
    config.vocab_size = vocab_size_result.value();

    if (std::filesystem::exists(generation_config_path)) {
        auto generation_json_result = parse_json_file(generation_config_path);
        if (!generation_json_result) {
            return Err<Qwen3Config>(std::move(generation_json_result.error()));
        }

        const auto& generation_json = generation_json_result.value();

        auto generation_eos_token_ids_result =
            read_optional_i32_list(generation_json, "eos_token_id", generation_config_path);
        if (!generation_eos_token_ids_result) {
            return Err<Qwen3Config>(std::move(generation_eos_token_ids_result.error()));
        }
        if (!generation_eos_token_ids_result.value().empty()) {
            config.eos_token_ids = std::move(generation_eos_token_ids_result.value());
        }

        auto temperature_result =
            read_optional_f32(generation_json, "temperature", config.temperature, generation_config_path);
        if (!temperature_result) {
            return Err<Qwen3Config>(std::move(temperature_result.error()));
        }
        config.temperature = temperature_result.value();

        auto top_k_result = read_optional_i32(generation_json, "top_k", config.top_k, generation_config_path);
        if (!top_k_result) {
            return Err<Qwen3Config>(std::move(top_k_result.error()));
        }
        config.top_k = top_k_result.value();

        auto top_p_result = read_optional_f32(generation_json, "top_p", config.top_p, generation_config_path);
        if (!top_p_result) {
            return Err<Qwen3Config>(std::move(top_p_result.error()));
        }
        config.top_p = top_p_result.value();
    }

    if (config.eos_token_ids.empty()) {
        return Err<Qwen3Config>(ErrorCode::InvalidArgument,
                                std::format("Missing or invalid eos_token_id in '{}'",
                                            config_path.string()));
    }

    auto validate_result = validate_config(config, config_path);
    if (!validate_result) {
        return Err<Qwen3Config>(std::move(validate_result.error()));
    }

    return Ok(std::move(config));
}

}  // namespace pulse::model
