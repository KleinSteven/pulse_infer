#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>

#include "pulse/core/var.hpp"

int main() {
    namespace fs = std::filesystem;
    using pulse::VarMap;
    using pulse::VarBuilder;

    const fs::path model_dir = "../../model";
    if (!fs::exists(model_dir) || !fs::is_directory(model_dir)) {
        std::cerr << std::format("model directory not found: {}\n", model_dir.string());
        return 1;
    }

    std::vector<fs::path> safetensor_files;

    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const auto extension = entry.path().extension().string();
        if (extension == ".safetensor" || extension == ".safetensors") {
            safetensor_files.push_back(entry.path());
        }
    }

    std::ranges::sort(safetensor_files);

    if (safetensor_files.empty()) {
        std::cerr << std::format("no safetensor file found in {}\n", model_dir.string());
        return 1;
    }

    for (const auto& path : safetensor_files) {
        auto var_map_result = VarMap::from_safetensors(path, pulse::DeviceType::CPU);
        if (!var_map_result) {
            std::cerr << std::format("failed to load {}: {}\n",
                                     path.string(),
                                     var_map_result.error().message());
            return 1;
        }

        auto var_map = std::move(var_map_result.value());
        const auto builder = VarBuilder::from_var_map(var_map);

        std::cout << std::format("file: {}\n", path.string());
        std::cout << std::format("  file_size: {} bytes\n", fs::file_size(path));
        std::cout << std::format("  tensor_count: {}\n", var_map.size());

        std::cout << "  tensors:\n";
        for (const auto& [name, tensor] : var_map.tensors()) {
            std::cout << std::format("    {}: {}\n", name, tensor.to_string());
        }

        if (!var_map.tensors().empty()) {
            const auto& first_name = var_map.tensors().begin()->first;
            auto named_tensor_result = builder.get(first_name);
            if (!named_tensor_result) {
                std::cerr << std::format("  builder lookup failed for {}: {}\n",
                                         first_name,
                                         named_tensor_result.error().message());
                return 1;
            }

            std::cout << "  builder_lookup:\n";
            std::cout << std::format("    name: {}\n", first_name);
            std::cout << std::format("    tensor: {}\n", named_tensor_result.value()->to_string());
        }

        std::cout << '\n';
    }
}
