#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "pulse/core/safetensor.hpp"

namespace {

std::string shape_to_string(std::span<const pulse::i64> shape) {
    std::string result = "[";

    for (pulse::usize index = 0; index < shape.size(); ++index) {
        result += std::to_string(shape[index]);
        if (index + 1 < shape.size()) {
            result += ", ";
        }
    }

    result += "]";
    return result;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    using pulse::SafeTensorLoader;

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
        auto load_result = SafeTensorLoader::load(path, pulse::DeviceType::CPU);
        if (!load_result) {
            std::cerr << std::format("failed to load {}: {}\n", path.string(), load_result.error().message());
            return 1;
        }

        const auto loader = std::move(load_result.value());

        std::cout << std::format("file: {}\n", loader.path().string());
        std::cout << std::format("  file_size: {} bytes\n", loader.file_size());
        std::cout << std::format("  header_size: {} bytes\n", loader.header_size());
        std::cout << std::format("  payload_offset: {}\n", loader.payload_offset());
        std::cout << std::format("  tensor_count: {}\n", loader.tensor_count());

        if (!loader.metadata().empty()) {
            std::cout << "  metadata:\n";
            for (const auto& [key, value] : loader.metadata()) {
                std::cout << std::format("    {}: {}\n", key, value);
            }
        }

        std::cout << "  tensors:\n";
        for (const auto& [name, tensor] : loader.tensors()) {
            const auto* tensor_metadata = loader.find_tensor_metadata(name);
            if (tensor_metadata == nullptr) {
                std::cout << std::format("    {}: missing metadata\n", name);
                continue;
            }

            std::cout << std::format("    {}\n", name);
            std::cout << std::format("      dtype: {}\n", tensor_metadata->dtype);
            std::cout << std::format("      shape: {}\n", shape_to_string(tensor_metadata->shape));
            std::cout << std::format("      num_elements: {}\n", tensor_metadata->num_elements);
            std::cout << std::format("      byte_size: {}\n", tensor.byte_size());
            std::cout << std::format("      data_offsets: [{}, {}]\n",
                                     tensor_metadata->data_begin,
                                     tensor_metadata->data_end);
        }

        std::cout << "  tensor_load_preview:\n";
        pulse::usize preview_count = 0;
        for (const auto& [name, tensor] : loader.tensors()) {
            std::cout << std::format("    {}: {}, data={}\n",
                                     name,
                                     tensor.to_string(),
                                     static_cast<const void*>(tensor.data()));
            ++preview_count;
            if (preview_count == 3) {
                break;
            }
        }

        std::cout << '\n';
    }
}
