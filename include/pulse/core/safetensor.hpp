#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/types.hpp"

namespace pulse {

struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::vector<i64> shape;
    u64 data_begin = 0;
    u64 data_end = 0;
    u64 item_size = 0;
    u64 num_elements = 0;

    [[nodiscard]] u64 byte_size() const noexcept {
        return data_end - data_begin;
    }
};

class SafeTensorLoader {
public:
    SafeTensorLoader() = default;

    [[nodiscard]] static Result<SafeTensorLoader> load(const std::filesystem::path& path,
                                                       DeviceType device = DeviceType::CPU);

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

    [[nodiscard]] u64 file_size() const noexcept {
        return file_size_;
    }

    [[nodiscard]] u64 header_size() const noexcept {
        return header_size_;
    }

    [[nodiscard]] u64 payload_offset() const noexcept {
        return payload_offset_;
    }

    [[nodiscard]] u64 payload_size() const noexcept {
        return file_size_ - payload_offset_;
    }

    [[nodiscard]] usize tensor_count() const noexcept {
        return tensors_.size();
    }

    [[nodiscard]] const std::map<std::string, Tensor>& tensors() const noexcept {
        return tensors_;
    }

    [[nodiscard]] const std::map<std::string, std::string>& metadata() const noexcept {
        return metadata_;
    }

    [[nodiscard]] const std::map<std::string, TensorMetadata>& tensor_metadata() const noexcept {
        return tensor_metadata_;
    }

    [[nodiscard]] const Tensor* find_tensor(std::string_view name) const noexcept;

    [[nodiscard]] const TensorMetadata* find_tensor_metadata(std::string_view name) const noexcept;

private:
    std::filesystem::path path_;
    u64 file_size_ = 0;
    u64 header_size_ = 0;
    u64 payload_offset_ = 0;
    std::map<std::string, std::string> metadata_;
    std::map<std::string, TensorMetadata> tensor_metadata_;
    std::map<std::string, Tensor> tensors_;
};

}  // namespace pulse
