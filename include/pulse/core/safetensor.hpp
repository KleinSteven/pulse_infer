#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pulse/core/allocator.hpp"
#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/types.hpp"

namespace pulse {

struct TensorMetadata {
    std::string name;
    std::string dtype;
    DataType data_type = DataType::Float32;
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
    ~SafeTensorLoader() = default;

    SafeTensorLoader(const SafeTensorLoader&) = delete;
    SafeTensorLoader& operator=(const SafeTensorLoader&) = delete;

    SafeTensorLoader(SafeTensorLoader&&) noexcept = default;
    SafeTensorLoader& operator=(SafeTensorLoader&&) noexcept = default;

    [[nodiscard]] static Result<SafeTensorLoader> load(const std::filesystem::path& path);

    [[nodiscard]] Result<Tensor> get_tensor(std::string_view name, DeviceType device) const;

    [[nodiscard]] std::vector<std::string> get_tensor_names() const;

private:
    std::filesystem::path path_;
    u64 file_size_ = 0;
    u64 header_size_ = 0;
    u64 payload_offset_ = 0;
    std::map<std::string, TensorMetadata, std::less<>> tensor_metadata_;
    std::unique_ptr<MmapAllocator> mmap_allocator_;
};

}  // namespace pulse
