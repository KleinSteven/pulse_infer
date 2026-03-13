#include "pulse/core/safetensor.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

#include "pulse/core/allocator.hpp"

#ifdef PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace pulse {

namespace {

using json = nlohmann::json;

struct ParsedTensor {
    TensorMetadata metadata;
    DataType dtype = DataType::Float32;
};

enum class SafeTensorDType : u8 {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
    F8E5M2,
    F8E4M3FN,
    C64,
    C128,
};

[[nodiscard]] constexpr std::optional<SafeTensorDType>
parse_safetensor_dtype(std::string_view dtype) noexcept {
    if (dtype == "BOOL") {
        return SafeTensorDType::Bool;
    }
    if (dtype == "U8") {
        return SafeTensorDType::U8;
    }
    if (dtype == "U16") {
        return SafeTensorDType::U16;
    }
    if (dtype == "U32") {
        return SafeTensorDType::U32;
    }
    if (dtype == "U64") {
        return SafeTensorDType::U64;
    }
    if (dtype == "I8") {
        return SafeTensorDType::I8;
    }
    if (dtype == "I16") {
        return SafeTensorDType::I16;
    }
    if (dtype == "I32") {
        return SafeTensorDType::I32;
    }
    if (dtype == "I64") {
        return SafeTensorDType::I64;
    }
    if (dtype == "F16") {
        return SafeTensorDType::F16;
    }
    if (dtype == "BF16") {
        return SafeTensorDType::BF16;
    }
    if (dtype == "F32") {
        return SafeTensorDType::F32;
    }
    if (dtype == "F64") {
        return SafeTensorDType::F64;
    }
    if (dtype == "F8_E5M2") {
        return SafeTensorDType::F8E5M2;
    }
    if (dtype == "F8_E4M3FN") {
        return SafeTensorDType::F8E4M3FN;
    }
    if (dtype == "C64") {
        return SafeTensorDType::C64;
    }
    if (dtype == "C128") {
        return SafeTensorDType::C128;
    }

    return std::nullopt;
}

[[nodiscard]] Result<u64> parse_u64(const json& value, std::string_view field_name) {
    if (value.is_number_unsigned()) {
        return Ok(value.get<u64>());
    }

    if (value.is_number_integer()) {
        const auto signed_value = value.get<i64>();
        if (signed_value < 0) {
            return Err<u64>(ErrorCode::InvalidArgument,
                            std::format("{} must be a non-negative integer", field_name));
        }

        return Ok(static_cast<u64>(signed_value));
    }

    return Err<u64>(ErrorCode::InvalidArgument, std::format("{} must be an integer", field_name));
}

[[nodiscard]] Result<std::vector<i64>> parse_shape(const json& value, std::string_view tensor_name) {
    if (!value.is_array()) {
        return Err<std::vector<i64>>(ErrorCode::InvalidArgument,
                                     std::format("Tensor '{}' shape must be an array", tensor_name));
    }

    std::vector<i64> shape;
    shape.reserve(value.size());

    for (const auto& dim_value : value) {
        if (!dim_value.is_number_integer()) {
            return Err<std::vector<i64>>(
                ErrorCode::InvalidArgument,
                std::format("Tensor '{}' shape contains a non-integer dimension", tensor_name));
        }

        const auto dim = dim_value.get<i64>();

        if (dim < 0) {
            return Err<std::vector<i64>>(
                ErrorCode::InvalidArgument,
                std::format("Tensor '{}' shape contains a negative dimension", tensor_name));
        }

        shape.push_back(dim);
    }

    return Ok(std::move(shape));
}

[[nodiscard]] Result<std::pair<u64, u64>> parse_data_offsets(const json& value,
                                                             std::string_view tensor_name) {
    if (!value.is_array() || value.size() != 2) {
        return Err<std::pair<u64, u64>>(
            ErrorCode::InvalidArgument,
            std::format("Tensor '{}' data_offsets must be an array of two integers", tensor_name));
    }

    auto begin_result = parse_u64(value[0], "data_offsets[0]");
    if (!begin_result) {
        return Err<std::pair<u64, u64>>(std::move(begin_result.error()));
    }

    auto end_result = parse_u64(value[1], "data_offsets[1]");
    if (!end_result) {
        return Err<std::pair<u64, u64>>(std::move(end_result.error()));
    }

    if (begin_result.value() > end_result.value()) {
        return Err<std::pair<u64, u64>>(
            ErrorCode::InvalidArgument,
            std::format("Tensor '{}' data_offsets are invalid: begin > end", tensor_name));
    }

    return Ok(std::pair<u64, u64>{begin_result.value(), end_result.value()});
}

[[nodiscard]] constexpr u64 dtype_size_bytes(std::string_view dtype) noexcept {
    const auto parsed_dtype = parse_safetensor_dtype(dtype);
    if (!parsed_dtype.has_value()) {
        return 0;
    }

    switch (*parsed_dtype) {
        case SafeTensorDType::Bool:
        case SafeTensorDType::U8:
        case SafeTensorDType::I8:
        case SafeTensorDType::F8E5M2:
        case SafeTensorDType::F8E4M3FN:
            return 1;
        case SafeTensorDType::U16:
        case SafeTensorDType::I16:
        case SafeTensorDType::F16:
        case SafeTensorDType::BF16:
            return 2;
        case SafeTensorDType::U32:
        case SafeTensorDType::I32:
        case SafeTensorDType::F32:
            return 4;
        case SafeTensorDType::U64:
        case SafeTensorDType::I64:
        case SafeTensorDType::F64:
        case SafeTensorDType::C64:
            return 8;
        case SafeTensorDType::C128:
            return 16;
    }

    return 0;
}

[[nodiscard]] Result<DataType> safetensor_dtype(std::string_view dtype) {
    const auto parsed_dtype = parse_safetensor_dtype(dtype);
    if (!parsed_dtype.has_value()) {
        return Err<DataType>(ErrorCode::InvalidArgument,
                             std::format("Unsupported safetensors dtype '{}'", dtype));
    }

    switch (*parsed_dtype) {
        case SafeTensorDType::Bool:
            return Ok(DataType::Bool);
        case SafeTensorDType::U8:
            return Ok(DataType::UInt8);
        case SafeTensorDType::U16:
            return Ok(DataType::UInt16);
        case SafeTensorDType::U32:
            return Ok(DataType::UInt32);
        case SafeTensorDType::U64:
            return Ok(DataType::UInt64);
        case SafeTensorDType::I8:
            return Ok(DataType::Int8);
        case SafeTensorDType::I16:
            return Ok(DataType::Int16);
        case SafeTensorDType::I32:
            return Ok(DataType::Int32);
        case SafeTensorDType::I64:
            return Ok(DataType::Int64);
        case SafeTensorDType::F16:
            return Ok(DataType::Float16);
        case SafeTensorDType::BF16:
            return Ok(DataType::BFloat16);
        case SafeTensorDType::F32:
            return Ok(DataType::Float32);
        case SafeTensorDType::F64:
            return Ok(DataType::Float64);
        case SafeTensorDType::F8E5M2:
        case SafeTensorDType::F8E4M3FN:
        case SafeTensorDType::C64:
        case SafeTensorDType::C128:
            break;
    }

    return Err<DataType>(ErrorCode::InvalidArgument,
                         std::format("Unsupported safetensors dtype '{}'", dtype));
}

[[nodiscard]] Result<u64> checked_element_count(const std::vector<i64>& shape, std::string_view tensor_name) {
    u64 count = 1;

    for (const auto dim : shape) {
        const auto u_dim = static_cast<u64>(dim);
        if (u_dim == 0) {
            return Ok<u64>(0);
        }

        if (count > std::numeric_limits<u64>::max() / u_dim) {
            return Err<u64>(ErrorCode::InvalidArgument,
                            std::format("Tensor '{}' element count overflow", tensor_name));
        }

        count *= u_dim;
    }

    return Ok(count);
}

[[nodiscard]] Result<ParsedTensor> parse_tensor_entry(const std::string& tensor_name,
                                                      const json& value,
                                                      u64 payload_size) {
    if (!value.is_object()) {
        return Err<ParsedTensor>(ErrorCode::InvalidArgument,
                                 std::format("Tensor '{}' metadata must be an object", tensor_name));
    }

    const auto dtype_it = value.find("dtype");
    const auto shape_it = value.find("shape");
    const auto offsets_it = value.find("data_offsets");

    if (dtype_it == value.end() || !dtype_it->is_string()) {
        return Err<ParsedTensor>(ErrorCode::InvalidArgument,
                                 std::format("Tensor '{}' is missing string field 'dtype'", tensor_name));
    }

    if (shape_it == value.end()) {
        return Err<ParsedTensor>(ErrorCode::InvalidArgument,
                                 std::format("Tensor '{}' is missing field 'shape'", tensor_name));
    }

    if (offsets_it == value.end()) {
        return Err<ParsedTensor>(ErrorCode::InvalidArgument,
                                 std::format("Tensor '{}' is missing field 'data_offsets'", tensor_name));
    }

    ParsedTensor parsed_tensor;
    parsed_tensor.metadata.name = tensor_name;
    parsed_tensor.metadata.dtype = dtype_it->get<std::string>();
    parsed_tensor.metadata.item_size = dtype_size_bytes(parsed_tensor.metadata.dtype);

    if (parsed_tensor.metadata.item_size == 0) {
        return Err<ParsedTensor>(ErrorCode::InvalidArgument,
                                 std::format("Tensor '{}' uses unsupported dtype '{}'",
                                             tensor_name,
                                             parsed_tensor.metadata.dtype));
    }

    auto dtype_result = safetensor_dtype(parsed_tensor.metadata.dtype);
    if (!dtype_result) {
        return Err<ParsedTensor>(std::move(dtype_result.error()));
    }
    parsed_tensor.dtype = dtype_result.value();

    auto shape_result = parse_shape(*shape_it, tensor_name);
    if (!shape_result) {
        return Err<ParsedTensor>(std::move(shape_result.error()));
    }
    parsed_tensor.metadata.shape = std::move(shape_result.value());

    auto offsets_result = parse_data_offsets(*offsets_it, tensor_name);
    if (!offsets_result) {
        return Err<ParsedTensor>(std::move(offsets_result.error()));
    }

    parsed_tensor.metadata.data_begin = offsets_result.value().first;
    parsed_tensor.metadata.data_end = offsets_result.value().second;

    if (parsed_tensor.metadata.data_end > payload_size) {
        return Err<ParsedTensor>(ErrorCode::InvalidArgument,
                                 std::format("Tensor '{}' data range [{} , {}) exceeds payload size {}",
                                             tensor_name,
                                             parsed_tensor.metadata.data_begin,
                                             parsed_tensor.metadata.data_end,
                                             payload_size));
    }

    auto count_result = checked_element_count(parsed_tensor.metadata.shape, tensor_name);
    if (!count_result) {
        return Err<ParsedTensor>(std::move(count_result.error()));
    }
    parsed_tensor.metadata.num_elements = count_result.value();

    const auto expected_bytes = parsed_tensor.metadata.num_elements * parsed_tensor.metadata.item_size;
    if (expected_bytes != parsed_tensor.metadata.byte_size()) {
        return Err<ParsedTensor>(ErrorCode::InvalidArgument,
                                 std::format("Tensor '{}' byte size mismatch: expected {}, got {}",
                                             tensor_name,
                                             expected_bytes,
                                             parsed_tensor.metadata.byte_size()));
    }

    return Ok(std::move(parsed_tensor));
}

[[nodiscard]] std::span<const u8>
tensor_bytes(const u8* file_data, u64 file_size, u64 payload_offset, const TensorMetadata& tensor) noexcept {
    if (file_data == nullptr || payload_offset + tensor.data_end > file_size) {
        return {};
    }

    return std::span<const u8>(file_data + payload_offset + tensor.data_begin,
                               static_cast<usize>(tensor.byte_size()));
}

[[nodiscard]] Result<std::vector<i32>> to_i32_dims(const TensorMetadata& tensor) {
    std::vector<i32> dims;
    dims.reserve(tensor.shape.size());

    for (const auto dim : tensor.shape) {
        if (dim > static_cast<i64>(std::numeric_limits<i32>::max())) {
            return Err<std::vector<i32>>(
                ErrorCode::InvalidArgument,
                std::format("Tensor '{}' dimension {} exceeds i32 range", tensor.name, dim));
        }

        dims.push_back(static_cast<i32>(dim));
    }

    return Ok(std::move(dims));
}

[[nodiscard]] Result<Tensor> build_tensor_from_bytes(const ParsedTensor& parsed_tensor,
                                                     std::span<const u8> bytes,
                                                     DeviceType device) {
    auto dims_result = to_i32_dims(parsed_tensor.metadata);
    if (!dims_result) {
        return Err<Tensor>(std::move(dims_result.error()));
    }

    auto tensor_result = Tensor::create(dims_result.value(), parsed_tensor.dtype, device);
    if (!tensor_result) {
        return tensor_result;
    }

    Tensor tensor(std::move(tensor_result.value()));

    if (device == DeviceType::CPU) {
        std::memcpy(tensor.data(), bytes.data(), bytes.size_bytes());
        return Ok(std::move(tensor));
    }

#ifdef PULSE_USE_CUDA
    if (device == DeviceType::CUDA) {
        cudaError_t err = cudaMemcpy(tensor.data(),
                                     bytes.data(),
                                     bytes.size_bytes(),
                                     cudaMemcpyKind::cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return Err<Tensor>(ErrorCode::CudaError,
                               std::format("cudaMemcpy failed: {}", cudaGetErrorString(err)));
        }

        return Ok(std::move(tensor));
    }
#endif

    return Err<Tensor>(ErrorCode::InvalidArgument, "NotSupport this device");
}

}  // namespace

Result<SafeTensorLoader> SafeTensorLoader::load(const std::filesystem::path& path, DeviceType device) {
    auto mmap_allocator = std::make_unique<MmapAllocator>(path.string());
    auto init_result = mmap_allocator->init();
    if (!init_result) {
        return Err<SafeTensorLoader>(std::move(init_result.error()));
    }

    const auto file_size = static_cast<u64>(mmap_allocator->size());
    if (file_size < 8) {
        return Err<SafeTensorLoader>(
            ErrorCode::InvalidArgument,
            std::format("'{}' is too small to be a safetensors file", path.string()));
    }

    const auto* file_data = static_cast<const u8*>(mmap_allocator->data());
    if (file_data == nullptr) {
        return Err<SafeTensorLoader>(ErrorCode::MmapError, std::format("Failed to mmap '{}'", path.string()));
    }

    u64 header_size = 0;
    std::memcpy(&header_size, file_data, sizeof(header_size));
    const auto payload_offset = header_size + 8;

    if (payload_offset < 8 || payload_offset > file_size) {
        return Err<SafeTensorLoader>(
            ErrorCode::InvalidArgument,
            std::format("'{}' has an invalid safetensors header size {}", path.string(), header_size));
    }

    const auto* header_begin = reinterpret_cast<const char*>(file_data + 8);
    const auto* header_end = header_begin + static_cast<isize>(header_size);
    json header_json = json::parse(header_begin, header_end, nullptr, false);

    if (header_json.is_discarded()) {
        return Err<SafeTensorLoader>(ErrorCode::InvalidArgument,
                                     std::format("Failed to parse safetensors header in '{}'",
                                                 path.string()));
    }

    if (!header_json.is_object()) {
        return Err<SafeTensorLoader>(
            ErrorCode::InvalidArgument,
            std::format("Safetensors header in '{}' must be a JSON object", path.string()));
    }

    SafeTensorLoader loader;
    loader.path_ = path;
    loader.file_size_ = file_size;
    loader.header_size_ = header_size;
    loader.payload_offset_ = payload_offset;

    const auto payload_size = file_size - payload_offset;
    std::vector<ParsedTensor> parsed_tensors;

    for (const auto& [key, value] : header_json.items()) {
        if (key == "__metadata__") {
            if (!value.is_object()) {
                return Err<SafeTensorLoader>(ErrorCode::InvalidArgument,
                                             "__metadata__ must be a JSON object");
            }

            for (const auto& [meta_key, meta_value] : value.items()) {
                if (!meta_value.is_string()) {
                    return Err<SafeTensorLoader>(ErrorCode::InvalidArgument,
                                                 std::format("Metadata '{}' must be a string", meta_key));
                }

                loader.metadata_.emplace(meta_key, meta_value.get<std::string>());
            }

            continue;
        }

        auto parsed_result = parse_tensor_entry(key, value, payload_size);
        if (!parsed_result) {
            return Err<SafeTensorLoader>(std::move(parsed_result.error()));
        }

        parsed_tensors.push_back(std::move(parsed_result.value()));
    }

    std::ranges::sort(parsed_tensors, [](const ParsedTensor& lhs, const ParsedTensor& rhs) {
        return lhs.metadata.data_begin < rhs.metadata.data_begin;
    });

    u64 previous_end = 0;
    for (const auto& parsed_tensor : parsed_tensors) {
        if (parsed_tensor.metadata.data_begin < previous_end) {
            return Err<SafeTensorLoader>(ErrorCode::InvalidArgument,
                                         std::format("Tensor '{}' overlaps a previous tensor data range",
                                                     parsed_tensor.metadata.name));
        }

        previous_end = parsed_tensor.metadata.data_end;

        const auto bytes = tensor_bytes(file_data, file_size, payload_offset, parsed_tensor.metadata);
        if (bytes.empty() && parsed_tensor.metadata.byte_size() != 0) {
            return Err<SafeTensorLoader>(
                ErrorCode::InvalidArgument,
                std::format("Tensor '{}' has an invalid byte view", parsed_tensor.metadata.name));
        }

        auto tensor_result = build_tensor_from_bytes(parsed_tensor, bytes, device);
        if (!tensor_result) {
            return Err<SafeTensorLoader>(std::move(tensor_result.error()));
        }

        loader.tensor_metadata_.emplace(parsed_tensor.metadata.name, parsed_tensor.metadata);
        loader.tensors_.emplace(parsed_tensor.metadata.name, std::move(tensor_result.value()));
    }

    return Ok(std::move(loader));
}

const Tensor* SafeTensorLoader::find_tensor(std::string_view name) const noexcept {
    const auto it = tensors_.find(std::string(name));
    if (it == tensors_.end()) {
        return nullptr;
    }

    return std::addressof(it->second);
}

const TensorMetadata* SafeTensorLoader::find_tensor_metadata(std::string_view name) const noexcept {
    const auto it = tensor_metadata_.find(std::string(name));
    if (it == tensor_metadata_.end()) {
        return nullptr;
    }

    return std::addressof(it->second);
}

}  // namespace pulse
