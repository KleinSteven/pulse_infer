#include "pulse/core/var.hpp"

#include <format>
#include <utility>

#include "pulse/core/safetensor.hpp"

namespace pulse {

namespace {

[[nodiscard]] Result<void> validate_name(std::string_view name) {
    if (name.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Tensor name cannot be empty");
    }

    if (name.front() == '.' || name.back() == '.') {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("Invalid tensor name '{}': leading or trailing '.'", name));
    }

    if (name.find("..") != std::string_view::npos) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("Invalid tensor name '{}': contains empty path segment", name));
    }

    return Ok();
}

[[nodiscard]] Result<void> validate_non_empty_name(std::string_view name, std::string_view api_name) {
    if (name.empty()) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("{} name cannot be empty", api_name));
    }

    return Ok();
}

[[nodiscard]] std::string join_name(std::string_view prefix, std::string_view name) {
    if (prefix.empty()) {
        return std::string(name);
    }

    if (name.empty()) {
        return std::string(prefix);
    }

    std::string full_name;
    full_name.reserve(prefix.size() + name.size() + 1);
    full_name.append(prefix);
    full_name.push_back('.');
    full_name.append(name);
    return full_name;
}

[[nodiscard]] Result<void> validate_tensor_spec(const Tensor& tensor,
                                                const std::vector<i32>& dims,
                                                DataType dtype,
                                                DeviceType device,
                                                std::string_view name) {
    if (tensor.dims() != dims) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("Tensor '{}' shape mismatch: expected rank {}, got {}",
                                     name,
                                     dims.size(),
                                     tensor.dims().size()));
    }

    if (tensor.dtype() != dtype) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("Tensor '{}' dtype mismatch: expected {}, got {}",
                                     name,
                                     data_type_str(dtype),
                                     data_type_str(tensor.dtype())));
    }

    if (tensor.device() != device) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("Tensor '{}' device mismatch: expected {}, got {}",
                                     name,
                                     device_type_str(device),
                                     device_type_str(tensor.device())));
    }

    return Ok();
}

}  // namespace

Result<VarMap> VarMap::from_safetensors(const std::filesystem::path& path, DeviceType device) {
    if (device != DeviceType::CPU
#ifdef PULSE_USE_CUDA
        && device != DeviceType::CUDA
#endif
    ) {
        return Err<VarMap>(ErrorCode::InvalidArgument,
                           std::format("Unsupported device for VarMap::from_safetensors: {}",
                                       device_type_str(device)));
    }

    auto loader_result = SafeTensorLoader::load(path);
    if (!loader_result) {
        return Err<VarMap>(std::move(loader_result.error()));
    }

    auto loader = std::move(loader_result.value());
    VarMap var_map;

    for (const auto& name : loader.get_tensor_names()) {
        auto tensor_result = loader.get_tensor(name, device);
        if (!tensor_result) {
            return Err<VarMap>(std::move(tensor_result.error()));
        }

        var_map.tensors_.emplace(name, std::move(tensor_result.value()));
    }

    return Ok(std::move(var_map));
}

usize VarMap::size() const noexcept {
    return tensors_.size();
}

bool VarMap::empty() const noexcept {
    return size() == 0;
}

const VarMap::TensorMap& VarMap::tensors() const noexcept {
    return tensors_;
}

bool VarMap::contains(std::string_view name) const noexcept {
    return find(name) != nullptr;
}

const Tensor* VarMap::find(std::string_view name) const noexcept {
    const auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return std::addressof(it->second);
    }

    return nullptr;
}

Tensor* VarMap::find(std::string_view name) noexcept {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }

    return std::addressof(it->second);
}

Result<const Tensor*> VarMap::get(std::string_view name) const {
    auto validate_result = validate_name(name);
    if (!validate_result) {
        return Err<const Tensor*>(std::move(validate_result.error()));
    }

    const auto* tensor = find(name);
    if (tensor == nullptr) {
        return Err<const Tensor*>(ErrorCode::InvalidArgument, std::format("Tensor '{}' not found", name));
    }

    return Ok(tensor);
}

Result<Tensor*> VarMap::get(std::string_view name) {
    auto validate_result = validate_name(name);
    if (!validate_result) {
        return Err<Tensor*>(std::move(validate_result.error()));
    }

    auto* tensor = find(name);
    if (tensor == nullptr) {
        return Err<Tensor*>(ErrorCode::InvalidArgument, std::format("Tensor '{}' not found", name));
    }

    return Ok(tensor);
}

Result<void> VarMap::insert(std::string name, Tensor tensor) {
    auto validate_result = validate_name(name);
    if (!validate_result) {
        return validate_result;
    }

    if (contains(name)) {
        return Err<void>(ErrorCode::InvalidArgument, std::format("Tensor '{}' already exists", name));
    }

    tensors_.emplace(std::move(name), std::move(tensor));
    return Ok();
}

Result<Tensor*>
VarMap::get_or_create(std::string name, std::vector<i32> dims, DataType dtype, DeviceType device) {
    auto validate_result = validate_name(name);
    if (!validate_result) {
        return Err<Tensor*>(std::move(validate_result.error()));
    }

    if (auto* existing = find(name); existing != nullptr) {
        auto spec_result = validate_tensor_spec(*existing, dims, dtype, device, name);
        if (!spec_result) {
            return Err<Tensor*>(std::move(spec_result.error()));
        }

        return Ok(existing);
    }

    auto tensor_result = Tensor::create(std::move(dims), dtype, device);
    if (!tensor_result) {
        return Err<Tensor*>(std::move(tensor_result.error()));
    }

    auto [it, inserted] = tensors_.try_emplace(std::move(name), std::move(tensor_result.value()));
    if (!inserted) {
        return Err<Tensor*>(ErrorCode::InvalidArgument, "Failed to insert created tensor");
    }

    return Ok(std::addressof(it->second));
}

VarBuilder VarBuilder::from_var_map(VarMap& var_map) noexcept {
    return VarBuilder(std::addressof(var_map), {});
}

Result<VarBuilder> VarBuilder::pp(std::string_view name) const {
    auto validate_result = validate_non_empty_name(name, "VarBuilder::pp");
    if (!validate_result) {
        return Err<VarBuilder>(std::move(validate_result.error()));
    }

    if (name.find('.') != std::string_view::npos) {
        return Err<VarBuilder>(ErrorCode::InvalidArgument,
                               std::format("VarBuilder::pp segment '{}' cannot contain '.'", name));
    }

    return Ok(VarBuilder(var_map_, join_name(prefix_, name)));
}

Result<const Tensor*> VarBuilder::get(std::string_view name) const {
    auto validate_input_result = validate_non_empty_name(name, "VarBuilder::get");
    if (!validate_input_result) {
        return Err<const Tensor*>(std::move(validate_input_result.error()));
    }

    const auto full_name = join_name(prefix_, name);

    auto validate_result = validate_name(full_name);
    if (!validate_result) {
        return Err<const Tensor*>(std::move(validate_result.error()));
    }

    if (var_map_ == nullptr) {
        return Err<const Tensor*>(ErrorCode::InvalidArgument, "VarBuilder has no backing storage");
    }

    return std::as_const(*var_map_).get(full_name);
}

const Tensor* VarBuilder::find(std::string_view name) const noexcept {
    if (var_map_ == nullptr || name.empty()) {
        return nullptr;
    }

    return std::as_const(*var_map_).find(join_name(prefix_, name));
}

Result<const Tensor*> VarBuilder::get(std::string_view name,
                                      const std::vector<i32>& dims,
                                      DataType dtype) const {
    auto validate_input_result = validate_non_empty_name(name, "VarBuilder::get");
    if (!validate_input_result) {
        return Err<const Tensor*>(std::move(validate_input_result.error()));
    }

    const auto full_name = join_name(prefix_, name);
    auto tensor_result = get(name);
    if (!tensor_result) {
        return tensor_result;
    }

    const auto* tensor = tensor_result.value();
    auto validate_result = validate_tensor_spec(*tensor, dims, dtype, tensor->device(), full_name);
    if (!validate_result) {
        return Err<const Tensor*>(std::move(validate_result.error()));
    }

    return tensor_result;
}

Result<Tensor*> VarBuilder::get_or_create(std::string_view name,
                                          std::vector<i32> dims,
                                          DataType dtype,
                                          DeviceType device) const {
    auto validate_input_result = validate_non_empty_name(name, "VarBuilder::get_or_create");
    if (!validate_input_result) {
        return Err<Tensor*>(std::move(validate_input_result.error()));
    }

    if (var_map_ == nullptr) {
        return Err<Tensor*>(ErrorCode::InvalidArgument, "VarBuilder has no backing storage");
    }

    return var_map_->get_or_create(join_name(prefix_, name), std::move(dims), dtype, device);
}

}  // namespace pulse
