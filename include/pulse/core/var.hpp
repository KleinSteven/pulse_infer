#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"

namespace pulse {

class VarMap {
public:
    using TensorMap = std::map<std::string, Tensor, std::less<>>;

    VarMap() = default;

    VarMap(const VarMap&) = delete;
    VarMap& operator=(const VarMap&) = delete;

    VarMap(VarMap&&) = default;
    VarMap& operator=(VarMap&&) = default;

    [[nodiscard]] static Result<VarMap> from_safetensors(const std::filesystem::path& path,
                                                         DeviceType device = DeviceType::CPU);

    [[nodiscard]] usize size() const noexcept;

    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] const TensorMap& tensors() const noexcept;

    [[nodiscard]] bool contains(std::string_view name) const noexcept;

    [[nodiscard]] const Tensor* find(std::string_view name) const noexcept;

    [[nodiscard]] Tensor* find(std::string_view name) noexcept;

    [[nodiscard]] Result<const Tensor*> get(std::string_view name) const;

    [[nodiscard]] Result<Tensor*> get(std::string_view name);

    [[nodiscard]] Result<void> insert(std::string name, Tensor tensor);

    [[nodiscard]] Result<Tensor*> get_or_create(std::string name,
                                                std::vector<i32> dims,
                                                DataType dtype,
                                                DeviceType device = DeviceType::CPU);

private:
    TensorMap tensors_;
};

class VarBuilder {
public:
    VarBuilder() = delete;

    [[nodiscard]] static VarBuilder from_var_map(VarMap& var_map) noexcept;

    [[nodiscard]] Result<VarBuilder> pp(std::string_view name) const;

    [[nodiscard]] const std::string& prefix() const noexcept {
        return prefix_;
    }

    [[nodiscard]] Result<const Tensor*> get(std::string_view name) const;

    [[nodiscard]] Result<const Tensor*> get(std::string_view name,
                                            const std::vector<i32>& dims,
                                            DataType dtype) const;

    [[nodiscard]] Result<Tensor*> get_or_create(std::string_view name,
                                                std::vector<i32> dims,
                                                DataType dtype,
                                                DeviceType device) const;

private:
    explicit VarBuilder(VarMap* var_map, std::string prefix) noexcept
        : var_map_(var_map), prefix_(std::move(prefix)) {}

    VarMap* var_map_ = nullptr;
    std::string prefix_;
};

}  // namespace pulse
