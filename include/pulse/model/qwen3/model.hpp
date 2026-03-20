#pragma once

#include <memory>
#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/var.hpp"
#include "pulse/model/qwen3/config.hpp"

namespace pulse::layer {
class Embedding;
class Linear;
class RMSNorm;
}  // namespace pulse::layer

namespace pulse::model {

class Qwen3Model {
public:
    Qwen3Model(const Qwen3Config& config,
               DeviceType device,
               DataType dtype,
               const VarBuilder& builder) noexcept;

    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;

    Qwen3Model(Qwen3Model&&) noexcept;
    Qwen3Model& operator=(Qwen3Model&&) noexcept;

    ~Qwen3Model();

    [[nodiscard]] Result<void> init();
    [[nodiscard]] Result<void> forward(i32 token, i32 position, Tensor& logits);

    void reset_cache();

    [[nodiscard]] const Qwen3Config& config() const noexcept {
        return config_;
    }

    [[nodiscard]] DeviceType device() const noexcept {
        return device_;
    }

    [[nodiscard]] DataType dtype() const noexcept {
        return dtype_;
    }

private:
    class DecoderLayer;

    Qwen3Config config_;
    VarBuilder builder_;
    DeviceType device_ = DeviceType::CPU;
    DataType dtype_ = DataType::Float32;

    std::unique_ptr<pulse::layer::Embedding> embedding_;
    std::unique_ptr<pulse::layer::RMSNorm> final_norm_;
    std::unique_ptr<pulse::layer::Linear> lm_head_linear_;
    std::vector<std::unique_ptr<DecoderLayer>> layers_;
    std::vector<Tensor> key_cache_;
    std::vector<Tensor> value_cache_;
    Tensor final_norm_buffer_;

    bool initialized_ = false;
};

}  // namespace pulse::model
