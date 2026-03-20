#pragma once

#include "pulse/core/error.hpp"
#include "pulse/core/tensor.hpp"
#include "pulse/core/var.hpp"

namespace pulse::layer {

class Embedding {
public:
    [[nodiscard]] static Result<Embedding> embedding(i32 num_embeddings,
                                                     i32 embedding_dim,
                                                     DeviceType device,
                                                     DataType dtype,
                                                     const VarBuilder& builder);

    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;

    Embedding(Embedding&&) = default;
    Embedding& operator=(Embedding&&) = default;

    [[nodiscard]] Result<Tensor> forward(const Tensor& input) const;

    [[nodiscard]] const Tensor* weight() const noexcept {
        return weight_;
    }

    [[nodiscard]] i32 num_embeddings() const noexcept {
        return num_embeddings_;
    }

    [[nodiscard]] i32 embedding_dim() const noexcept {
        return embedding_dim_;
    }

private:
    Embedding(i32 num_embeddings, i32 embedding_dim, DeviceType device, DataType dtype) noexcept
        : num_embeddings_(num_embeddings),
          embedding_dim_(embedding_dim),
          device_(device),
          dtype_(dtype) {}

    [[nodiscard]] Result<void> init(const VarBuilder& builder);

    const Tensor* weight_ = nullptr;
    i32 num_embeddings_ = 0;
    i32 embedding_dim_ = 0;
    DeviceType device_ = DeviceType::CPU;
    DataType dtype_ = DataType::Float32;
};

}  // namespace pulse::layer
