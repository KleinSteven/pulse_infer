#include "pulse/model/qwen3/model.hpp"

#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <utility>

#include "pulse/layer/embedding.hpp"
#include "pulse/layer/linear.hpp"
#include "pulse/layer/rmsnorm.hpp"
#include "pulse/layer/silu.hpp"
#include "pulse/ops/add.hpp"
#include "pulse/ops/matmul.hpp"
#include "pulse/ops/mha.hpp"
#include "pulse/ops/mul.hpp"
#include "pulse/ops/rope.hpp"

#ifdef PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace pulse::model {

namespace {

[[nodiscard]] Result<void> copy_tensor_bytes(const Tensor& src, Tensor& dst) {
    if (src.dims() != dst.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Tensor copy shape mismatch");
    }

    if (src.dtype() != dst.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Tensor copy dtype mismatch");
    }

    if (src.device() == DeviceType::CPU && dst.device() == DeviceType::CPU) {
        std::memcpy(dst.data(), src.data(), src.byte_size());
        return Ok();
    }

#ifdef PULSE_USE_CUDA
    cudaMemcpyKind kind;
    if (src.device() == DeviceType::CUDA && dst.device() == DeviceType::CUDA) {
        kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
    } else if (src.device() == DeviceType::CUDA && dst.device() == DeviceType::CPU) {
        kind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
    } else if (src.device() == DeviceType::CPU && dst.device() == DeviceType::CUDA) {
        kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
    } else {
        return Err<void>(ErrorCode::DeviceMismatch, "Tensor copy device mismatch");
    }

    const auto err = cudaMemcpy(dst.data(), src.data(), src.byte_size(), kind);
    if (err != cudaSuccess) {
        return Err<void>(ErrorCode::CudaError, std::format("cudaMemcpy failed: {}", cudaGetErrorString(err)));
    }

    return Ok();
#else
    return Err<void>(ErrorCode::DeviceMismatch, "Tensor copy device mismatch");
#endif
}

[[nodiscard]] Result<void> copy_cache_row(const Tensor& src, Tensor& cache, i32 row) {
    if (src.ndim() != 2 || src.dim(0) != 1 || cache.ndim() != 2 || src.dim(1) != cache.dim(1)) {
        return Err<void>(ErrorCode::ShapeMismatch, "KV cache row copy shape mismatch");
    }

    if (row < 0 || row >= cache.dim(0)) {
        return Err<void>(ErrorCode::InvalidArgument, "KV cache row out of bounds");
    }

    const usize row_bytes = static_cast<usize>(cache.dim(1)) * data_type_size(cache.dtype());
    auto* dst = static_cast<std::byte*>(cache.data()) + static_cast<usize>(row) * row_bytes;

    if (cache.device() == DeviceType::CPU) {
        std::memcpy(dst, src.data(), row_bytes);
        return Ok();
    }

#ifdef PULSE_USE_CUDA
    const auto err = cudaMemcpy(dst, src.data(), row_bytes, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return Err<void>(ErrorCode::CudaError, std::format("cudaMemcpy failed: {}", cudaGetErrorString(err)));
    }

    return Ok();
#else
    return Err<void>(ErrorCode::NotImplemented, "KV cache row copy not implemented for this device");
#endif
}

}  // namespace

class Qwen3Model::DecoderLayer {
public:
    struct Workspace {
        Tensor attn_input;
        Tensor q;
        Tensor k;
        Tensor v;
        Tensor q_norm;
        Tensor k_norm;
        Tensor q_rope;
        Tensor k_rope;
        Tensor attn_score;
        Tensor attn_output;
        Tensor attn_projected;
        Tensor attn_residual;
        Tensor mlp_input;
        Tensor gate;
        Tensor up;
        Tensor gate_act;
        Tensor gated;
        Tensor mlp_output;
        Tensor mlp_residual;
    };

    DecoderLayer(const Qwen3Config& config,
                 DeviceType device,
                 DataType dtype,
                 const VarBuilder& layer_builder)
        : config_(config),
          layer_builder_(layer_builder),
          device_(device),
          dtype_(dtype),
          q_hidden_size_(config.num_attention_heads * config.head_dim),
          kv_hidden_size_(config.kv_hidden_size()),
          input_norm_({config.hidden_size}, config.rms_norm_eps, true, device, dtype),
          q_norm_({config.head_dim}, config.rms_norm_eps, true, device, dtype),
          k_norm_({config.head_dim}, config.rms_norm_eps, true, device, dtype),
          post_attention_norm_({config.hidden_size}, config.rms_norm_eps, true, device, dtype) {}

    [[nodiscard]] Result<void> init() {
        auto input_norm_builder_result = layer_builder_.pp("input_layernorm");
        if (!input_norm_builder_result) {
            return Err<void>(std::move(input_norm_builder_result.error()));
        }
        auto input_norm_init = input_norm_.init(input_norm_builder_result.value());
        if (!input_norm_init) {
            return input_norm_init;
        }

        auto post_norm_builder_result = layer_builder_.pp("post_attention_layernorm");
        if (!post_norm_builder_result) {
            return Err<void>(std::move(post_norm_builder_result.error()));
        }
        auto post_norm_init = post_attention_norm_.init(post_norm_builder_result.value());
        if (!post_norm_init) {
            return post_norm_init;
        }

        auto attn_builder_result = layer_builder_.pp("self_attn");
        if (!attn_builder_result) {
            return Err<void>(std::move(attn_builder_result.error()));
        }

        auto q_norm_builder_result = attn_builder_result.value().pp("q_norm");
        if (!q_norm_builder_result) {
            return Err<void>(std::move(q_norm_builder_result.error()));
        }
        auto q_norm_init = q_norm_.init(q_norm_builder_result.value());
        if (!q_norm_init) {
            return q_norm_init;
        }

        auto k_norm_builder_result = attn_builder_result.value().pp("k_norm");
        if (!k_norm_builder_result) {
            return Err<void>(std::move(k_norm_builder_result.error()));
        }
        auto k_norm_init = k_norm_.init(k_norm_builder_result.value());
        if (!k_norm_init) {
            return k_norm_init;
        }

        auto q_proj_builder_result = attn_builder_result.value().pp("q_proj");
        if (!q_proj_builder_result) {
            return Err<void>(std::move(q_proj_builder_result.error()));
        }

        auto q_proj_result = pulse::layer::Linear::liner(config_.hidden_size,
                                                         q_hidden_size_,
                                                         false,
                                                         device_,
                                                         dtype_,
                                                         q_proj_builder_result.value());
        if (!q_proj_result) {
            return Err<void>(std::move(q_proj_result.error()));
        }

        q_proj_ = std::make_unique<pulse::layer::Linear>(std::move(q_proj_result.value()));

        auto k_proj_builder_result = attn_builder_result.value().pp("k_proj");
        if (!k_proj_builder_result) {
            return Err<void>(std::move(k_proj_builder_result.error()));
        }

        auto k_proj_result = pulse::layer::Linear::liner(config_.hidden_size,
                                                         kv_hidden_size_,
                                                         false,
                                                         device_,
                                                         dtype_,
                                                         k_proj_builder_result.value());
        if (!k_proj_result) {
            return Err<void>(std::move(k_proj_result.error()));
        }

        k_proj_ = std::make_unique<pulse::layer::Linear>(std::move(k_proj_result.value()));

        auto v_proj_builder_result = attn_builder_result.value().pp("v_proj");
        if (!v_proj_builder_result) {
            return Err<void>(std::move(v_proj_builder_result.error()));
        }

        auto v_proj_result = pulse::layer::Linear::liner(config_.hidden_size,
                                                         kv_hidden_size_,
                                                         false,
                                                         device_,
                                                         dtype_,
                                                         v_proj_builder_result.value());
        if (!v_proj_result) {
            return Err<void>(std::move(v_proj_result.error()));
        }
        v_proj_ = std::make_unique<pulse::layer::Linear>(std::move(v_proj_result.value()));

        auto o_proj_builder_result = attn_builder_result.value().pp("o_proj");
        if (!o_proj_builder_result) {
            return Err<void>(std::move(o_proj_builder_result.error()));
        }
        auto o_proj_result = pulse::layer::Linear::liner(q_hidden_size_,
                                                         config_.hidden_size,
                                                         false,
                                                         device_,
                                                         dtype_,
                                                         o_proj_builder_result.value());
        if (!o_proj_result) {
            return Err<void>(std::move(o_proj_result.error()));
        }
        o_proj_ = std::make_unique<pulse::layer::Linear>(std::move(o_proj_result.value()));

        auto mlp_builder_result = layer_builder_.pp("mlp");
        if (!mlp_builder_result) {
            return Err<void>(std::move(mlp_builder_result.error()));
        }

        auto gate_proj_builder_result = mlp_builder_result.value().pp("gate_proj");
        if (!gate_proj_builder_result) {
            return Err<void>(std::move(gate_proj_builder_result.error()));
        }
        auto gate_proj_result = pulse::layer::Linear::liner(config_.hidden_size,
                                                            config_.intermediate_size,
                                                            false,
                                                            device_,
                                                            dtype_,
                                                            gate_proj_builder_result.value());
        if (!gate_proj_result) {
            return Err<void>(std::move(gate_proj_result.error()));
        }
        gate_proj_ = std::make_unique<pulse::layer::Linear>(std::move(gate_proj_result.value()));

        auto up_proj_builder_result = mlp_builder_result.value().pp("up_proj");
        if (!up_proj_builder_result) {
            return Err<void>(std::move(up_proj_builder_result.error()));
        }
        auto up_proj_result = pulse::layer::Linear::liner(config_.hidden_size,
                                                          config_.intermediate_size,
                                                          false,
                                                          device_,
                                                          dtype_,
                                                          up_proj_builder_result.value());
        if (!up_proj_result) {
            return Err<void>(std::move(up_proj_result.error()));
        }
        up_proj_ = std::make_unique<pulse::layer::Linear>(std::move(up_proj_result.value()));

        auto down_proj_builder_result = mlp_builder_result.value().pp("down_proj");
        if (!down_proj_builder_result) {
            return Err<void>(std::move(down_proj_builder_result.error()));
        }
        auto down_proj_result = pulse::layer::Linear::liner(config_.intermediate_size,
                                                            config_.hidden_size,
                                                            false,
                                                            device_,
                                                            dtype_,
                                                            down_proj_builder_result.value());
        if (!down_proj_result) {
            return Err<void>(std::move(down_proj_result.error()));
        }
        down_proj_ = std::make_unique<pulse::layer::Linear>(std::move(down_proj_result.value()));

        auto workspace_result = init_workspace();
        if (!workspace_result) {
            return workspace_result;
        }

        initialized_ = true;
        return Ok();
    }

    [[nodiscard]] Result<void> forward(Tensor& hidden, i32 position, Tensor& key_cache, Tensor& value_cache) {
        if (!initialized_) {
            return Err<void>(ErrorCode::InvalidOperator, "Qwen3 decoder layer is not initialized");
        }

        auto shape_result = prepare_workspace_shapes();
        if (!shape_result) {
            return shape_result;
        }

        auto attn_input_result = input_norm_.forward(hidden, workspace_.attn_input);
        if (!attn_input_result) {
            return attn_input_result;
        }

        auto q_result = q_proj_->forward(workspace_.attn_input, workspace_.q);
        if (!q_result) {
            return q_result;
        }

        auto k_result = k_proj_->forward(workspace_.attn_input, workspace_.k);
        if (!k_result) {
            return k_result;
        }

        auto v_result = v_proj_->forward(workspace_.attn_input, workspace_.v);
        if (!v_result) {
            return v_result;
        }

        auto q_reshape_result = workspace_.q.reshape({config_.num_attention_heads, 1, config_.head_dim});
        if (!q_reshape_result) {
            return q_reshape_result;
        }
        auto q_norm_result = q_norm_.forward(workspace_.q, workspace_.q_norm);
        if (!q_norm_result) {
            return q_norm_result;
        }
        auto q_rope_result =
            ops::rope(workspace_.q_norm, workspace_.q_rope, position, static_cast<f32>(config_.rope_theta), config_.head_dim);
        if (!q_rope_result) {
            return q_rope_result;
        }
        auto q_rope_shape_result = workspace_.q_rope.reshape({1, q_hidden_size_});
        if (!q_rope_shape_result) {
            return q_rope_shape_result;
        }

        auto k_reshape_result = workspace_.k.reshape({config_.num_key_value_heads, 1, config_.head_dim});
        if (!k_reshape_result) {
            return k_reshape_result;
        }
        auto k_norm_result = k_norm_.forward(workspace_.k, workspace_.k_norm);
        if (!k_norm_result) {
            return k_norm_result;
        }
        auto k_rope_result =
            ops::rope(workspace_.k_norm, workspace_.k_rope, position, static_cast<f32>(config_.rope_theta), config_.head_dim);
        if (!k_rope_result) {
            return k_rope_result;
        }
        auto k_rope_shape_result = workspace_.k_rope.reshape({1, kv_hidden_size_});
        if (!k_rope_shape_result) {
            return k_rope_shape_result;
        }

        auto write_key_result = copy_cache_row(workspace_.k_rope, key_cache, position);
        if (!write_key_result) {
            return write_key_result;
        }
        auto write_value_result = copy_cache_row(workspace_.v, value_cache, position);
        if (!write_value_result) {
            return write_value_result;
        }

        auto mha_result = ops::mha(workspace_.q_rope,
                                   key_cache,
                                   value_cache,
                                   workspace_.attn_score,
                                   workspace_.attn_output,
                                   position,
                                   config_.num_attention_heads,
                                   config_.head_dim);
        if (!mha_result) {
            return mha_result;
        }

        auto attn_projected_result = o_proj_->forward(workspace_.attn_output, workspace_.attn_projected);
        if (!attn_projected_result) {
            return attn_projected_result;
        }
        auto attn_residual_result = ops::add(hidden, workspace_.attn_projected, workspace_.attn_residual);
        if (!attn_residual_result) {
            return attn_residual_result;
        }
        std::swap(hidden, workspace_.attn_residual);

        auto mlp_input_result = post_attention_norm_.forward(hidden, workspace_.mlp_input);
        if (!mlp_input_result) {
            return mlp_input_result;
        }

        auto gate_result = gate_proj_->forward(workspace_.mlp_input, workspace_.gate);
        if (!gate_result) {
            return gate_result;
        }

        auto up_result = up_proj_->forward(workspace_.mlp_input, workspace_.up);
        if (!up_result) {
            return up_result;
        }

        auto gate_act_result = silu_.forward(workspace_.gate, workspace_.gate_act);
        if (!gate_act_result) {
            return gate_act_result;
        }

        auto gated_result = ops::mul(workspace_.gate_act, workspace_.up, workspace_.gated);
        if (!gated_result) {
            return gated_result;
        }

        auto mlp_output_result = down_proj_->forward(workspace_.gated, workspace_.mlp_output);
        if (!mlp_output_result) {
            return mlp_output_result;
        }
        auto mlp_residual_result = ops::add(hidden, workspace_.mlp_output, workspace_.mlp_residual);
        if (!mlp_residual_result) {
            return mlp_residual_result;
        }
        std::swap(hidden, workspace_.mlp_residual);

        return Ok();
    }

private:
    [[nodiscard]] Result<void> init_workspace() {
        auto result = Tensor::create({1, config_.hidden_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.attn_input = std::move(result.value());

        result = Tensor::create({1, q_hidden_size_}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.q = std::move(result.value());

        result = Tensor::create({1, kv_hidden_size_}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.k = std::move(result.value());

        result = Tensor::create({1, kv_hidden_size_}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.v = std::move(result.value());

        result = Tensor::create({config_.num_attention_heads, 1, config_.head_dim}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.q_norm = std::move(result.value());

        result = Tensor::create({config_.num_key_value_heads, 1, config_.head_dim}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.k_norm = std::move(result.value());

        result = Tensor::create({config_.num_attention_heads, 1, config_.head_dim}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.q_rope = std::move(result.value());

        result = Tensor::create({config_.num_key_value_heads, 1, config_.head_dim}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.k_rope = std::move(result.value());

        const DataType score_dtype = device_ == DeviceType::CUDA ? DataType::Float32 : dtype_;
        result = Tensor::create({config_.num_attention_heads, config_.max_position_embeddings},
                                score_dtype,
                                device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.attn_score = std::move(result.value());

        result = Tensor::create({1, q_hidden_size_}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.attn_output = std::move(result.value());

        result = Tensor::create({1, config_.hidden_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.attn_projected = std::move(result.value());

        result = Tensor::create({1, config_.hidden_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.attn_residual = std::move(result.value());

        result = Tensor::create({1, config_.hidden_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.mlp_input = std::move(result.value());

        result = Tensor::create({1, config_.intermediate_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.gate = std::move(result.value());

        result = Tensor::create({1, config_.intermediate_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.up = std::move(result.value());

        result = Tensor::create({1, config_.intermediate_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.gate_act = std::move(result.value());

        result = Tensor::create({1, config_.intermediate_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.gated = std::move(result.value());

        result = Tensor::create({1, config_.hidden_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.mlp_output = std::move(result.value());

        result = Tensor::create({1, config_.hidden_size}, dtype_, device_);
        if (!result) {
            return Err<void>(std::move(result.error()));
        }
        workspace_.mlp_residual = std::move(result.value());
        return Ok();
    }

    [[nodiscard]] Result<void> prepare_workspace_shapes() {
        auto result = workspace_.q.reshape({1, q_hidden_size_});
        if (!result) {
            return result;
        }
        result = workspace_.k.reshape({1, kv_hidden_size_});
        if (!result) {
            return result;
        }
        result = workspace_.q_norm.reshape({config_.num_attention_heads, 1, config_.head_dim});
        if (!result) {
            return result;
        }
        result = workspace_.k_norm.reshape({config_.num_key_value_heads, 1, config_.head_dim});
        if (!result) {
            return result;
        }
        result = workspace_.q_rope.reshape({config_.num_attention_heads, 1, config_.head_dim});
        if (!result) {
            return result;
        }
        return workspace_.k_rope.reshape({config_.num_key_value_heads, 1, config_.head_dim});
    }

    Qwen3Config config_;
    VarBuilder layer_builder_;
    DeviceType device_ = DeviceType::CPU;
    DataType dtype_ = DataType::Float32;
    i32 q_hidden_size_ = 0;
    i32 kv_hidden_size_ = 0;

    pulse::layer::RMSNorm input_norm_;
    pulse::layer::RMSNorm q_norm_;
    pulse::layer::RMSNorm k_norm_;
    pulse::layer::RMSNorm post_attention_norm_;
    std::unique_ptr<pulse::layer::Linear> q_proj_;
    std::unique_ptr<pulse::layer::Linear> k_proj_;
    std::unique_ptr<pulse::layer::Linear> v_proj_;
    std::unique_ptr<pulse::layer::Linear> o_proj_;
    std::unique_ptr<pulse::layer::Linear> gate_proj_;
    std::unique_ptr<pulse::layer::Linear> up_proj_;
    std::unique_ptr<pulse::layer::Linear> down_proj_;
    pulse::layer::SiLU silu_;
    Workspace workspace_;
    bool initialized_ = false;
};

Qwen3Model::Qwen3Model(const Qwen3Config& config,
                       DeviceType device,
                       DataType dtype,
                       const VarBuilder& builder) noexcept
    : config_(config), builder_(builder), device_(device), dtype_(dtype) {}

Qwen3Model::Qwen3Model(Qwen3Model&&) noexcept = default;

Qwen3Model& Qwen3Model::operator=(Qwen3Model&&) noexcept = default;

Qwen3Model::~Qwen3Model() = default;

Result<void> Qwen3Model::init() {
    if (initialized_) {
        return Ok();
    }

    auto model_builder_result = builder_.pp("model");
    if (!model_builder_result) {
        return Err<void>(std::move(model_builder_result.error()));
    }

    auto embed_builder_result = model_builder_result.value().pp("embed_tokens");
    if (!embed_builder_result) {
        return Err<void>(std::move(embed_builder_result.error()));
    }
    auto embedding_result = pulse::layer::Embedding::embedding(config_.vocab_size,
                                                               config_.hidden_size,
                                                               device_,
                                                               dtype_,
                                                               embed_builder_result.value());
    if (!embedding_result) {
        return Err<void>(std::move(embedding_result.error()));
    }
    embedding_ = std::make_unique<pulse::layer::Embedding>(std::move(embedding_result.value()));

    auto norm_builder_result = model_builder_result.value().pp("norm");
    if (!norm_builder_result) {
        return Err<void>(std::move(norm_builder_result.error()));
    }
    final_norm_ = std::make_unique<pulse::layer::RMSNorm>(std::vector<i32>{config_.hidden_size},
                                                          config_.rms_norm_eps,
                                                          true,
                                                          device_,
                                                          dtype_);
    auto final_norm_init = final_norm_->init(norm_builder_result.value());
    if (!final_norm_init) {
        return final_norm_init;
    }

    auto final_norm_buffer_result = Tensor::create({1, config_.hidden_size}, dtype_, device_);
    if (!final_norm_buffer_result) {
        return Err<void>(std::move(final_norm_buffer_result.error()));
    }
    final_norm_buffer_ = std::move(final_norm_buffer_result.value());

    auto lm_head_builder_result = builder_.pp("lm_head");
    if (lm_head_builder_result) {
        auto lm_head_result = pulse::layer::Linear::liner(config_.hidden_size,
                                                          config_.vocab_size,
                                                          false,
                                                          device_,
                                                          dtype_,
                                                          lm_head_builder_result.value());
        if (!lm_head_result) {
            return Err<void>(std::move(lm_head_result.error()));
        }
        lm_head_linear_ = std::make_unique<pulse::layer::Linear>(std::move(lm_head_result.value()));
    }

    auto layers_builder_result = model_builder_result.value().pp("layers");
    if (!layers_builder_result) {
        return Err<void>(std::move(layers_builder_result.error()));
    }

    layers_.clear();
    layers_.reserve(static_cast<usize>(config_.num_hidden_layers));
    for (i32 layer_idx = 0; layer_idx < config_.num_hidden_layers; ++layer_idx) {
        auto layer_builder_result = layers_builder_result.value().pp(std::to_string(layer_idx));
        if (!layer_builder_result) {
            return Err<void>(std::move(layer_builder_result.error()));
        }

        auto layer = std::make_unique<DecoderLayer>(config_, device_, dtype_, layer_builder_result.value());
        auto layer_init = layer->init();
        if (!layer_init) {
            return layer_init;
        }
        layers_.push_back(std::move(layer));
    }

    key_cache_.clear();
    value_cache_.clear();
    key_cache_.reserve(static_cast<usize>(config_.num_hidden_layers));
    value_cache_.reserve(static_cast<usize>(config_.num_hidden_layers));
    for (i32 layer_idx = 0; layer_idx < config_.num_hidden_layers; ++layer_idx) {
        auto key_result =
            Tensor::zeros({config_.max_position_embeddings, config_.kv_hidden_size()}, dtype_, device_);
        if (!key_result) {
            return Err<void>(std::move(key_result.error()));
        }
        key_cache_.push_back(std::move(key_result.value()));

        auto value_result =
            Tensor::zeros({config_.max_position_embeddings, config_.kv_hidden_size()}, dtype_, device_);
        if (!value_result) {
            return Err<void>(std::move(value_result.error()));
        }
        value_cache_.push_back(std::move(value_result.value()));
    }

    initialized_ = true;
    return Ok();
}

Result<void> Qwen3Model::forward(i32 token, i32 position, Tensor& logits) {
    if (!initialized_) {
        return Err<void>(ErrorCode::InvalidOperator, "Qwen3Model is not initialized");
    }

    if (position < 0 || position >= config_.max_position_embeddings) {
        return Err<void>(ErrorCode::InvalidArgument, "Position out of bounds");
    }

    auto token_result = Tensor::from_vector<i32>({token}, device_);
    if (!token_result) {
        return Err<void>(std::move(token_result.error()));
    }

    auto hidden_result = embedding_->forward(token_result.value());
    if (!hidden_result) {
        return Err<void>(std::move(hidden_result.error()));
    }
    Tensor hidden(std::move(hidden_result.value()));

    for (i32 layer_idx = 0; layer_idx < config_.num_hidden_layers; ++layer_idx) {
        auto layer_result =
            layers_[static_cast<usize>(layer_idx)]->forward(hidden,
                                                            position,
                                                            key_cache_[static_cast<usize>(layer_idx)],
                                                            value_cache_[static_cast<usize>(layer_idx)]);
        if (!layer_result) {
            return layer_result;
        }
    }

    auto norm_result = final_norm_->forward(hidden, final_norm_buffer_);
    if (!norm_result) {
        return norm_result;
    }

    if (lm_head_linear_ != nullptr) {
        if (logits.device() == device_) {
            return lm_head_linear_->forward(final_norm_buffer_, logits);
        }

        auto device_logits_result = Tensor::create({1, config_.vocab_size}, dtype_, device_);
        if (!device_logits_result) {
            return Err<void>(std::move(device_logits_result.error()));
        }

        auto forward_result = lm_head_linear_->forward(final_norm_buffer_, device_logits_result.value());
        if (!forward_result) {
            return forward_result;
        }

        auto to_output_device_result = device_logits_result.value().to(logits.device());
        if (!to_output_device_result) {
            return Err<void>(std::move(to_output_device_result.error()));
        }

        return copy_tensor_bytes(to_output_device_result.value(), logits);
    }

    if (config_.tie_word_embeddings && embedding_->weight() != nullptr) {
        if (logits.device() == device_) {
            return ops::matmul(final_norm_buffer_, *embedding_->weight(), logits, false, true);
        }

        auto device_logits_result = Tensor::create({1, config_.vocab_size}, dtype_, device_);
        if (!device_logits_result) {
            return Err<void>(std::move(device_logits_result.error()));
        }

        auto matmul_result =
            ops::matmul(final_norm_buffer_, *embedding_->weight(), device_logits_result.value(), false, true);
        if (!matmul_result) {
            return matmul_result;
        }

        auto to_output_device_result = device_logits_result.value().to(logits.device());
        if (!to_output_device_result) {
            return Err<void>(std::move(to_output_device_result.error()));
        }

        return copy_tensor_bytes(to_output_device_result.value(), logits);
    }

    return Err<void>(ErrorCode::InvalidOperator, "lm_head is not initialized");
}

void Qwen3Model::reset_cache() {
    for (auto& key_cache : key_cache_) {
        auto zero_result = Tensor::zeros(key_cache.dims(), key_cache.dtype(), key_cache.device());
        if (zero_result) {
            key_cache = std::move(zero_result.value());
        }
    }

    for (auto& value_cache : value_cache_) {
        auto zero_result = Tensor::zeros(value_cache.dims(), value_cache.dtype(), value_cache.device());
        if (zero_result) {
            value_cache = std::move(zero_result.value());
        }
    }
}

}  // namespace pulse::model
