#include "pulse/ops/mha.hpp"

#include <span>

#include "pulse/ops/kernels/mha_kernel.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/mha_kernel.cuh"
#endif

namespace pulse::ops {

namespace {

Result<void> validate_mha_inputs(const Tensor& query,
                                 const Tensor& key_cache,
                                 const Tensor& value_cache,
                                 const Tensor& output,
                                 i32 pos,
                                 i32 head_num,
                                 i32 head_size) {
    if (query.device() != key_cache.device() || query.device() != value_cache.device() ||
        query.device() != output.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "All tensors must be on the same device");
    }

    if (query.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Query tensor is empty");
    }

    if (key_cache.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Key cache tensor is empty");
    }

    if (value_cache.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Value cache tensor is empty");
    }

    if (output.empty()) {
        return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
    }

    if (head_num <= 0 || head_size <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "head_num and head_size must be positive");
    }

    if (key_cache.ndim() != 2 || value_cache.ndim() != 2) {
        return Err<void>(ErrorCode::InvalidArgument, "Key/value cache tensors must be 2D");
    }

    if (key_cache.dims() != value_cache.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Key/value cache tensor shape mismatch");
    }

    if (query.dims() != output.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Output tensor shape mismatch");
    }

    if (query.dtype() != key_cache.dtype() || query.dtype() != value_cache.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Input tensors dtype mismatch");
    }

    if (query.dtype() != output.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Output tensor dtype mismatch");
    }

    const i64 expected_query_size = static_cast<i64>(head_num) * static_cast<i64>(head_size);
    if (static_cast<i64>(query.size()) != expected_query_size) {
        return Err<void>(ErrorCode::ShapeMismatch, "Query tensor size does not match head layout");
    }

    if (output.size() != query.size()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Output tensor size mismatch");
    }

    const i32 seq_len = key_cache.dim(0);
    const i32 kv_dim = key_cache.dim(1);
    if (seq_len <= 0 || kv_dim <= 0) {
        return Err<void>(ErrorCode::InvalidArgument, "Key/value cache dimensions must be positive");
    }

    if (pos < 0 || pos >= seq_len) {
        return Err<void>(ErrorCode::InvalidArgument, "Position out of bounds");
    }

    if ((kv_dim % head_size) != 0) {
        return Err<void>(ErrorCode::ShapeMismatch, "kv_dim must be divisible by head_size");
    }

    const i32 kv_head_num = kv_dim / head_size;
    if (kv_head_num <= 0 || (head_num % kv_head_num) != 0) {
        return Err<void>(ErrorCode::ShapeMismatch, "head_num must be divisible by kv_head_num");
    }

    return Ok();
}

template<typename T>
Result<void> mha_cpu(const Tensor& query,
                     const Tensor& key_cache,
                     const Tensor& value_cache,
                     Tensor& output,
                     Tensor& score,
                     i32 pos,
                     i32 head_num,
                     i32 head_size) {
    const i32 seq_len = key_cache.dim(0);
    const i32 kv_dim = key_cache.dim(1);
    const i32 kv_mul = head_num / (kv_dim / head_size);

    std::span<const T> query_span(query.ptr<T>(), query.size());
    std::span<const T> key_span(key_cache.ptr<T>(), key_cache.size());
    std::span<const T> value_span(value_cache.ptr<T>(), value_cache.size());
    std::span<T> output_span(output.ptr<T>(), output.size());
    std::span<T> score_span(score.ptr<T>(), score.size());
    kernels::mha<T>(query_span,
                    key_span,
                    value_span,
                    output_span,
                    score_span,
                    pos,
                    kv_dim,
                    head_num,
                    head_size,
                    seq_len,
                    kv_mul);
    return Ok();
}

}  // namespace

Result<void> mha(const Tensor& query,
                 const Tensor& key_cache,
                 const Tensor& value_cache,
                 Tensor& output,
                 i32 pos,
                 i32 head_num,
                 i32 head_size) {
    auto validation = validate_mha_inputs(query, key_cache, value_cache, output, pos, head_num, head_size);
    if (!validation) {
        return validation;
    }

    const i32 seq_len = key_cache.dim(0);
    const i32 kv_dim = key_cache.dim(1);
    const i32 kv_mul = head_num / (kv_dim / head_size);

    auto score_result = Tensor::create({head_num, seq_len}, query.dtype(), query.device());
    if (!score_result) {
        return Err<void>(std::move(score_result.error()));
    }
    Tensor score(std::move(score_result.value()));

    if (query.device() == DeviceType::CPU) {
        if (query.dtype() == DataType::Float32) {
            return mha_cpu<f32>(query, key_cache, value_cache, output, score, pos, head_num, head_size);
        }

        if (query.dtype() == DataType::Float64) {
            return mha_cpu<f64>(query, key_cache, value_cache, output, score, pos, head_num, head_size);
        }

        return Err<void>(ErrorCode::NotImplemented, "CPU MHA only supports Float32/Float64");
    }

#ifdef PULSE_USE_CUDA
    if (query.device() == DeviceType::CUDA) {
        return kernels::cuda::mha_cuda_launch(query.data(),
                                              key_cache.data(),
                                              value_cache.data(),
                                              score.data(),
                                              output.data(),
                                              pos,
                                              seq_len,
                                              kv_dim,
                                              head_num,
                                              head_size,
                                              kv_mul,
                                              query.dtype(),
                                              nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "MHA operation not implemented for this device");
}

}  // namespace pulse::ops
