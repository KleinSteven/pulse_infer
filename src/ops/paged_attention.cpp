#include "pulse/ops/paged_attention.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/paged_attention.cuh"
#endif

namespace pulse::ops {

Result<void> paged_attention(const Tensor& query,
                             const Tensor& key_cache,
                             const Tensor& value_cache,
                             const Tensor& block_table,
                             const Tensor& context_lens,
                             Tensor& output) {
    if (query.device() != key_cache.device() || query.device() != value_cache.device() ||
        query.device() != block_table.device() || query.device() != context_lens.device() ||
        query.device() != output.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "Paged attention tensors must be on the same device");
    }

    if (query.dtype() != key_cache.dtype() || query.dtype() != value_cache.dtype() || query.dtype() != output.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Paged attention dtype mismatch");
    }

    if (block_table.dtype() != DataType::Int32 || context_lens.dtype() != DataType::Int32) {
        return Err<void>(ErrorCode::DtypeMismatch, "Paged attention block_table and context_lens must be Int32");
    }

    if (query.ndim() != 3 || output.ndim() != 3 || query.dims() != output.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged attention query/output tensors must be matching 3D tensors");
    }

    if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.dims() != value_cache.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged attention cache tensors must be matching 4D tensors");
    }

    if (block_table.ndim() != 2 || context_lens.ndim() != 1 || block_table.dim(0) != query.dim(0) ||
        context_lens.dim(0) != query.dim(0)) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged attention block table or context_lens shape mismatch");
    }

    if (query.dim(1) <= 0 || key_cache.dim(1) <= 0 || query.dim(1) % key_cache.dim(1) != 0 ||
        query.dim(2) != key_cache.dim(3)) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged attention head layout mismatch");
    }

#ifdef PULSE_USE_CUDA
    if (query.device() == DeviceType::CUDA) {
        return kernels::cuda::paged_attention_cuda_launch(query.data(),
                                                          key_cache.data(),
                                                          value_cache.data(),
                                                          block_table.ptr<i32>(),
                                                          context_lens.ptr<i32>(),
                                                          output.data(),
                                                          query.dim(0),
                                                          query.dim(1),
                                                          key_cache.dim(1),
                                                          query.dim(2),
                                                          key_cache.dim(2),
                                                          block_table.dim(1),
                                                          query.dtype(),
                                                          nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "Paged attention only supports CUDA");
}

}  // namespace pulse::ops
