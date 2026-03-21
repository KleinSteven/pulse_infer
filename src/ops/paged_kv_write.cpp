#include "pulse/ops/paged_kv_write.hpp"

#ifdef PULSE_USE_CUDA
#include "pulse/ops/kernels/cuda/paged_kv_write.cuh"
#endif

namespace pulse::ops {

Result<void> paged_kv_write(const Tensor& key,
                            const Tensor& value,
                            Tensor& key_cache,
                            Tensor& value_cache,
                            const Tensor& block_table,
                            const Tensor& positions) {
    if (key.device() != value.device() || key.device() != key_cache.device() || key.device() != value_cache.device() ||
        key.device() != block_table.device() || key.device() != positions.device()) {
        return Err<void>(ErrorCode::DeviceMismatch, "Paged KV write tensors must be on the same device");
    }

    if (key.dtype() != value.dtype() || key.dtype() != key_cache.dtype() || key.dtype() != value_cache.dtype()) {
        return Err<void>(ErrorCode::DtypeMismatch, "Paged KV write dtype mismatch");
    }

    if (block_table.dtype() != DataType::Int32 || positions.dtype() != DataType::Int32) {
        return Err<void>(ErrorCode::DtypeMismatch, "Paged KV write block_table and positions must be Int32");
    }

    if (key.ndim() != 3 || value.ndim() != 3) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged KV write key/value tensors must be 3D");
    }

    if (key.dims() != value.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged KV write key/value shape mismatch");
    }

    if (key_cache.ndim() != 4 || value_cache.ndim() != 4 || key_cache.dims() != value_cache.dims()) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged KV write cache tensor shape mismatch");
    }

    if (block_table.ndim() != 2 || positions.ndim() != 1 || block_table.dim(0) != key.dim(0) ||
        positions.dim(0) != key.dim(0)) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged KV write block table or positions shape mismatch");
    }

    if (key.dim(1) != key_cache.dim(1) || key.dim(2) != key_cache.dim(3) || key_cache.dim(2) <= 0) {
        return Err<void>(ErrorCode::ShapeMismatch, "Paged KV write KV cache layout mismatch");
    }

#ifdef PULSE_USE_CUDA
    if (key.device() == DeviceType::CUDA) {
        return kernels::cuda::paged_kv_write_cuda_launch(key.data(),
                                                         value.data(),
                                                         key_cache.data(),
                                                         value_cache.data(),
                                                         block_table.ptr<i32>(),
                                                         positions.ptr<i32>(),
                                                         key.dim(0),
                                                         key.dim(1),
                                                         key.dim(2),
                                                         key_cache.dim(2),
                                                         block_table.dim(1),
                                                         key.dtype(),
                                                         nullptr);
    }
#endif

    return Err<void>(ErrorCode::NotImplemented, "Paged KV write only supports CUDA");
}

}  // namespace pulse::ops
