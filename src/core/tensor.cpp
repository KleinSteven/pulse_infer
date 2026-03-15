#include "pulse/core/tensor.hpp"

#include <numeric>
#include <sstream>

#include "pulse/ops/add.hpp"

#ifdef PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif


namespace pulse {

Tensor::Tensor(Buffer&& buffer, std::vector<i32> dims, DataType dtype, DeviceType device)
    : buffer_(std::move(buffer)),
      dims_(std::move(dims)),
      size_(compute_size(dims_)),
      dtype_(dtype),
      device_(device) {}

usize Tensor::compute_size(const std::vector<i32> dims) noexcept {
    if (dims.empty())
        return 1;

    return std::accumulate(dims.begin(), dims.end(), usize(1), [](usize lhs, usize rhs) {
        return lhs * rhs;
    });
}

Result<Tensor> Tensor::create(std::vector<i32> dims, DataType dtype, DeviceType device) {
    usize size = compute_size(dims);
    usize type_size = data_type_size(dtype);
    usize total_bytes = size * type_size;

    if (total_bytes == 0) {
        return Err<Tensor>(ErrorCode::InvalidArgument, "Cannot create a tensor with zero size");
    }

    auto buffer_result = Buffer::create(total_bytes, device);

    if (!buffer_result) {
        return Err<Tensor>(std::move(buffer_result.error()));
    }

    return Ok(Tensor(std::move(buffer_result.value()), std::move(dims), dtype, device));
}

Result<Tensor> Tensor::zeros(std::vector<i32> dims, DataType dtype, DeviceType device) {
    auto buffer_result = create(dims, dtype, device);

    if (!buffer_result) {
        return Err<Tensor>(std::move(buffer_result.error()));
    }

    Tensor tensor(std::move(buffer_result.value()));

    auto zero_result = tensor.buffer_.zero();

    if (!zero_result) {
        return Err<Tensor>(std::move(zero_result.error()));
    }

    return Ok(std::move(tensor));
}

Result<Tensor> Tensor::to(DeviceType device) const {
    if (device_ == device) {
        auto result = Tensor::create(dims_, dtype_, device);
        if (!result) {
            return result;
        }

        Tensor tensor(std::move(result.value()));

        if (device == DeviceType::CPU) {
            std::memcpy(tensor.data(), data(), byte_size());
            return Ok(std::move(tensor));
        }

#ifdef PULSE_USE_CUDA
        if (device == DeviceType::CUDA) {
            cudaError_t err =
                cudaMemcpy(tensor.data(), data(), byte_size(), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

            if (err != cudaSuccess) {
                auto str = std::format("cudaMemcpy falied: {}", cudaGetErrorString(err));
                return Err<Tensor>(ErrorCode::CudaError, str);
            }

            return Ok(std::move(tensor));
        }
#endif

        return Err<Tensor>(ErrorCode::InvalidArgument, "NotSupport this device");
    }

#ifdef PULSE_USE_CUDA
    auto result = Tensor::create(dims_, dtype_, device);

    if (!result) {
        return result;
    }

    Tensor tensor(std::move(result.value()));

    cudaMemcpyKind kind;

    if (device_ == DeviceType::CPU && device == DeviceType::CUDA) {
        kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
    } else if (device_ == DeviceType::CUDA && device == DeviceType::CPU) {
        kind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
    } else {
        return Err<Tensor>(ErrorCode::InvalidArgument, "NotSupport this device");
    }

    cudaError_t err = cudaMemcpy(tensor.data(), data(), byte_size(), kind);

    if (err != cudaSuccess) {
        auto str = std::format("cudaMemcpy falied: {}", cudaGetErrorString(err));
        return Err<Tensor>(ErrorCode::CudaError, str);
    }

    return Ok(std::move(tensor));
#endif

    return Err<Tensor>(ErrorCode::InvalidArgument, "NotSupport this device");
}

Result<Tensor> Tensor::add(const Tensor& other) const {
    auto output_result = Tensor::create(dims_, dtype_, device_);
    if (!output_result) {
        return Err<Tensor>(std::move(output_result.error()));
    }

    Tensor output(std::move(output_result.value()));
    auto add_result = ops::add(*this, other, output);
    if (!add_result) {
        return Err<Tensor>(std::move(add_result.error()));
    }

    return Ok(std::move(output));
}

std::vector<usize> Tensor::stride() const {
    if (dims_.empty()) {
        return {};
    }

    std::vector<usize> strides(dims_.size());

    strides.back() = 1;


    for (i32 index = static_cast<i32>(dims_.size()) - 2; index >= 0; --index) {
        size_t u_index = static_cast<size_t>(index);
        strides[u_index] = strides[u_index + 1] * static_cast<size_t>(dims_[u_index + 1]);
    }

    return strides;
}

Result<void> Tensor::reshape(const std::vector<i32>& new_dims) {
    usize new_size = compute_size(new_dims);

    if (new_size != size_) {
        auto str =
            std::format("Cannot reshape: element count mismatch. Current: {}, New: {}", size_, new_size);
        return Err<void>(ErrorCode::InvalidArgument, str);
    }

    dims_ = new_dims;
    return Ok();
}

Result<Tensor> Tensor::clone() const {
    auto result = buffer_.clone();

    if (!result) {
        return Err<Tensor>(std::move(result.error()));
    }

    return Ok(Tensor(std::move(result.value()), dims_, dtype_, device_));
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";

    for (size_t i = 0; i < dims_.size(); ++i) {
        oss << dims_[i];
        if (i < dims_.size() - 1) {
            oss << ", ";
        }
    }

    oss << "], dtype=" << data_type_str(dtype_) << ", device=" << device_type_str(device_)
        << ", size=" << size_ << ")";

    return oss.str();
}

}  // namespace pulse
