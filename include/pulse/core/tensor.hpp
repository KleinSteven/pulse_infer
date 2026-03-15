#pragma once

#include <cstring>
#include <format>
#include <vector>

#include "pulse/core/buffer.hpp"
#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse {

class Tensor {
public:
    Tensor() = default;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    [[nodiscard]] static Result<Tensor> create(std::vector<i32> dims,
                                               DataType dtype,
                                               DeviceType device = DeviceType::CPU);

    [[nodiscard]] static Result<Tensor> zeros(std::vector<i32> dims,
                                              DataType dtype,
                                              DeviceType device = DeviceType::CPU);


    template<typename T>
    [[nodiscard]] static Result<Tensor> from_vector(const std::vector<T>& data,
                                                    DeviceType device = DeviceType::CPU) {
        DataType dtype = cpp_type_to_data_type_v<T>;

        std::vector<i32> dims{static_cast<i32>(data.size())};

        auto result = create(dims, dtype, device);

        if (!result) {
            return Err<Tensor>(std::move(result.error()));
        }

        Tensor tensor(std::move(result.value()));

        if (device == DeviceType::CPU) {
            std::memcpy(tensor.data(), data.data(), data.size() * sizeof(T));
            return Ok(std::move(tensor));
        }

#ifdef PULSE_USE_CUDA
        if (device == DeviceType::CUDA) {
            cudaError_t err = cudaMemcpy(tensor.data(),
                                         data.data(),
                                         data.size() * sizeof(T),
                                         cudaMemcpyKind::cudaMemcpyHostToDevice);

            if (err != cudaSuccess) {
                auto str = std::format("cudaMemcpy failed: {}", cudaGetErrorString(err));
                return Err<Tensor>(ErrorCode::CudaError, str);
            }

            return Ok(std::move(tensor));
        }
#endif

        return Err<Tensor>(ErrorCode::InvalidArgument, "Unsupported device");
    }

    [[nodiscard]] Result<Tensor> to(DeviceType device) const;

    [[nodiscard]] Result<Tensor> add(const Tensor& other) const;

    [[nodiscard]] const Buffer& buffer() const noexcept {
        return buffer_;
    }

    [[nodiscard]] i32 ndim() const noexcept {
        return static_cast<i32>(dims_.size());
    }

    [[nodiscard]] i32 dim(i32 index) const {
        if (index < 0 || index >= ndim()) {
            return 0;
        }
        return dims_[static_cast<usize>(index)];
    }


    [[nodiscard]] const std::vector<i32>& dims() const noexcept {
        return dims_;
    }

    [[nodiscard]] DataType dtype() const noexcept {
        return dtype_;
    }

    [[nodiscard]] DeviceType device() const noexcept {
        return device_;
    }

    [[nodiscard]] usize size() const noexcept {
        return size_;
    }

    [[nodiscard]] usize byte_size() const noexcept {
        return size_ * data_type_size(dtype_);
    }

    [[nodiscard]] void* data() noexcept {
        return buffer_.data();
    }

    [[nodiscard]] const void* data() const noexcept {
        return buffer_.data();
    }

    template<typename T>
    [[nodiscard]] T* ptr() noexcept {
        return static_cast<T*>(data());
    }

    template<typename T>
    [[nodiscard]] const T* ptr() const noexcept {
        return static_cast<const T*>(data());
    }

    template<typename T>
    [[nodiscard]] T* ptr(i64 offset) noexcept {
        return static_cast<T*>(data()) + offset;
    }

    template<typename T>
    [[nodiscard]] const T* ptr(i64 offset) const noexcept {
        return static_cast<const T*>(data()) + offset;
    }

    template<typename T>
    [[nodiscard]] T& index(i64 idx) noexcept {
        return static_cast<T*>(data())[idx];
    }

    template<typename T>
    [[nodiscard]] const T& index(i64 idx) const noexcept {
        return static_cast<const T*>(data())[idx];
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] std::vector<usize> stride() const;

    [[nodiscard]] Result<void> reshape(const std::vector<i32>& new_dims);

    [[nodiscard]] Result<Tensor> clone() const;

    [[nodiscard]] std::string to_string() const;


private:
    Tensor(Buffer&& buffer, std::vector<i32> dims, DataType dtype, DeviceType device);

    [[nodiscard]] static usize compute_size(const std::vector<i32> dims) noexcept;

    Buffer buffer_;
    std::vector<i32> dims_;
    usize size_ = 0;
    DataType dtype_ = DataType::Float32;
    DeviceType device_ = DeviceType::CPU;
};

}  // namespace pulse
