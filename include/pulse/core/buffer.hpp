#pragma once

#include <cstring>
#include <format>
#include <span>

#include "pulse/core/allocator.hpp"
#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

#ifdef PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace pulse {

/**
 * @class Buffer
 * @brief RAII wrapper for device memory
 *
 * Buffer manages a contiguous block of memory on a specific device.
 * It provides ownership semantics and automatic deallocation.
 */
class Buffer {
public:
    Buffer() = default;

    ~Buffer() {
        free();
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_), device_(other.device_), alignment_(other.alignment_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != std::addressof(other)) {
            // free old this
            free();
            data_ = other.data_;
            size_ = other.size_;
            device_ = other.device_;
            alignment_ = other.alignment_;

            other.data_ = nullptr;
            other.size_ = 0;
        }

        return *this;
    }

    [[nodiscard]] static Result<Buffer> create(usize size,
                                               DeviceType device = DeviceType::CPU,
                                               usize alignment = 64) {
        if (size == 0) {
            return Err<Buffer>(ErrorCode::InvalidArgument, "Cannot create a Buffer with zero size");
        }

        if (device == DeviceType::CPU) {
            CPUAllocator allocator;
            Result<void*> result = allocator.allocate(size, alignment);
            if (!result) {
                return Err<Buffer>(std::move(result.error()));
            }

            return Ok(Buffer(result.value(), size, device, alignment));
        }

#ifdef PULSE_USE_CUDA
        if (device == DeviceType::CUDA) {
            CUDAAllocator allocator;
            Result<void*> result = allocator.allocate(size, alignment);
            if (!result) {
                return Err<Buffer>(std::move(result.error()));
            }

            return Ok(Buffer(result.value(), size, device, alignment));
        }
#endif

        return Err<Buffer>(ErrorCode::InvalidArgument, "Unsupported Device type");
    }

    [[nodiscard]] usize size() const noexcept {
        return size_;
    }

    [[nodiscard]] usize alignment() const noexcept {
        return alignment_;
    }

    [[nodiscard]] DeviceType device() const noexcept {
        return device_;
    }

    [[nodiscard]] void* data() noexcept {
        return data_;
    }

    [[nodiscard]] const void* data() const noexcept {
        return data_;
    }

    template<typename T>
    [[nodiscard]] T* data_as() noexcept {
        return static_cast<T*>(data_);
    }

    template<typename T>
    [[nodiscard]] const T* data_as() const noexcept {
        return static_cast<T*>(data_);
    }

    template<typename T>
    [[nodiscard]] usize num_elements() const noexcept {
        return size_ / sizeof(T);
    }

    [[nodiscard]] bool empty() const noexcept {
        return data_ == nullptr || size_ == 0;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return !empty();
    }

    template<typename T>
    [[nodiscard]] std::span<T> as_span() {
        if (device_ != DeviceType::CPU) {
            return {};
        }

        return std::span<T>(data_as<T>(), num_elements<T>());
    }

    template<typename T>
    [[nodiscard]] std::span<const T> as_span() const {
        if (device_ != DeviceType::CPU) {
            return {};
        }

        return std::span<const T>(data_as<T>(), num_elements<T>());
    }

    [[nodiscard]] Result<void> copy_from(const Buffer& src) {
        if (size_ != src.size_) {
            return Err<void>(ErrorCode::InvalidArgument,
                             "Source and destination buffers must have same size");
        }

        if (src.empty())
            return Ok();

        if (device_ == DeviceType::CPU && src.device_ == DeviceType::CPU) {
            std::memcpy(data_, src.data_, size_);
            return Ok();
        }

#ifdef PULSE_USE_CUDA
        cudaMemcpyKind kind;

        if (device_ == DeviceType::CUDA && src.device_ == DeviceType::CUDA) {
            kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
        } else if (device_ == DeviceType::CUDA && src.device_ == DeviceType::CPU) {
            kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
        } else if (device_ == DeviceType::CPU && src.device_ == DeviceType::CUDA) {
            kind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
        } else {
            return Err<void>(ErrorCode::InvalidArgument, "Unsupported device combination");
        }

        cudaError_t err = cudaMemcpy(data_, src.data_, size_, kind);

        if (err != cudaSuccess) {
            auto str = std::format("cudaMemcpy failed: {}", cudaGetErrorString(err));
            return Err<void>(ErrorCode::CudaError, str);
        }

        return Ok();
#else
        return Err<void>(ErrorCode::InvalidArgument, "CUDA support not enabled");
#endif
    }

    [[nodiscard]] Result<Buffer> clone() const {
        if (empty())
            return Ok(Buffer());

        auto buffer_res = Buffer::create(size_, device_, alignment_);

        if (!buffer_res) {
            return Err<Buffer>(std::move(buffer_res.error()));
        }

        Buffer new_buf(std::move(buffer_res.value()));

        auto copy_res = new_buf.copy_from(*this);

        if (!copy_res) {
            return Err<Buffer>(std::move(copy_res.error()));
        }

        return Ok(std::move(new_buf));
    }

    [[nodiscard]] Result<void> zero() {
        if (empty())
            return Ok();

        if (device_ == DeviceType::CPU) {
            std::memset(data_, 0, size_);
            return Ok();
        }

#ifdef PULSE_USE_CUDA
        if (device_ == DeviceType::CUDA) {
            cudaError_t err = cudaMemset(data_, 0, size_);

            if (err != cudaSuccess) {
                auto str = std::format("cudaMemset failed : {}", cudaGetErrorString(err));
                return Err<void>(ErrorCode::CudaError, str);
            }

            return Ok();
        }
#endif

        return Err<void>(ErrorCode::InvalidArgument, "Unsupported device");
    }

private:
    Buffer(void* data, usize size, DeviceType device, usize alignment)
        : data_(data), size_(size), device_(device), alignment_(alignment) {}

    void free() {
        if (data_ != nullptr) {
            if (device_ == DeviceType::CPU) {
                CPUAllocator allocator;
                auto result = allocator.deallocate(data_, size_);
                (void)result;
            }
#ifdef PULSE_USE_CUDA
            else if (device_ == DeviceType::CUDA) {
                CUDAAllocator allocator;
                auto result = allocator.deallocate(data_, size_);
                (void)result;
            }
#endif

            data_ = nullptr;
            size_ = 0;
        }
    }

    void* data_ = nullptr;
    usize size_ = 0;
    DeviceType device_ = DeviceType::CPU;
    usize alignment_ = 64;
};

}  // namespace pulse
