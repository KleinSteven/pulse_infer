#pragma once

#include <cstdlib>
#include <memory>
#include <string>

#if defined(_WIN32) || defined(_WIN64)

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#endif

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse {

template<typename T>
concept Allocator = requires(T alloc, usize size, usize, usize alignment, void* ptr) {
    { alloc.allocate(size, alignment) } -> std::same_as<Result<void*>>;
    { alloc.deallocate(ptr, size) } -> std::same_as<Result<void>>;
    { alloc.device_type() } -> std::same_as<DeviceType>;
};

class CPUAllocator {
public:
    /**
     * @brief Most of modern CPU's cache line size is 64
     */
    static constexpr usize kDefaultAlignment = 64;

    /**
     * @brief Allocate Unified memory using Thrust
     *
     * @param size Number of bytes to allocate
     * @param alignment Memory alignment (Most of modern CPU's cache line size is 64)
     * @return Pointer to allocated memory or error
     */
    [[nodiscard]] Result<void*> allocate(usize size, usize alignment = kDefaultAlignment);

    [[nodiscard]] Result<void> deallocate(void* ptr, usize size) const;

    [[nodiscard]] constexpr DeviceType device_type() const noexcept {
        return DeviceType::CPU;
    }

    [[nodiscard]] constexpr usize default_alignment() const noexcept;
};

// Verify CPUAllocator satisfies Allocator concept
static_assert(Allocator<CPUAllocator>);

#ifdef PULSE_USE_CUDA
class CUDAAllocator {
public:
    /**
     * @brief Allocate CUDA memory using Thrust
     *
     * @param size Number of bytes to allocate
     * @param alignment Memory alignment (ignored, CUDAAllocator handles this)
     * @return Pointer to allocated memory or error
     */
    [[nodiscard]] Result<void*> allocate(usize size, usize alignment);

    [[nodiscard]] Result<void> deallocate(void* ptr, usize size) const;

    [[nodiscard]] constexpr DeviceType device_type() const noexcept {
        return DeviceType::CUDA;
    }
};

// Verify CUDAAllocator satisfies Allocator concept
static_assert(Allocator<CUDAAllocator>);

class UnifiedAllocator {
public:
    /**
     * @brief Allocate Unified memory using Thrust
     *
     * @param size Number of bytes to allocate
     * @param alignment Memory alignment (ignored, UnifiedAllocator handles this)
     * @return Pointer to allocated memory or error
     */
    [[nodiscard]] Result<void*> allocate(usize size, usize alignment);

    [[nodiscard]] Result<void> deallocate(void* ptr, usize size) const;

    [[nodiscard]] constexpr DeviceType device_type() const noexcept {
        return DeviceType::Unified;
    }
};

// Verify UnifiedAllocator satisfies Allocator concept
static_assert(Allocator<UnifiedAllocator>);

#endif

class MmapAllocator {
public:
    explicit MmapAllocator(const std::string& file_path);

    ~MmapAllocator();

    MmapAllocator(const MmapAllocator&) = delete;

    MmapAllocator& operator=(const MmapAllocator&) = delete;

    MmapAllocator(MmapAllocator&& other) noexcept;

    MmapAllocator& operator=(MmapAllocator&& other) noexcept;

    [[nodiscard]] Result<void> init();

    /**
     * @brief Allocate Unified memory using Thrust
     *
     * @param size Number of bytes to allocate
     * @param alignment Memory alignment (ignored, UnifiedAllocator handles this)
     * @return Pointer to allocated memory or error
     */
    [[nodiscard]] Result<void*> allocate(usize size, usize alignment);

    [[nodiscard]] Result<void> deallocate(void* ptr, usize size) const;

    [[nodiscard]] constexpr DeviceType device_type() const noexcept {
        return DeviceType::Mmap;
    }

private:
    void reset_state() noexcept;

    void close_current() noexcept;

    std::string file_path_;

#if defined(_WIN32) || defined(_WIN64)
    HANDLE file_handle_{};
    HANDLE map_handle_{};
#else
    i32 fd_{-1};
#endif

    void* base_ptr_{nullptr};
    usize size_{0};
};

// Verify MmapAllocator satisfies Allocator concept
static_assert(Allocator<MmapAllocator>);


/// Unique Pointer with Custom Deleter

/**
 * @class AllocatorDeleter
 * @brief Custom deleter for unique_ptr that uses an Allocator
 *
 * This allows us to use std::unique_ptr with our custom allocators.
 */
template<Allocator A>
class AllocatorDeleter {
public:
    explicit AllocatorDeleter(A allocator, usize size) : allocator_(std::move(allocator)), size_(size) {}

    void operator()(void* ptr) const {
        if (ptr != nullptr) {
            auto result = allocator_.deallocate(ptr, size_);

            (void)result;
        }
    }

private:
    A allocator_;
    usize size_;
};

/**
 * @brief Unique pointer type that uses an allocator for deallocation
 */
template<Allocator A>
using AllocatorUniquePtr = std::unique_ptr<void, AllocatorDeleter<A>>;

/**
 * @brief Create a unique_ptr with memory allocated by an allocator
 *
 * @tparam A Allocator type
 * @param allocator The allocator to use
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement
 * @return Result containing unique_ptr to allocated memory, or error
 */
template<Allocator A>
[[nodiscard]] Result<AllocatorUniquePtr<A>> make_unique_alloc(A allocator, usize size, usize alignment) {
    auto alloc_result = allocator.allocate(size, alignment);

    if (!alloc_result) {
        return Err<AllocatorUniquePtr<A>>(std::move(alloc_result.error()));
    }

    void* ptr = alloc_result.value();

    AllocatorDeleter<A> deleter(std::move(allocator), size);

    return Ok(AllocatorUniquePtr<A>(ptr, std::move(deleter)));
}

}  // namespace pulse
