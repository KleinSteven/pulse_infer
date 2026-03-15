#include <cstdlib>
#include <format>
#include <limits>

#if defined PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif

#if defined(_WIN32) || defined(_WIN64)

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <filesystem>
#include <windows.h>
#else

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

#include "pulse/core/allocator.hpp"
#include "pulse/logging.hpp"

namespace pulse {

/// CPUAllocator
[[nodiscard]] Result<void*> CPUAllocator::allocate(usize size, usize alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return Err<void*>(ErrorCode::InvalidArgument, "Alignment must be power of 2");
    }

    if (size == 0) {
        return Err<void*>(ErrorCode::InvalidArgument, "Allocate size cannot be zero");
    }

    if (size > std::numeric_limits<usize>::max() - (alignment - 1)) {
        return Err<void*>(ErrorCode::OutOfMemory, "Aligned allocation size overflow");
    }

    // Align size to alignment boundary
    size = (size + alignment - 1) & ~(alignment - 1);

#if defined(_WIN32)
    void* ptr = _aligned_malloc(size, alignment);
#else
    void* ptr = std::aligned_alloc(alignment, size);
#endif

    if (ptr == nullptr) {
        auto str = std::format("Fali to allocate {} bytes", size);
        return Err<void*>(ErrorCode::OutOfMemory, str);
    }

    return Ok(ptr);
}

[[nodiscard]] Result<void> CPUAllocator::deallocate(void* ptr, [[maybe_unused]] usize size) const {
    if (ptr == nullptr) {
        return Err<void>(ErrorCode::NullPointer, "Cannot deallocate nullptr");
    }
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif

    return Ok();
}

[[nodiscard]] constexpr usize CPUAllocator::default_alignment() const noexcept {
    return kDefaultAlignment;
}

#ifdef PULSE_USE_CUDA

/// CUDAAllocator
[[nodiscard]] Result<void*> CUDAAllocator::allocate(usize size, [[maybe_unused]] usize alignment) {
    if (size == 0) {
        return Err<void*>(ErrorCode::InvalidArgument, "Cannot allocate zero bytes");
    }

    void* ptr = nullptr;

    cudaError_t err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
        auto err_str = cudaGetErrorString(err);
        auto str = std::format("cudaMalloc fali: {}", err_str);
        pulse::error("cudaMalloc fali: {}", err_str);
        return Err<void*>(ErrorCode::CudaOutOfMemory, str);
    }

    return Ok(ptr);
}

[[nodiscard]] Result<void> CUDAAllocator::deallocate(void* ptr, [[maybe_unused]] usize size) const {
    if (ptr == nullptr) {
        return Err<void>(ErrorCode::NullPointer, "Cannot deallocate nullptr");
    }

    cudaError_t err = cudaFree(ptr);

    if (err != cudaSuccess) {
        auto err_str = cudaGetErrorString(err);
        (void)cudaGetLastError();
        auto str = std::format("cudaFree fali: {}", err_str);
        pulse::error("cudaFree fali: {}", err_str);
        return Err<void>(ErrorCode::CudaError, str);
    }

    return Ok();
}

/// UnifiedAllocator
[[nodiscard]] Result<void*> UnifiedAllocator::allocate(usize size, [[maybe_unused]] usize alignment) {
    if (size == 0) {
        return Err<void*>(ErrorCode::InvalidArgument, "Cannot allocate zero bytes");
    }

    void* ptr = nullptr;

    cudaError_t err = cudaMallocManaged(&ptr, size);

    if (err != cudaSuccess) {
        auto err_str = cudaGetErrorString(err);
        auto str = std::format("cudaMallocManaged fali: {}", err_str);
        pulse::error("cudaMallocManaged fali: {}", err_str);
        return Err<void*>(ErrorCode::CudaError, str);
    }

    return Ok(ptr);
}

[[nodiscard]] Result<void> UnifiedAllocator::deallocate(void* ptr, [[maybe_unused]] usize size) const {
    if (ptr == nullptr) {
        return Err<void>(ErrorCode::NullPointer, "Cannot deallocate nullptr");
    }

    cudaError_t err = cudaFree(ptr);

    if (err != cudaSuccess) {
        auto err_str = cudaGetErrorString(err);
        (void)cudaGetLastError();
        auto str = std::format("cudaFree fali: {}", err_str);
        pulse::error("cudaFree fali: {}", err_str);
        return Err<void>(ErrorCode::CudaError, str);
    }

    return Ok();
}

#endif

/// MmapAllocator
MmapAllocator::MmapAllocator(const std::string& file_path) : file_path_(file_path) {}

void MmapAllocator::reset_state() noexcept {
#if defined(_WIN32) || defined(_WIN64)
    file_handle_ = nullptr;
    map_handle_ = nullptr;
#else
    fd_ = -1;
#endif
    base_ptr_ = nullptr;
    size_ = 0;
}

void MmapAllocator::close_current() noexcept {
#if defined(_WIN32) || defined(_WIN64)
    if (base_ptr_ != nullptr) {
        UnmapViewOfFile(base_ptr_);
    }

    if (map_handle_ != nullptr) {
        CloseHandle(map_handle_);
    }

    if (file_handle_ != nullptr && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
    }
#else
    if (base_ptr_ != nullptr) {
        munmap(base_ptr_, size_);
    }

    if (fd_ != -1) {
        close(fd_);
    }
#endif

    reset_state();
}

MmapAllocator::MmapAllocator(MmapAllocator&& other) noexcept
    : file_path_(std::move(other.file_path_)),
#if defined(_WIN32) || defined(_WIN64)
      file_handle_(other.file_handle_),
      map_handle_(other.map_handle_),
#else
      fd_(other.fd_),
#endif
      base_ptr_(other.base_ptr_),
      size_(other.size_) {
    other.reset_state();
}

MmapAllocator& MmapAllocator::operator=(MmapAllocator&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    close_current();

    file_path_ = std::move(other.file_path_);
    base_ptr_ = other.base_ptr_;
    size_ = other.size_;

#if defined(_WIN32) || defined(_WIN64)
    file_handle_ = other.file_handle_;
    map_handle_ = other.map_handle_;
#else
    fd_ = other.fd_;
#endif

    other.reset_state();

    return *this;
}

[[nodiscard]] Result<void> MmapAllocator::init() {
#if defined(_WIN32) || defined(_WIN64)
    if (base_ptr_ != nullptr || map_handle_ != nullptr ||
        (file_handle_ != nullptr && file_handle_ != INVALID_HANDLE_VALUE)) {
#else
    if (base_ptr_ != nullptr || fd_ != -1) {
#endif
        return Err<void>(ErrorCode::Unknown, "MmapAllocator is already initialized");
    }

#if defined(_WIN32) || defined(_WIN64)
    std::wstring wide_path = std::filesystem::path(file_path_).wstring();

    file_handle_ = CreateFileW(wide_path.c_str(),
                               GENERIC_READ,
                               FILE_SHARE_READ,
                               NULL,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                               NULL);

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        file_handle_ = nullptr;
        return Err<void>(ErrorCode::OpenFileError, "Failed to open file for mmap");
    }

    LARGE_INTEGER li_size;

    if (!GetFileSizeEx(file_handle_, &li_size)) {
        close_current();
        return Err<void>(ErrorCode::GetFileSizeError, "Failed to get file size");
    }

    size_ = static_cast<size_t>(li_size.QuadPart);

    map_handle_ = CreateFileMappingW(file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);

    if (map_handle_ == NULL) {
        map_handle_ = nullptr;
        close_current();
        return Err<void>(ErrorCode::MmapError, "CreateFileMapping failed");
    }

    base_ptr_ = MapViewOfFile(map_handle_, FILE_MAP_READ, 0, 0, 0);

    if (base_ptr_ == NULL) {
        close_current();
        return Err<void>(ErrorCode::MmapError, "MapViewOfFile failed");
    }

#else
    fd_ = open(file_path_.c_str(), O_RDONLY);

    if (fd_ == -1) {
        return Err<void>(ErrorCode::OpenFileError, "Failed to open file for mmap");
    }

    struct stat sb;

    if (fstat(fd_, &sb) == -1) {
        close_current();
        return Err<void>(ErrorCode::GetFileSizeError, "Failed to get file size");
    }

    size_ = static_cast<usize>(sb.st_size);

    base_ptr_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd_, 0);
    if (base_ptr_ == MAP_FAILED) {
        base_ptr_ = nullptr;
        close_current();
        return Err<void>(ErrorCode::MmapError, "Mmap failed");
    }

    // Transparent Huge Pages
    if (madvise(base_ptr_, size_, MADV_HUGEPAGE) == -1) {
        pulse::warn("Transparent Huge Pages failed");
    }
#endif

    return Ok();
}

MmapAllocator::~MmapAllocator() {
    close_current();
}

[[nodiscard]] Result<void*> MmapAllocator::allocate([[maybe_unused]] usize size,
                                                    [[maybe_unused]] usize alignment) {
    // MmapAllocator does not support dynamic allocation, it only manages the mapping of the entire file.
    return Err<void*>(ErrorCode::NotImplemented, "MmapAllocator does not support dynamic allocation");
}

[[nodiscard]] Result<void> MmapAllocator::deallocate([[maybe_unused]] void* ptr,
                                                     [[maybe_unused]] usize size) const {
    // MmapAllocator does not support dynamic allocation, it only manages the mapping of the entire file.
    return Err<void>(ErrorCode::NotImplemented, "MmapAllocator does not support dynamic allocation");
}

}  // namespace pulse
