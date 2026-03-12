#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#ifdef PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>

#include "pulse/core/allocator.hpp"

using namespace pulse;

namespace {

std::filesystem::path make_temp_path(const std::string& suffix) {
    const auto unique_id = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    return std::filesystem::path(::testing::TempDir()) / ("pulse_allocator_" + unique_id + suffix);
}

std::filesystem::path write_temp_file(const std::string& contents) {
    auto path = make_temp_path(".bin");
    std::ofstream stream(path, std::ios::binary);
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.close();
    return path;
}

#ifdef PULSE_USE_CUDA

template<typename AllocatorT>
struct CudaAllocatorTraits;

template<>
struct CudaAllocatorTraits<CUDAAllocator> {
    [[maybe_unused]] static constexpr DeviceType kDeviceType = DeviceType::CUDA;
    [[maybe_unused]] static constexpr cudaMemoryType kMemoryType = cudaMemoryTypeDevice;
};

template<>
struct CudaAllocatorTraits<UnifiedAllocator> {
    [[maybe_unused]] static constexpr DeviceType kDeviceType = DeviceType::Unified;
    [[maybe_unused]] static constexpr cudaMemoryType kMemoryType = cudaMemoryTypeManaged;
};

template<typename AllocatorT>
void expect_cuda_pointer_type(void* ptr) {
    cudaPointerAttributes attributes{};
    const auto err = cudaPointerGetAttributes(&attributes, ptr);

    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    EXPECT_EQ(attributes.type, CudaAllocatorTraits<AllocatorT>::kMemoryType);
}

template<typename AllocatorT>
class CudaBackedAllocatorTest : public ::testing::Test {};

using CudaBackedAllocatorTypes = ::testing::Types<CUDAAllocator, UnifiedAllocator>;
TYPED_TEST_SUITE(CudaBackedAllocatorTest, CudaBackedAllocatorTypes);

#endif

}  // namespace

// ============================================================================
// CPUAllocator Tests
// ============================================================================

TEST(CPUAllocatorTest, BasicAllocation) {
    CPUAllocator allocator;
    auto result = allocator.allocate(1024);

    ASSERT_TRUE(result.is_ok());
    void* ptr = result.value();
    EXPECT_NE(ptr, nullptr);

    // Check alignment
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % CPUAllocator::kDefaultAlignment, 0);

    // Deallocate
    auto dealloc_result = allocator.deallocate(ptr, 1024);
    EXPECT_TRUE(dealloc_result.is_ok());
}

TEST(CPUAllocatorTest, CustomAlignment) {
    CPUAllocator allocator;
    usize alignment = 256;
    auto result = allocator.allocate(1024, alignment);

    ASSERT_TRUE(result.is_ok());
    void* ptr = result.value();

    auto addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % alignment, 0);

    auto dealloc_result = allocator.deallocate(ptr, 1024);
    EXPECT_TRUE(dealloc_result.is_ok());
}

TEST(CPUAllocatorTest, ZeroSizeAllocation) {
    CPUAllocator allocator;
    auto result = allocator.allocate(0);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(CPUAllocatorTest, InvalidAlignment) {
    CPUAllocator allocator;
    // Alignment must be power of 2
    auto result = allocator.allocate(1024, 3);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(CPUAllocatorTest, RejectsAlignedSizeOverflow) {
    CPUAllocator allocator;
    constexpr usize alignment = CPUAllocator::kDefaultAlignment;
    const usize size = std::numeric_limits<usize>::max() - (alignment - 2);

    auto result = allocator.allocate(size, alignment);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::OutOfMemory);
}

TEST(CPUAllocatorTest, NullPointerDeallocation) {
    CPUAllocator allocator;
    auto result = allocator.deallocate(nullptr, 0);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::NullPointer);
}

TEST(CPUAllocatorTest, DeviceType) {
    CPUAllocator allocator;
    EXPECT_EQ(allocator.device_type(), DeviceType::CPU);
}

TEST(CPUAllocatorTest, LargeAllocation) {
    CPUAllocator allocator;
    usize size = 100 * 1024 * 1024;  // 100 MB
    auto result = allocator.allocate(size);

    ASSERT_TRUE(result.is_ok());
    void* ptr = result.value();
    EXPECT_NE(ptr, nullptr);

    // Write to memory to ensure it's actually allocated
    std::memset(ptr, 0, size);

    auto dealloc_result = allocator.deallocate(ptr, size);
    EXPECT_TRUE(dealloc_result.is_ok());
}

TEST(CPUAllocatorTest, MultipleAllocations) {
    CPUAllocator allocator;
    std::vector<void*> ptrs;

    // Allocate multiple blocks
    for (int i = 0; i < 10; ++i) {
        auto result = allocator.allocate(1024);
        ASSERT_TRUE(result.is_ok());
        ptrs.push_back(result.value());
    }

    // All pointers should be different
    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            EXPECT_NE(ptrs[i], ptrs[j]);
        }
    }

    // Deallocate all
    for (void* ptr : ptrs) {
        auto result = allocator.deallocate(ptr, 1024);
        EXPECT_TRUE(result.is_ok());
    }
}

#ifdef PULSE_USE_CUDA

// ============================================================================
// CUDA-backed Allocator Tests
// ============================================================================

TYPED_TEST(CudaBackedAllocatorTest, ZeroSizeAllocation) {
    TypeParam allocator;
    auto result = allocator.allocate(0, 64);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TYPED_TEST(CudaBackedAllocatorTest, NullPointerDeallocation) {
    TypeParam allocator;
    auto result = allocator.deallocate(nullptr, 0);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::NullPointer);
}

TYPED_TEST(CudaBackedAllocatorTest, DeviceType) {
    TypeParam allocator;
    EXPECT_EQ(allocator.device_type(), CudaAllocatorTraits<TypeParam>::kDeviceType);
}

TYPED_TEST(CudaBackedAllocatorTest, AlignmentArgumentIsIgnored) {
    TypeParam allocator;
    auto result = allocator.allocate(256, 3);

    ASSERT_TRUE(result.is_ok()) << result.error().message();
    void* ptr = result.value();
    ASSERT_NE(ptr, nullptr);
    expect_cuda_pointer_type<TypeParam>(ptr);

    auto dealloc_result = allocator.deallocate(ptr, 256);
    EXPECT_TRUE(dealloc_result.is_ok()) << dealloc_result.error().message();
}

TYPED_TEST(CudaBackedAllocatorTest, BasicAllocation) {
    TypeParam allocator;
    constexpr usize size = 4096;
    auto result = allocator.allocate(size, 128);

    ASSERT_TRUE(result.is_ok()) << result.error().message();
    void* ptr = result.value();
    ASSERT_NE(ptr, nullptr);
    expect_cuda_pointer_type<TypeParam>(ptr);

    const auto memset_result = cudaMemset(ptr, 0x2A, size);
    EXPECT_EQ(memset_result, cudaSuccess) << cudaGetErrorString(memset_result);

    const auto sync_result = cudaDeviceSynchronize();
    EXPECT_EQ(sync_result, cudaSuccess) << cudaGetErrorString(sync_result);

    auto dealloc_result = allocator.deallocate(ptr, size);
    EXPECT_TRUE(dealloc_result.is_ok()) << dealloc_result.error().message();
}

TYPED_TEST(CudaBackedAllocatorTest, DeallocateRejectsHostPointer) {
    TypeParam allocator;
    int host_value = 7;

    auto result = allocator.deallocate(&host_value, sizeof(host_value));
    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::CudaError);
}

TYPED_TEST(CudaBackedAllocatorTest, MultipleAllocations) {
    TypeParam allocator;
    std::vector<void*> ptrs;

    for (int i = 0; i < 4; ++i) {
        auto result = allocator.allocate(512, 1);
        ASSERT_TRUE(result.is_ok()) << result.error().message();
        ptrs.push_back(result.value());
    }

    for (size_t i = 0; i < ptrs.size(); ++i) {
        expect_cuda_pointer_type<TypeParam>(ptrs[i]);
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            EXPECT_NE(ptrs[i], ptrs[j]);
        }
    }

    for (void* ptr : ptrs) {
        auto result = allocator.deallocate(ptr, 512);
        EXPECT_TRUE(result.is_ok()) << result.error().message();
    }
}

TYPED_TEST(CudaBackedAllocatorTest, MakeUniqueAllocBasicUsage) {
    TypeParam allocator;
    constexpr usize size = 1024;
    auto result = make_unique_alloc(allocator, size, 0);

    ASSERT_TRUE(result.is_ok()) << result.error().message();
    auto ptr = std::move(result.value());
    ASSERT_NE(ptr.get(), nullptr);
    expect_cuda_pointer_type<TypeParam>(ptr.get());

    const auto memset_result = cudaMemset(ptr.get(), 0x11, size);
    EXPECT_EQ(memset_result, cudaSuccess) << cudaGetErrorString(memset_result);
}

TYPED_TEST(CudaBackedAllocatorTest, MakeUniqueAllocPropagatesAllocationFailure) {
    TypeParam allocator;
    auto result = make_unique_alloc(allocator, 0, 64);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

#endif

// ============================================================================
// MmapAllocator Tests
// ============================================================================

TEST(MmapAllocatorTest, DeviceType) {
    MmapAllocator allocator("unused-path");
    EXPECT_EQ(allocator.device_type(), DeviceType::Mmap);
}

TEST(MmapAllocatorTest, InitMissingFile) {
    const auto path = make_temp_path(".missing");
    MmapAllocator allocator(path.string());

    auto result = allocator.init();

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::OpenFileError);
}

TEST(MmapAllocatorTest, InitEmptyFile) {
    const auto path = make_temp_path(".empty");
    std::ofstream(path, std::ios::binary).close();

    MmapAllocator allocator(path.string());
    auto result = allocator.init();

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::MmapError);

    std::filesystem::remove(path);
}

TEST(MmapAllocatorTest, InitNonEmptyFile) {
    const auto path = write_temp_file("pulse allocator mmap test payload");
    {
        MmapAllocator allocator(path.string());
        auto result = allocator.init();
        ASSERT_TRUE(result.is_ok()) << result.error().message();
    }

    std::filesystem::remove(path);
}

TEST(MmapAllocatorTest, MoveConstructBeforeInitPreservesEmptyState) {
    const auto path = write_temp_file("pulse allocator mmap move test payload");
    {
        MmapAllocator source(path.string());
        MmapAllocator moved(std::move(source));

        auto result = moved.init();
        ASSERT_TRUE(result.is_ok()) << result.error().message();
    }

    std::filesystem::remove(path);
}

TEST(MmapAllocatorTest, FailedInitCanBeFollowedByMoveAssignment) {
    const auto missing_path = make_temp_path(".missing");
    const auto valid_path = write_temp_file("pulse allocator mmap move assignment payload");
    {
        MmapAllocator allocator(missing_path.string());
        auto missing_result = allocator.init();
        ASSERT_TRUE(missing_result.is_err());

        allocator = MmapAllocator(valid_path.string());
        auto valid_result = allocator.init();
        ASSERT_TRUE(valid_result.is_ok()) << valid_result.error().message();
    }

    std::filesystem::remove(valid_path);
}

TEST(MmapAllocatorTest, RejectsRepeatedInit) {
    const auto path = write_temp_file("pulse allocator repeated init payload");
    {
        MmapAllocator allocator(path.string());
        auto first_result = allocator.init();
        ASSERT_TRUE(first_result.is_ok()) << first_result.error().message();

        auto second_result = allocator.init();
        EXPECT_TRUE(second_result.is_err());
        EXPECT_EQ(second_result.error().code(), ErrorCode::Unknown);
    }

    std::filesystem::remove(path);
}

#ifdef PULSE_USE_CUDA
#undef SKIP_IF_NO_CUDA_DEVICE
#endif

TEST(MmapAllocatorTest, AllocateIsNotImplemented) {
    MmapAllocator allocator("unused-path");
    auto result = allocator.allocate(1024, 64);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::NotImplemented);
}

TEST(MmapAllocatorTest, DeallocateIsNotImplemented) {
    MmapAllocator allocator("unused-path");
    int value = 0;
    auto result = allocator.deallocate(&value, sizeof(value));

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::NotImplemented);
}

TEST(MmapAllocatorTest, IsMoveOnly) {
    EXPECT_FALSE((std::is_copy_constructible_v<MmapAllocator>));
    EXPECT_FALSE((std::is_copy_assignable_v<MmapAllocator>));
    EXPECT_TRUE((std::is_move_constructible_v<MmapAllocator>));
    EXPECT_TRUE((std::is_move_assignable_v<MmapAllocator>));
}

// ============================================================================
// Allocator Concept Tests
// ============================================================================

// Verify CPUAllocator satisfies Allocator concept at compile time
static_assert(Allocator<CPUAllocator>, "CPUAllocator should satisfy Allocator concept");
static_assert(Allocator<MmapAllocator>, "MmapAllocator should satisfy Allocator concept");
static_assert(!std::is_copy_constructible_v<MmapAllocator>, "MmapAllocator should not be copy constructible");
static_assert(!std::is_copy_assignable_v<MmapAllocator>, "MmapAllocator should not be copy assignable");
static_assert(std::is_move_constructible_v<MmapAllocator>, "MmapAllocator should be move constructible");
static_assert(std::is_move_assignable_v<MmapAllocator>, "MmapAllocator should be move assignable");

#ifdef PULSE_USE_CUDA
static_assert(Allocator<CUDAAllocator>, "CUDAAllocator should satisfy Allocator concept");
static_assert(Allocator<UnifiedAllocator>, "UnifiedAllocator should satisfy Allocator concept");
#endif

// Test concept requirements
TEST(AllocatorConceptTest, CPUAllocatorSatisfiesConcept) {
    CPUAllocator alloc;

    // Test allocate signature
    Result<void*> alloc_result = alloc.allocate(1024, 64);
    EXPECT_TRUE(alloc_result.is_ok() || alloc_result.is_err());

    if (alloc_result) {
        // Test deallocate signature
        Result<void> dealloc_result = alloc.deallocate(alloc_result.value(), 1024);
        EXPECT_TRUE(dealloc_result.is_ok());
    }

    // Test device_type signature
    DeviceType device = alloc.device_type();
    EXPECT_EQ(device, DeviceType::CPU);
}

TEST(AllocatorConceptTest, MmapAllocatorSatisfiesConcept) {
    MmapAllocator alloc("unused-path");

    Result<void*> alloc_result = alloc.allocate(1024, 64);
    EXPECT_TRUE(alloc_result.is_err());
    EXPECT_EQ(alloc_result.error().code(), ErrorCode::NotImplemented);

    int value = 0;
    Result<void> dealloc_result = alloc.deallocate(&value, sizeof(value));
    EXPECT_TRUE(dealloc_result.is_err());
    EXPECT_EQ(dealloc_result.error().code(), ErrorCode::NotImplemented);

    DeviceType device = alloc.device_type();
    EXPECT_EQ(device, DeviceType::Mmap);
}

#ifdef PULSE_USE_CUDA

TEST(AllocatorConceptTest, CUDAAllocatorSatisfiesConcept) {
    CUDAAllocator alloc;
    DeviceType device = alloc.device_type();
    EXPECT_EQ(device, DeviceType::CUDA);
}

TEST(AllocatorConceptTest, UnifiedAllocatorSatisfiesConcept) {
    UnifiedAllocator alloc;
    DeviceType device = alloc.device_type();
    EXPECT_EQ(device, DeviceType::Unified);
}

#endif

// ============================================================================
// UniquePtr with Allocator Tests
// ============================================================================

TEST(AllocatorUniquePtrTest, BasicUsage) {
    CPUAllocator allocator;
    usize size = 1024;

    auto result = make_unique_alloc(allocator, size, 64);
    ASSERT_TRUE(result.is_ok());

    auto ptr = std::move(result.value());
    EXPECT_NE(ptr.get(), nullptr);

    // Write to memory
    std::memset(ptr.get(), 42, size);

    // ptr will be automatically deallocated when it goes out of scope
}

TEST(AllocatorUniquePtrTest, MoveSemantics) {
    CPUAllocator allocator;
    usize size = 1024;

    auto result = make_unique_alloc(allocator, size, 64);
    ASSERT_TRUE(result.is_ok());

    auto ptr1 = std::move(result.value());
    void* raw_ptr = ptr1.get();

    // Move to another unique_ptr
    auto ptr2 = std::move(ptr1);

    EXPECT_EQ(ptr1.get(), nullptr);  // ptr1 is now null
    EXPECT_EQ(ptr2.get(), raw_ptr);  // ptr2 owns the memory
}
