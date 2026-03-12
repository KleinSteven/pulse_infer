#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>

#include "pulse/core/buffer.hpp"

using namespace pulse;

namespace {

template<typename SpanT, typename T, usize N>
void expect_span_eq(std::span<SpanT> span, const std::array<T, N>& expected) {
    static_assert(std::is_same_v<std::remove_const_t<SpanT>, T>);

    ASSERT_EQ(span.size(), expected.size());

    for (usize i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(span[i], expected[i]);
    }
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif
void self_move_assign(Buffer& buffer) {
    Buffer* alias = &buffer;
    *alias = std::move(*alias);
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

}  // namespace

TEST(BufferTest, DefaultConstructedBufferIsEmpty) {
    Buffer buffer;

    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(static_cast<bool>(buffer));
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.alignment(), 64);
    EXPECT_EQ(buffer.device(), DeviceType::CPU);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.as_span<std::byte>().size(), 0);
    EXPECT_EQ(buffer.num_elements<std::uint32_t>(), 0);
}

TEST(BufferTest, BasicCreation) {
    auto result = Buffer::create(1024, DeviceType::CPU);

    ASSERT_TRUE(result.is_ok());
    Buffer buffer = std::move(result.value());

    EXPECT_FALSE(buffer.empty());
    EXPECT_TRUE(static_cast<bool>(buffer));
    EXPECT_EQ(buffer.size(), 1024);
    EXPECT_EQ(buffer.alignment(), 64);
    EXPECT_EQ(buffer.device(), DeviceType::CPU);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST(BufferTest, CreationPreservesCustomAlignment) {
    constexpr usize kAlignment = 256;
    auto result = Buffer::create(512, DeviceType::CPU, kAlignment);

    ASSERT_TRUE(result.is_ok());
    Buffer buffer = std::move(result.value());

    EXPECT_EQ(buffer.alignment(), kAlignment);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data()) % kAlignment, 0U);
}

TEST(BufferTest, ZeroSizeCreation) {
    auto result = Buffer::create(0, DeviceType::CPU);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(BufferTest, UnsupportedDeviceCreationFails) {
    auto result = Buffer::create(128, DeviceType::Mmap);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(BufferTest, DataViewsExposeTypedAccessForCpuBuffers) {
    constexpr std::array<std::uint32_t, 4> kValues{1U, 5U, 9U, 13U};
    auto result = Buffer::create(sizeof(kValues), DeviceType::CPU);

    ASSERT_TRUE(result.is_ok());
    Buffer buffer = std::move(result.value());

    auto span = buffer.as_span<std::uint32_t>();
    ASSERT_EQ(span.size(), kValues.size());

    for (usize i = 0; i < kValues.size(); ++i) {
        span[i] = kValues[i];
    }

    EXPECT_EQ(buffer.data_as<std::uint32_t>(), span.data());
    EXPECT_EQ(buffer.num_elements<std::uint32_t>(), kValues.size());

    const Buffer& const_buffer = buffer;
    expect_span_eq(const_buffer.as_span<std::uint32_t>(), kValues);
}

TEST(BufferTest, TypedViewsFloorElementCountWhenSizeIsNotAMultiple) {
    auto result = Buffer::create(10, DeviceType::CPU);

    ASSERT_TRUE(result.is_ok());
    Buffer buffer = std::move(result.value());

    EXPECT_EQ(buffer.num_elements<std::uint32_t>(), 2);
    EXPECT_EQ(buffer.as_span<std::uint32_t>().size(), 2);

    const Buffer& const_buffer = buffer;
    EXPECT_EQ(const_buffer.as_span<std::uint32_t>().size(), 2);
}

TEST(BufferTest, MoveConstructorTransfersOwnership) {
    auto result = Buffer::create(128, DeviceType::CPU, 128);

    ASSERT_TRUE(result.is_ok());
    Buffer source = std::move(result.value());
    void* original_ptr = source.data();

    Buffer moved(std::move(source));

    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_EQ(moved.size(), 128);
    EXPECT_EQ(moved.alignment(), 128);
    EXPECT_EQ(moved.device(), DeviceType::CPU);
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.size(), 0);
}

TEST(BufferTest, MoveAssignmentTransfersOwnershipAndReleasesPreviousStorage) {
    auto src_result = Buffer::create(256, DeviceType::CPU, 128);
    auto dst_result = Buffer::create(64, DeviceType::CPU);

    ASSERT_TRUE(src_result.is_ok());
    ASSERT_TRUE(dst_result.is_ok());

    Buffer source = std::move(src_result.value());
    Buffer destination = std::move(dst_result.value());
    void* source_ptr = source.data();

    destination = std::move(source);

    EXPECT_EQ(destination.data(), source_ptr);
    EXPECT_EQ(destination.size(), 256);
    EXPECT_EQ(destination.alignment(), 128);
    EXPECT_EQ(destination.device(), DeviceType::CPU);
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.size(), 0);
}

TEST(BufferTest, SelfMoveAssignmentLeavesBufferValid) {
    auto result = Buffer::create(96, DeviceType::CPU);

    ASSERT_TRUE(result.is_ok());
    Buffer buffer = std::move(result.value());
    void* original_ptr = buffer.data();

    self_move_assign(buffer);

    EXPECT_EQ(buffer.data(), original_ptr);
    EXPECT_EQ(buffer.size(), 96);
    EXPECT_FALSE(buffer.empty());
}

TEST(BufferTest, CopyFromCopiesCpuContents) {
    constexpr std::array<std::uint32_t, 4> kValues{10U, 20U, 30U, 40U};
    auto src_result = Buffer::create(sizeof(kValues), DeviceType::CPU);
    auto dst_result = Buffer::create(sizeof(kValues), DeviceType::CPU);

    ASSERT_TRUE(src_result.is_ok());
    ASSERT_TRUE(dst_result.is_ok());

    Buffer source = std::move(src_result.value());
    Buffer destination = std::move(dst_result.value());

    auto src_span = source.as_span<std::uint32_t>();
    ASSERT_EQ(src_span.size(), kValues.size());

    for (usize i = 0; i < kValues.size(); ++i) {
        src_span[i] = kValues[i];
    }

    auto copy_result = destination.copy_from(source);
    ASSERT_TRUE(copy_result.is_ok()) << copy_result.error().message();

    expect_span_eq(destination.as_span<std::uint32_t>(), kValues);
}

TEST(BufferTest, CopyFromRejectsSizeMismatch) {
    auto src_result = Buffer::create(64, DeviceType::CPU);
    auto dst_result = Buffer::create(32, DeviceType::CPU);

    ASSERT_TRUE(src_result.is_ok());
    ASSERT_TRUE(dst_result.is_ok());

    Buffer source = std::move(src_result.value());
    Buffer destination = std::move(dst_result.value());

    auto copy_result = destination.copy_from(source);

    EXPECT_TRUE(copy_result.is_err());
    EXPECT_EQ(copy_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(BufferTest, CopyFromEmptySourceIntoEmptyDestinationSucceeds) {
    Buffer source;
    Buffer destination;

    auto copy_result = destination.copy_from(source);

    EXPECT_TRUE(copy_result.is_ok());
}

TEST(BufferTest, CloneDuplicatesContentAndMetadata) {
    constexpr std::array<std::uint32_t, 4> kValues{7U, 14U, 21U, 28U};
    auto result = Buffer::create(sizeof(kValues), DeviceType::CPU, 128);

    ASSERT_TRUE(result.is_ok());
    Buffer original = std::move(result.value());

    auto span = original.as_span<std::uint32_t>();
    ASSERT_EQ(span.size(), kValues.size());

    for (usize i = 0; i < kValues.size(); ++i) {
        span[i] = kValues[i];
    }

    auto clone_result = original.clone();
    ASSERT_TRUE(clone_result.is_ok()) << clone_result.error().message();

    Buffer clone = std::move(clone_result.value());

    EXPECT_NE(clone.data(), original.data());
    EXPECT_EQ(clone.size(), original.size());
    EXPECT_EQ(clone.alignment(), original.alignment());
    EXPECT_EQ(clone.device(), original.device());
    expect_span_eq(clone.as_span<std::uint32_t>(), kValues);

    original.as_span<std::uint32_t>()[0] = 999U;
    EXPECT_EQ(clone.as_span<std::uint32_t>()[0], kValues[0]);
}

TEST(BufferTest, CloneEmptyBufferReturnsEmptyClone) {
    Buffer buffer;

    auto clone_result = buffer.clone();

    ASSERT_TRUE(clone_result.is_ok());
    Buffer clone = std::move(clone_result.value());
    EXPECT_TRUE(clone.empty());
    EXPECT_EQ(clone.data(), nullptr);
    EXPECT_EQ(clone.size(), 0);
}

TEST(BufferTest, ZeroClearsCpuBuffer) {
    auto result = Buffer::create(5 * sizeof(std::uint32_t), DeviceType::CPU);

    ASSERT_TRUE(result.is_ok());
    Buffer buffer = std::move(result.value());

    auto span = buffer.as_span<std::uint32_t>();
    ASSERT_EQ(span.size(), 5U);

    for (usize i = 0; i < span.size(); ++i) {
        span[i] = static_cast<std::uint32_t>(i + 1);
    }

    auto zero_result = buffer.zero();
    ASSERT_TRUE(zero_result.is_ok()) << zero_result.error().message();

    for (auto value : buffer.as_span<std::uint32_t>()) {
        EXPECT_EQ(value, 0U);
    }
}

TEST(BufferTest, ZeroEmptyBufferSucceeds) {
    Buffer buffer;

    auto zero_result = buffer.zero();

    EXPECT_TRUE(zero_result.is_ok());
}

#ifdef PULSE_USE_CUDA

TEST(BufferTest, CudaBufferCreationAndViews) {
    auto result = Buffer::create(256, DeviceType::CUDA);

    if (!result.is_ok()) {
        GTEST_SKIP() << result.error().message();
    }

    Buffer buffer = std::move(result.value());

    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size(), 256);
    EXPECT_EQ(buffer.device(), DeviceType::CUDA);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_TRUE(buffer.as_span<std::uint32_t>().empty());

    const Buffer& const_buffer = buffer;
    EXPECT_TRUE(const_buffer.as_span<std::uint32_t>().empty());
}

TEST(BufferTest, CopyFromSupportsCpuCudaRoundTrip) {
    constexpr std::array<std::uint32_t, 4> kValues{3U, 6U, 9U, 12U};
    auto host_src_result = Buffer::create(sizeof(kValues), DeviceType::CPU);
    auto device_result = Buffer::create(sizeof(kValues), DeviceType::CUDA);
    auto host_dst_result = Buffer::create(sizeof(kValues), DeviceType::CPU);

    ASSERT_TRUE(host_src_result.is_ok());
    if (!device_result.is_ok()) {
        GTEST_SKIP() << device_result.error().message();
    }

    ASSERT_TRUE(host_dst_result.is_ok());

    Buffer host_src = std::move(host_src_result.value());
    Buffer device = std::move(device_result.value());
    Buffer host_dst = std::move(host_dst_result.value());

    auto host_span = host_src.as_span<std::uint32_t>();
    ASSERT_EQ(host_span.size(), kValues.size());

    for (usize i = 0; i < kValues.size(); ++i) {
        host_span[i] = kValues[i];
    }

    auto h2d_result = device.copy_from(host_src);
    ASSERT_TRUE(h2d_result.is_ok()) << h2d_result.error().message();

    auto d2h_result = host_dst.copy_from(device);
    ASSERT_TRUE(d2h_result.is_ok()) << d2h_result.error().message();

    expect_span_eq(host_dst.as_span<std::uint32_t>(), kValues);
}

TEST(BufferTest, CloneDuplicatesCudaBufferContent) {
    constexpr std::array<std::uint32_t, 4> kValues{11U, 22U, 33U, 44U};
    auto host_result = Buffer::create(sizeof(kValues), DeviceType::CPU);
    auto device_result = Buffer::create(sizeof(kValues), DeviceType::CUDA);
    auto clone_host_result = Buffer::create(sizeof(kValues), DeviceType::CPU);

    ASSERT_TRUE(host_result.is_ok());
    if (!device_result.is_ok()) {
        GTEST_SKIP() << device_result.error().message();
    }

    ASSERT_TRUE(clone_host_result.is_ok());

    Buffer host = std::move(host_result.value());
    Buffer device = std::move(device_result.value());
    Buffer clone_host = std::move(clone_host_result.value());

    auto host_span = host.as_span<std::uint32_t>();
    ASSERT_EQ(host_span.size(), kValues.size());

    for (usize i = 0; i < kValues.size(); ++i) {
        host_span[i] = kValues[i];
    }

    auto upload_result = device.copy_from(host);
    ASSERT_TRUE(upload_result.is_ok()) << upload_result.error().message();

    auto clone_result = device.clone();
    ASSERT_TRUE(clone_result.is_ok()) << clone_result.error().message();
    Buffer clone = std::move(clone_result.value());

    EXPECT_NE(clone.data(), device.data());
    EXPECT_EQ(clone.size(), device.size());
    EXPECT_EQ(clone.device(), DeviceType::CUDA);

    auto download_result = clone_host.copy_from(clone);
    ASSERT_TRUE(download_result.is_ok()) << download_result.error().message();

    expect_span_eq(clone_host.as_span<std::uint32_t>(), kValues);
}

TEST(BufferTest, ZeroClearsCudaBuffer) {
    constexpr std::array<std::uint32_t, 4> kValues{8U, 16U, 24U, 32U};
    auto host_result = Buffer::create(sizeof(kValues), DeviceType::CPU);
    auto device_result = Buffer::create(sizeof(kValues), DeviceType::CUDA);
    auto roundtrip_result = Buffer::create(sizeof(kValues), DeviceType::CPU);

    ASSERT_TRUE(host_result.is_ok());
    if (!device_result.is_ok()) {
        GTEST_SKIP() << device_result.error().message();
    }

    ASSERT_TRUE(roundtrip_result.is_ok());

    Buffer host = std::move(host_result.value());
    Buffer device = std::move(device_result.value());
    Buffer roundtrip = std::move(roundtrip_result.value());

    auto host_span = host.as_span<std::uint32_t>();
    ASSERT_EQ(host_span.size(), kValues.size());

    for (usize i = 0; i < kValues.size(); ++i) {
        host_span[i] = kValues[i];
    }

    auto upload_result = device.copy_from(host);
    ASSERT_TRUE(upload_result.is_ok()) << upload_result.error().message();

    auto zero_result = device.zero();
    ASSERT_TRUE(zero_result.is_ok()) << zero_result.error().message();

    auto download_result = roundtrip.copy_from(device);
    ASSERT_TRUE(download_result.is_ok()) << download_result.error().message();

    for (auto value : roundtrip.as_span<std::uint32_t>()) {
        EXPECT_EQ(value, 0U);
    }
}

#endif
