#ifdef PULSE_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <array>
#include <string>

#include <gtest/gtest.h>

#include "pulse/core/tensor.hpp"

using namespace pulse;

namespace {

template<typename T, usize N>
void expect_array_eq(const T* actual, const std::array<T, N>& expected) {
    ASSERT_NE(actual, nullptr);

    for (usize i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i], expected[i]);
    }
}

}  // namespace

TEST(TensorTest, DefaultConstructedTensorIsEmpty) {
    Tensor tensor;

    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.size(), 0);
    EXPECT_EQ(tensor.byte_size(), 0);
    EXPECT_EQ(tensor.ndim(), 0);
    EXPECT_EQ(tensor.dtype(), DataType::Float32);
    EXPECT_EQ(tensor.device(), DeviceType::CPU);
    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_TRUE(tensor.dims().empty());
    EXPECT_TRUE(tensor.stride().empty());
}

TEST(TensorTest, Create) {
    auto result = Tensor::create({2, 3}, DataType::Float32);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());
    EXPECT_EQ(tensor.dims(), std::vector<i32>({2, 3}));
    EXPECT_EQ(tensor.dtype(), DataType::Float32);
    EXPECT_EQ(tensor.device(), DeviceType::CPU);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.byte_size(), 6 * sizeof(f32));
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_FALSE(tensor.empty());
    EXPECT_NE(tensor.data(), nullptr);
}

TEST(TensorTest, CreateScalarTensorWithEmptyDims) {
    auto result = Tensor::create({}, DataType::Int32);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    EXPECT_FALSE(tensor.empty());
    EXPECT_EQ(tensor.ndim(), 0);
    EXPECT_TRUE(tensor.dims().empty());
    EXPECT_EQ(tensor.size(), 1);
    EXPECT_EQ(tensor.byte_size(), sizeof(i32));
    EXPECT_TRUE(tensor.stride().empty());
}

TEST(TensorTest, CreateRejectsZeroElementTensor) {
    auto result = Tensor::create({2, 0, 3}, DataType::Float32);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorTest, DimsAndDimReportShapeInformation) {
    auto result = Tensor::create({4, 5, 6}, DataType::UInt16);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    EXPECT_EQ(tensor.ndim(), 3);
    EXPECT_EQ(tensor.dim(0), 4);
    EXPECT_EQ(tensor.dim(1), 5);
    EXPECT_EQ(tensor.dim(2), 6);
    EXPECT_EQ(tensor.dim(-1), 0);
    EXPECT_EQ(tensor.dim(3), 0);
}

TEST(TensorTest, Zeros) {
    auto result = Tensor::zeros({2, 3}, DataType::Float32);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());
    const f32* data = tensor.ptr<f32>();

    ASSERT_NE(data, nullptr);

    for (usize i = 0; i < tensor.size(); ++i) {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST(TensorTest, FromVectorCopiesFloatData) {
    constexpr std::array<f32, 6> kData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<f32> data(kData.begin(), kData.end());

    auto result = Tensor::from_vector(data);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());
    EXPECT_EQ(tensor.size(), kData.size());
    EXPECT_EQ(tensor.byte_size(), kData.size() * sizeof(f32));
    EXPECT_EQ(tensor.ndim(), 1);
    EXPECT_EQ(tensor.dim(0), static_cast<i32>(kData.size()));
    EXPECT_EQ(tensor.dtype(), DataType::Float32);

    expect_array_eq(tensor.ptr<f32>(), kData);
}

TEST(TensorTest, FromVectorMapsIntegralType) {
    constexpr std::array<i32, 4> kData{7, 8, 9, 10};
    std::vector<i32> data(kData.begin(), kData.end());

    auto result = Tensor::from_vector(data);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    EXPECT_EQ(tensor.dtype(), DataType::Int32);
    expect_array_eq(tensor.ptr<i32>(), kData);
}

TEST(TensorTest, FromVectorRejectsEmptyInput) {
    std::vector<f32> data;

    auto result = Tensor::from_vector(data);

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorTest, PtrAndIndexProvideMutableAndConstAccess) {
    auto result = Tensor::create({2, 2}, DataType::Int32);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    auto ptr = tensor.ptr<i32>();
    ASSERT_NE(ptr, nullptr);

    ptr[0] = 11;
    tensor.index<i32>(1) = 22;
    *tensor.ptr<i32>(2) = 33;
    tensor.ptr<i32>(3)[0] = 44;

    const Tensor& const_tensor = tensor;
    EXPECT_EQ(const_tensor.index<i32>(0), 11);
    EXPECT_EQ(const_tensor.index<i32>(1), 22);
    EXPECT_EQ(*const_tensor.ptr<i32>(2), 33);
    EXPECT_EQ(const_tensor.ptr<i32>(3)[0], 44);
}

TEST(TensorTest, StrideComputesRowMajorLayout) {
    auto result = Tensor::create({2, 3, 4}, DataType::Float32);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    EXPECT_EQ(tensor.stride(), std::vector<usize>({12, 4, 1}));
}

TEST(TensorTest, StrideForSingleDimensionTensorIsOne) {
    auto result = Tensor::create({5}, DataType::UInt8);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    EXPECT_EQ(tensor.stride(), std::vector<usize>({1}));
}

TEST(TensorTest, ReshapeUpdatesDimensionsWithoutChangingData) {
    constexpr std::array<i32, 6> kData{1, 2, 3, 4, 5, 6};
    std::vector<i32> data(kData.begin(), kData.end());
    auto result = Tensor::from_vector(data);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    auto reshape_result = tensor.reshape({2, 3});
    ASSERT_TRUE(reshape_result.is_ok()) << reshape_result.error().message();

    EXPECT_EQ(tensor.dims(), std::vector<i32>({2, 3}));
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.size(), kData.size());
    EXPECT_EQ(tensor.stride(), std::vector<usize>({3, 1}));
    expect_array_eq(tensor.ptr<i32>(), kData);
}

TEST(TensorTest, ReshapeSupportsScalarShapeWhenElementCountMatches) {
    auto result = Tensor::create({1}, DataType::Float64);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    auto reshape_result = tensor.reshape({});
    ASSERT_TRUE(reshape_result.is_ok()) << reshape_result.error().message();

    EXPECT_TRUE(tensor.dims().empty());
    EXPECT_EQ(tensor.ndim(), 0);
    EXPECT_EQ(tensor.size(), 1);
    EXPECT_TRUE(tensor.stride().empty());
}

TEST(TensorTest, ReshapeRejectsMismatchedElementCountAndPreservesShape) {
    auto result = Tensor::create({2, 3}, DataType::Float32);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());
    const auto original_dims = tensor.dims();

    auto reshape_result = tensor.reshape({4, 4});

    EXPECT_TRUE(reshape_result.is_err());
    EXPECT_EQ(reshape_result.error().code(), ErrorCode::InvalidArgument);
    EXPECT_EQ(tensor.dims(), original_dims);
}

TEST(TensorTest, CloneDuplicatesDataAndMetadata) {
    constexpr std::array<i32, 4> kData{4, 3, 2, 1};
    std::vector<i32> data(kData.begin(), kData.end());
    auto result = Tensor::from_vector(data);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());
    auto reshape_result = tensor.reshape({2, 2});
    ASSERT_TRUE(reshape_result.is_ok());

    auto clone_result = tensor.clone();
    ASSERT_TRUE(clone_result.is_ok()) << clone_result.error().message();

    Tensor clone = std::move(clone_result.value());

    EXPECT_NE(clone.data(), tensor.data());
    EXPECT_EQ(clone.dims(), tensor.dims());
    EXPECT_EQ(clone.dtype(), tensor.dtype());
    EXPECT_EQ(clone.device(), tensor.device());
    expect_array_eq(clone.ptr<i32>(), kData);

    tensor.index<i32>(0) = 99;
    EXPECT_EQ(clone.index<i32>(0), 4);
}

TEST(TensorTest, ToCpuFromCpuReturnsDeepCopy) {
    constexpr std::array<f32, 4> kData{0.5f, 1.5f, 2.5f, 3.5f};
    std::vector<f32> data(kData.begin(), kData.end());
    auto result = Tensor::from_vector(data);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    auto copy_result = tensor.to(DeviceType::CPU);
    ASSERT_TRUE(copy_result.is_ok()) << copy_result.error().message();

    Tensor copy = std::move(copy_result.value());

    EXPECT_NE(copy.data(), tensor.data());
    EXPECT_EQ(copy.dims(), tensor.dims());
    EXPECT_EQ(copy.dtype(), tensor.dtype());
    EXPECT_EQ(copy.device(), DeviceType::CPU);
    expect_array_eq(copy.ptr<f32>(), kData);

    tensor.index<f32>(0) = 9.5f;
    EXPECT_EQ(copy.index<f32>(0), 0.5f);
}

TEST(TensorTest, ToRejectsUnsupportedDevice) {
    auto result = Tensor::create({2, 3}, DataType::Float32);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());

    auto convert_result = tensor.to(DeviceType::Mmap);

    EXPECT_TRUE(convert_result.is_err());
    EXPECT_EQ(convert_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorTest, ToStringIncludesShapeTypeDeviceAndSize) {
    auto result = Tensor::create({2, 3}, DataType::Float32);
    ASSERT_TRUE(result.is_ok());

    Tensor tensor = std::move(result.value());
    const std::string description = tensor.to_string();

    EXPECT_NE(description.find("Tensor(shape=[2, 3]"), std::string::npos);
    EXPECT_NE(description.find("dtype=float32"), std::string::npos);
    EXPECT_NE(description.find("device=CPU"), std::string::npos);
    EXPECT_NE(description.find("size=6"), std::string::npos);
}

#ifdef PULSE_USE_CUDA

TEST(TensorTest, CreateCudaTensor) {
    auto result = Tensor::create({2, 3}, DataType::Float32, DeviceType::CUDA);

    if (!result.is_ok()) {
        GTEST_SKIP() << result.error().message();
    }

    Tensor tensor = std::move(result.value());

    EXPECT_EQ(tensor.device(), DeviceType::CUDA);
    EXPECT_EQ(tensor.dims(), std::vector<i32>({2, 3}));
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_NE(tensor.data(), nullptr);
}

TEST(TensorTest, ToSupportsCpuCudaRoundTrip) {
    constexpr std::array<f32, 4> kData{2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<f32> data(kData.begin(), kData.end());
    auto host_result = Tensor::from_vector(data);
    ASSERT_TRUE(host_result.is_ok());

    Tensor host = std::move(host_result.value());

    auto cuda_result = host.to(DeviceType::CUDA);
    if (!cuda_result.is_ok()) {
        GTEST_SKIP() << cuda_result.error().message();
    }

    Tensor cuda_tensor = std::move(cuda_result.value());
    EXPECT_EQ(cuda_tensor.device(), DeviceType::CUDA);

    auto roundtrip_result = cuda_tensor.to(DeviceType::CPU);
    ASSERT_TRUE(roundtrip_result.is_ok()) << roundtrip_result.error().message();

    Tensor roundtrip = std::move(roundtrip_result.value());
    EXPECT_EQ(roundtrip.device(), DeviceType::CPU);
    expect_array_eq(roundtrip.ptr<f32>(), kData);
}

#endif
