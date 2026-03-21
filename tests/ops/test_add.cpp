#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/ops/add.hpp"

using namespace pulse;

namespace {

usize numel(const std::vector<i32>& dims) {
    return std::accumulate(dims.begin(), dims.end(), usize(1), [](usize lhs, i32 rhs) {
        return lhs * static_cast<usize>(rhs);
    });
}

template<typename T>
struct TensorWithValues {
    Tensor tensor;
    std::vector<T> values;
};

template<typename T>
std::vector<T> make_random_values(usize count, u32 seed);

template<>
std::vector<f32> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-8.0f, 8.0f);
    std::vector<f32> values(count);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

template<>
std::vector<f64> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f64> dist(-8.0, 8.0);
    std::vector<f64> values(count);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

template<>
std::vector<i32> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<i32> dist(-50, 50);
    std::vector<i32> values(count);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

#ifdef PULSE_USE_CUDA
template<>
std::vector<f16> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-8.0f, 8.0f);
    std::vector<f16> values(count);
    for (auto& value : values) {
        value = __float2half(dist(rng));
    }
    return values;
}

template<>
std::vector<bf16> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-8.0f, 8.0f);
    std::vector<bf16> values(count);
    for (auto& value : values) {
        value = __float2bfloat16(dist(rng));
    }
    return values;
}
#endif

template<typename T>
TensorWithValues<T> make_random_tensor_or_fail(
    const std::vector<i32>& dims,
    DeviceType device,
    u32 seed) {
    auto values = make_random_values<T>(numel(dims), seed);
    auto tensor_result = Tensor::from_vector(values, device);
    if (!tensor_result.is_ok()) {
        return {};
    }

    Tensor tensor(std::move(tensor_result.value()));
    auto reshape_result = tensor.reshape(dims);
    if (!reshape_result.is_ok()) {
        return {};
    }

    return {std::move(tensor), std::move(values)};
}

template<typename T>
void assert_tensor_created(const TensorWithValues<T>& tensor_with_values) {
    ASSERT_FALSE(tensor_with_values.tensor.empty());
}

template<typename T>
std::vector<T> add_expected(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    EXPECT_EQ(lhs.size(), rhs.size());

    std::vector<T> result(lhs.size());
    for (usize i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

template<typename T>
void expect_values_eq(const T* actual, const std::vector<T>& expected) {
    ASSERT_NE(actual, nullptr);

    for (usize i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_same_v<T, f32>) {
            EXPECT_FLOAT_EQ(actual[i], expected[i]);
        } else if constexpr (std::is_same_v<T, f64>) {
            EXPECT_DOUBLE_EQ(actual[i], expected[i]);
        } else {
            EXPECT_EQ(actual[i], expected[i]);
        }
    }
}

#ifdef PULSE_USE_CUDA
template<typename T>
std::vector<float> add_expected_as_float(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    EXPECT_EQ(lhs.size(), rhs.size());

    std::vector<float> result(lhs.size());
    for (usize i = 0; i < lhs.size(); ++i) {
        if constexpr (std::is_same_v<T, f16>) {
            result[i] = __half2float(__float2half(__half2float(lhs[i]) + __half2float(rhs[i])));
        } else if constexpr (std::is_same_v<T, bf16>) {
            result[i] =
                __bfloat162float(__float2bfloat16(__bfloat162float(lhs[i]) + __bfloat162float(rhs[i])));
        } else {
            result[i] = static_cast<float>(lhs[i]) + static_cast<float>(rhs[i]);
        }
    }
    return result;
}

template<typename T>
void expect_values_near(const T* actual, const std::vector<float>& expected, float tolerance = 1e-3f) {
    ASSERT_NE(actual, nullptr);

    for (usize i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_same_v<T, f16>) {
            EXPECT_NEAR(__half2float(actual[i]), expected[i], tolerance);
        } else if constexpr (std::is_same_v<T, bf16>) {
            EXPECT_NEAR(__bfloat162float(actual[i]), expected[i], tolerance);
        } else {
            EXPECT_NEAR(static_cast<float>(actual[i]), expected[i], tolerance);
        }
    }
}
#endif

}  // namespace

TEST(TensorAddTest, AddsFloat32TensorsAndReturnsNewTensor) {
    const std::vector<i32> dims{2, 3, 4};
    auto input1 = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 101);
    auto input2 = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 202);
    assert_tensor_created(input1);
    assert_tensor_created(input2);
    auto expected = add_expected(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);

    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();
    Tensor output = std::move(add_result.value());

    EXPECT_EQ(output.device(), DeviceType::CPU);
    EXPECT_EQ(output.dtype(), DataType::Float32);
    EXPECT_EQ(output.dims(), dims);
    expect_values_eq(output.ptr<f32>(), expected);
    expect_values_eq(input1.tensor.ptr<f32>(), input1.values);
    expect_values_eq(input2.tensor.ptr<f32>(), input2.values);
}

TEST(TensorAddTest, AddsFloat64TensorsAndReturnsNewTensor) {
    const std::vector<i32> dims{2, 2, 3};
    auto input1 = make_random_tensor_or_fail<f64>(dims, DeviceType::CPU, 303);
    auto input2 = make_random_tensor_or_fail<f64>(dims, DeviceType::CPU, 404);
    assert_tensor_created(input1);
    assert_tensor_created(input2);
    auto expected = add_expected(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);

    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();
    Tensor output = std::move(add_result.value());

    EXPECT_EQ(output.dims(), dims);
    expect_values_eq(output.ptr<f64>(), expected);
}

TEST(TensorAddTest, RejectsMismatchedInputSizes) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 2, 3}, DeviceType::CPU, 505);
    auto input2 = make_random_tensor_or_fail<f32>({2, 3, 3}, DeviceType::CPU, 606);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto add_result = input1.tensor.add(input2.tensor);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(TensorAddTest, RejectsMismatchedShapesWithSameElementCount) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 515);
    auto input2 = make_random_tensor_or_fail<f32>({3, 2}, DeviceType::CPU, 616);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto add_result = input1.tensor.add(input2.tensor);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(TensorAddTest, OpsAddRejectsMismatchedOutputShapeWithSameElementCount) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 525);
    auto input2 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 626);
    auto output = make_random_tensor_or_fail<f32>({3, 2}, DeviceType::CPU, 727);
    assert_tensor_created(input1);
    assert_tensor_created(input2);
    assert_tensor_created(output);

    auto add_result = ops::add(input1.tensor, input2.tensor, output.tensor);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(TensorAddTest, RejectsMismatchedDtypes) {
    const std::vector<i32> dims{2, 2, 2};
    auto input1 = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 707);
    auto input2 = make_random_tensor_or_fail<i32>(dims, DeviceType::CPU, 808);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto add_result = input1.tensor.add(input2.tensor);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::DtypeMismatch);
}

TEST(TensorAddTest, RejectsEmptySelfTensor) {
    Tensor input1;
    auto input2 = make_random_tensor_or_fail<f32>({2, 2}, DeviceType::CPU, 909);
    assert_tensor_created(input2);

    auto add_result = input1.add(input2.tensor);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorAddTest, RejectsEmptyOtherTensor) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 2}, DeviceType::CPU, 1001);
    assert_tensor_created(input1);
    Tensor input2;

    auto add_result = input1.tensor.add(input2);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorAddTest, RejectsUnsupportedCpuDtype) {
    const std::vector<i32> dims{2, 2, 2};
    auto input1 = make_random_tensor_or_fail<i32>(dims, DeviceType::CPU, 1101);
    auto input2 = make_random_tensor_or_fail<i32>(dims, DeviceType::CPU, 1201);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto add_result = input1.tensor.add(input2.tensor);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::NotImplemented);
}

#ifdef PULSE_USE_CUDA

TEST(TensorAddTest, RejectsDeviceMismatch) {
    const std::vector<i32> dims{2, 3};
    auto input1 = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 1301);
    assert_tensor_created(input1);
    auto input2_result = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 1401);
    if (input2_result.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    auto add_result = input1.tensor.add(input2_result.tensor);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::DeviceMismatch);
}

TEST(TensorAddTest, AddsFloat32CudaTensorsUsingVectorizedPath) {
    const std::vector<i32> dims{2, 2, 2};
    auto input1 = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 1501);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 1601);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = add_expected(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);
    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();

    auto host_result = add_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_eq(host_output.ptr<f32>(), expected);
}

TEST(TensorAddTest, AddsFloat32CudaTensorsWithTail) {
    const std::vector<i32> dims{2, 1, 5};
    auto input1 = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 1701);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 1801);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = add_expected(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);
    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();

    auto host_result = add_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_eq(host_output.ptr<f32>(), expected);
}

TEST(TensorAddTest, AddsFloat64CudaTensors) {
    const std::vector<i32> dims{2, 1, 3};
    auto input1 = make_random_tensor_or_fail<f64>(dims, DeviceType::CUDA, 1901);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f64>(dims, DeviceType::CUDA, 2001);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = add_expected(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);
    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();

    auto host_result = add_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_eq(host_output.ptr<f64>(), expected);
}

TEST(TensorAddTest, AddsFloat16CudaTensorsUsingPackedPath) {
    const std::vector<i32> dims{2, 2, 2};
    auto input1 = make_random_tensor_or_fail<f16>(dims, DeviceType::CUDA, 2101);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f16>(dims, DeviceType::CUDA, 2201);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = add_expected_as_float(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);
    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();

    auto host_result = add_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<f16>(), expected);
}

TEST(TensorAddTest, AddsFloat16CudaTensorsWithTail) {
    const std::vector<i32> dims{1, 11};
    auto input1 = make_random_tensor_or_fail<f16>(dims, DeviceType::CUDA, 2301);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f16>(dims, DeviceType::CUDA, 2401);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = add_expected_as_float(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);
    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();

    auto host_result = add_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<f16>(), expected);
}

TEST(TensorAddTest, AddsBFloat16CudaTensorsUsingPackedPath) {
    const std::vector<i32> dims{2, 2, 2};
    auto input1 = make_random_tensor_or_fail<bf16>(dims, DeviceType::CUDA, 2501);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<bf16>(dims, DeviceType::CUDA, 2601);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = add_expected_as_float(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);
    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();

    auto host_result = add_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<bf16>(), expected);
}

TEST(TensorAddTest, AddsBFloat16CudaTensorsWithTail) {
    const std::vector<i32> dims{1, 11};
    auto input1 = make_random_tensor_or_fail<bf16>(dims, DeviceType::CUDA, 2701);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<bf16>(dims, DeviceType::CUDA, 2801);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = add_expected_as_float(input1.values, input2.values);

    auto add_result = input1.tensor.add(input2.tensor);
    ASSERT_TRUE(add_result.is_ok()) << add_result.error().message();

    auto host_result = add_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<bf16>(), expected);
}

TEST(TensorAddTest, RejectsUnsupportedCudaDtype) {
    const std::vector<i32> dims{2, 3};
    auto input1 = make_random_tensor_or_fail<i32>(dims, DeviceType::CUDA, 2901);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<i32>(dims, DeviceType::CUDA, 3001);
    ASSERT_FALSE(input2.tensor.empty());

    auto add_result = input1.tensor.add(input2.tensor);

    ASSERT_TRUE(add_result.is_err());
    EXPECT_EQ(add_result.error().code(), ErrorCode::NotImplemented);
}

#endif
