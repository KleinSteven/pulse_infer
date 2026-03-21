#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/layer/silu.hpp"
#include "pulse/ops/silu.hpp"

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
std::vector<T> silu_expected(const std::vector<T>& input) {
    std::vector<T> result(input.size());
    for (usize i = 0; i < input.size(); ++i) {
        result[i] = input[i] / (static_cast<T>(1) + std::exp(-input[i]));
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
std::vector<float> silu_expected_as_float(const std::vector<T>& input) {
    std::vector<float> result(input.size());
    for (usize i = 0; i < input.size(); ++i) {
        float value_f32 = 0.0f;
        if constexpr (std::is_same_v<T, f16>) {
            value_f32 = __half2float(input[i]);
        } else if constexpr (std::is_same_v<T, bf16>) {
            value_f32 = __bfloat162float(input[i]);
        } else {
            value_f32 = static_cast<float>(input[i]);
        }

        const float silu_value = value_f32 / (1.0f + std::exp(-value_f32));
        if constexpr (std::is_same_v<T, f16>) {
            result[i] = __half2float(__float2half(silu_value));
        } else if constexpr (std::is_same_v<T, bf16>) {
            result[i] = __bfloat162float(__float2bfloat16(silu_value));
        } else {
            result[i] = silu_value;
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

TEST(SiLULayerTest, AppliesFloat32TensorAndReturnsNewTensor) {
    const std::vector<i32> dims{2, 3, 4};
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 101);
    assert_tensor_created(input);
    auto expected = silu_expected(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);

    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();
    Tensor output = std::move(silu_result.value());

    EXPECT_EQ(output.device(), DeviceType::CPU);
    EXPECT_EQ(output.dtype(), DataType::Float32);
    EXPECT_EQ(output.dims(), dims);
    expect_values_eq(output.ptr<f32>(), expected);
    expect_values_eq(input.tensor.ptr<f32>(), input.values);
}

TEST(SiLULayerTest, ForwardAppliesActivation) {
    const std::vector<i32> dims{2, 2, 3};
    auto input = make_random_tensor_or_fail<f64>(dims, DeviceType::CPU, 202);
    assert_tensor_created(input);
    auto expected = silu_expected(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);

    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();
    Tensor output = std::move(silu_result.value());

    EXPECT_EQ(output.dims(), dims);
    expect_values_eq(output.ptr<f64>(), expected);
}

TEST(SiLULayerTest, OpsSiluRejectsMismatchedOutputShapeWithSameElementCount) {
    auto input = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 303);
    auto output = make_random_tensor_or_fail<f32>({3, 2}, DeviceType::CPU, 404);
    assert_tensor_created(input);
    assert_tensor_created(output);

    auto silu_result = ops::silu(input.tensor, output.tensor);

    ASSERT_TRUE(silu_result.is_err());
    EXPECT_EQ(silu_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(SiLULayerTest, OpsSiluRejectsMismatchedOutputDtype) {
    const std::vector<i32> dims{2, 2, 2};
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 505);
    auto output = make_random_tensor_or_fail<i32>(dims, DeviceType::CPU, 606);
    assert_tensor_created(input);
    assert_tensor_created(output);

    auto silu_result = ops::silu(input.tensor, output.tensor);

    ASSERT_TRUE(silu_result.is_err());
    EXPECT_EQ(silu_result.error().code(), ErrorCode::DtypeMismatch);
}

TEST(SiLULayerTest, RejectsEmptyInputTensor) {
    Tensor input;
    auto output = make_random_tensor_or_fail<f32>({2, 2}, DeviceType::CPU, 707);
    assert_tensor_created(output);

    auto silu_result = ops::silu(input, output.tensor);

    ASSERT_TRUE(silu_result.is_err());
    EXPECT_EQ(silu_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(SiLULayerTest, RejectsUnsupportedCpuDtype) {
    const std::vector<i32> dims{2, 2, 2};
    auto input = make_random_tensor_or_fail<i32>(dims, DeviceType::CPU, 808);
    assert_tensor_created(input);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);

    ASSERT_TRUE(silu_result.is_err());
    EXPECT_EQ(silu_result.error().code(), ErrorCode::NotImplemented);
}

#ifdef PULSE_USE_CUDA

TEST(SiLULayerTest, OpsSiluRejectsDeviceMismatch) {
    const std::vector<i32> dims{2, 3};
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 909);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto output = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 1001);
    assert_tensor_created(output);

    auto silu_result = ops::silu(input.tensor, output.tensor);

    ASSERT_TRUE(silu_result.is_err());
    EXPECT_EQ(silu_result.error().code(), ErrorCode::DeviceMismatch);
}

TEST(SiLULayerTest, AppliesFloat32CudaTensorUsingVectorizedPath) {
    const std::vector<i32> dims{2, 2, 2};
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 1101);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = silu_expected(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);
    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();

    auto host_result = silu_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_eq(host_output.ptr<f32>(), expected);
}

TEST(SiLULayerTest, AppliesFloat32CudaTensorWithTail) {
    const std::vector<i32> dims{2, 1, 5};
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 1201);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = silu_expected(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);
    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();

    auto host_result = silu_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_eq(host_output.ptr<f32>(), expected);
}

TEST(SiLULayerTest, AppliesFloat64CudaTensor) {
    const std::vector<i32> dims{2, 1, 3};
    auto input = make_random_tensor_or_fail<f64>(dims, DeviceType::CUDA, 1301);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = silu_expected(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);
    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();

    auto host_result = silu_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_eq(host_output.ptr<f64>(), expected);
}

TEST(SiLULayerTest, AppliesFloat16CudaTensorUsingPackedPath) {
    const std::vector<i32> dims{2, 2, 2};
    auto input = make_random_tensor_or_fail<f16>(dims, DeviceType::CUDA, 1401);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = silu_expected_as_float(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);
    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();

    auto host_result = silu_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<f16>(), expected);
}

TEST(SiLULayerTest, AppliesFloat16CudaTensorWithTail) {
    const std::vector<i32> dims{1, 11};
    auto input = make_random_tensor_or_fail<f16>(dims, DeviceType::CUDA, 1501);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = silu_expected_as_float(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);
    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();

    auto host_result = silu_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<f16>(), expected);
}

TEST(SiLULayerTest, AppliesBFloat16CudaTensorUsingPackedPath) {
    const std::vector<i32> dims{2, 2, 2};
    auto input = make_random_tensor_or_fail<bf16>(dims, DeviceType::CUDA, 1601);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = silu_expected_as_float(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);
    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();

    auto host_result = silu_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<bf16>(), expected);
}

TEST(SiLULayerTest, AppliesBFloat16CudaTensorWithTail) {
    const std::vector<i32> dims{1, 11};
    auto input = make_random_tensor_or_fail<bf16>(dims, DeviceType::CUDA, 1701);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = silu_expected_as_float(input.values);

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);
    ASSERT_TRUE(silu_result.is_ok()) << silu_result.error().message();

    auto host_result = silu_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<bf16>(), expected);
}

TEST(SiLULayerTest, RejectsUnsupportedCudaDtype) {
    const std::vector<i32> dims{2, 3};
    auto input = make_random_tensor_or_fail<i32>(dims, DeviceType::CUDA, 1801);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    layer::SiLU silu;
    auto silu_result = silu.forward(input.tensor);

    ASSERT_TRUE(silu_result.is_err());
    EXPECT_EQ(silu_result.error().code(), ErrorCode::NotImplemented);
}

#endif
