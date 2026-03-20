#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/ops/mul.hpp"

using namespace pulse;

namespace {

usize numel(const std::vector<i32>& dims) {
    return std::accumulate(dims.begin(), dims.end(), usize(1), [](usize lhs, i32 rhs) {
        return lhs * static_cast<usize>(rhs);
    });
}

template<typename T>
std::vector<T> make_random_values(usize count, u32 seed);

template<>
std::vector<f32> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-3.0f, 3.0f);
    std::vector<f32> values(count);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

#ifdef PULSE_USE_CUDA
template<>
std::vector<bf16> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-3.0f, 3.0f);
    std::vector<bf16> values(count);
    for (auto& value : values) {
        value = __float2bfloat16(dist(rng));
    }
    return values;
}
#endif

template<typename T>
std::vector<T> mul_expected(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    std::vector<T> result(lhs.size());
    for (usize i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

#ifdef PULSE_USE_CUDA
template<typename T>
float scalar_to_float(T value) {
    return static_cast<float>(value);
}

template<>
float scalar_to_float<bf16>(bf16 value) {
    return __bfloat162float(value);
}
#endif

}  // namespace

TEST(TensorMulTest, MultipliesFloat32CpuTensors) {
    const std::vector<i32> dims{2, 3, 4};
    const auto lhs_values = make_random_values<f32>(numel(dims), 101);
    const auto rhs_values = make_random_values<f32>(numel(dims), 202);
    auto lhs_result = Tensor::from_vector(lhs_values, DeviceType::CPU);
    auto rhs_result = Tensor::from_vector(rhs_values, DeviceType::CPU);
    ASSERT_TRUE(lhs_result.is_ok()) << lhs_result.error().message();
    ASSERT_TRUE(rhs_result.is_ok()) << rhs_result.error().message();
    Tensor lhs(std::move(lhs_result.value()));
    Tensor rhs(std::move(rhs_result.value()));
    ASSERT_TRUE(lhs.reshape(dims).is_ok());
    ASSERT_TRUE(rhs.reshape(dims).is_ok());
    auto output_result = Tensor::create(dims, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto mul_result = ops::mul(lhs, rhs, output);

    ASSERT_TRUE(mul_result.is_ok()) << mul_result.error().message();
    const auto expected = mul_expected(lhs_values, rhs_values);
    for (usize i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output.ptr<f32>()[i], expected[i]);
    }
}

#ifdef PULSE_USE_CUDA
TEST(TensorMulTest, MultipliesBFloat16CudaTensors) {
    const std::vector<i32> dims{4, 8};
    const auto lhs_values = make_random_values<bf16>(numel(dims), 303);
    const auto rhs_values = make_random_values<bf16>(numel(dims), 404);
    auto lhs_result = Tensor::from_vector(lhs_values, DeviceType::CUDA);
    auto rhs_result = Tensor::from_vector(rhs_values, DeviceType::CUDA);
    ASSERT_TRUE(lhs_result.is_ok()) << lhs_result.error().message();
    ASSERT_TRUE(rhs_result.is_ok()) << rhs_result.error().message();
    Tensor lhs(std::move(lhs_result.value()));
    Tensor rhs(std::move(rhs_result.value()));
    ASSERT_TRUE(lhs.reshape(dims).is_ok());
    ASSERT_TRUE(rhs.reshape(dims).is_ok());
    auto output_result = Tensor::create(dims, DataType::BFloat16, DeviceType::CUDA);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto mul_result = ops::mul(lhs, rhs, output);
    ASSERT_TRUE(mul_result.is_ok()) << mul_result.error().message();

    auto host_result = output.to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    const auto host_output = std::move(host_result.value());
    for (usize i = 0; i < lhs_values.size(); ++i) {
        const auto expected =
            __bfloat162float(__float2bfloat16(scalar_to_float(lhs_values[i]) * scalar_to_float(rhs_values[i])));
        EXPECT_NEAR(__bfloat162float(host_output.ptr<bf16>()[i]), expected, 1e-3f);
    }
}
#endif
