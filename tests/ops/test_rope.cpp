#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/ops/rope.hpp"

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
    std::uniform_real_distribution<f32> dist(-3.0f, 3.0f);
    std::vector<f32> values(count);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

template<>
std::vector<f64> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f64> dist(-3.0, 3.0);
    std::vector<f64> values(count);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

#ifdef PULSE_USE_CUDA
template<>
std::vector<f16> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-3.0f, 3.0f);
    std::vector<f16> values(count);
    for (auto& value : values) {
        value = __float2half(dist(rng));
    }
    return values;
}

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
TensorWithValues<T> make_random_tensor_or_fail(const std::vector<i32>& dims, DeviceType device, u32 seed) {
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
double quantize_output_to_double(double value) {
    return value;
}

#ifdef PULSE_USE_CUDA
template<>
double quantize_output_to_double<f16>(double value) {
    return static_cast<double>(__half2float(__float2half(static_cast<float>(value))));
}

template<>
double quantize_output_to_double<bf16>(double value) {
    return static_cast<double>(__bfloat162float(__float2bfloat16(static_cast<float>(value))));
}
#endif

template<typename T>
std::vector<double> rope_expected_as_double(const std::vector<T>& input,
                                            const std::vector<i32>& dims,
                                            i32 position_offset,
                                            double theta,
                                            i32 rotary_dim) {
    EXPECT_GE(dims.size(), 2U);

    const i32 seq_len = dims[dims.size() - 2];
    const i32 head_dim = dims.back();
    const usize rows = numel(dims) / (static_cast<usize>(seq_len) * static_cast<usize>(head_dim));
    const i32 effective_rotary_dim = rotary_dim < 0 ? head_dim : rotary_dim;
    const i32 rotary_pairs = effective_rotary_dim / 2;
    std::vector<double> output(input.size(), 0.0);

    for (usize row = 0; row < rows; ++row) {
        const auto row_base = row * static_cast<usize>(seq_len) * static_cast<usize>(head_dim);
        for (i32 pos = 0; pos < seq_len; ++pos) {
            const auto token_base = row_base + static_cast<usize>(pos) * static_cast<usize>(head_dim);
            const double absolute_pos = static_cast<double>(position_offset + pos);

            for (i32 pair_idx = 0; pair_idx < rotary_pairs; ++pair_idx) {
                const auto even_idx = token_base + static_cast<usize>(pair_idx * 2);
                const auto odd_idx = even_idx + 1;
                const double exponent =
                    static_cast<double>(pair_idx * 2) / static_cast<double>(effective_rotary_dim);
                const double angle = absolute_pos / std::pow(theta, exponent);
                const double cos_value = std::cos(angle);
                const double sin_value = std::sin(angle);
                double x0 = 0.0;
                double x1 = 0.0;
#ifdef PULSE_USE_CUDA
                if constexpr (std::is_same_v<T, f16>) {
                    x0 = static_cast<double>(__half2float(input[even_idx]));
                    x1 = static_cast<double>(__half2float(input[odd_idx]));
                } else if constexpr (std::is_same_v<T, bf16>) {
                    x0 = static_cast<double>(__bfloat162float(input[even_idx]));
                    x1 = static_cast<double>(__bfloat162float(input[odd_idx]));
                } else {
                    x0 = static_cast<double>(input[even_idx]);
                    x1 = static_cast<double>(input[odd_idx]);
                }
#else
                x0 = static_cast<double>(input[even_idx]);
                x1 = static_cast<double>(input[odd_idx]);
#endif

                output[even_idx] = quantize_output_to_double<T>(x0 * cos_value - x1 * sin_value);
                output[odd_idx] = quantize_output_to_double<T>(x0 * sin_value + x1 * cos_value);
            }

            for (i32 dim = effective_rotary_dim; dim < head_dim; ++dim) {
                const auto idx = token_base + static_cast<usize>(dim);
#ifdef PULSE_USE_CUDA
                if constexpr (std::is_same_v<T, f16>) {
                    output[idx] = quantize_output_to_double<T>(static_cast<double>(__half2float(input[idx])));
                } else if constexpr (std::is_same_v<T, bf16>) {
                    output[idx] =
                        quantize_output_to_double<T>(static_cast<double>(__bfloat162float(input[idx])));
                } else {
                    output[idx] = quantize_output_to_double<T>(static_cast<double>(input[idx]));
                }
#else
                output[idx] = quantize_output_to_double<T>(static_cast<double>(input[idx]));
#endif
            }
        }
    }

    return output;
}

template<typename T>
void expect_values_near(const T* actual, const std::vector<double>& expected, double tolerance) {
    ASSERT_NE(actual, nullptr);

    for (usize i = 0; i < expected.size(); ++i) {
#ifdef PULSE_USE_CUDA
        if constexpr (std::is_same_v<T, f16>) {
            EXPECT_NEAR(static_cast<double>(__half2float(actual[i])), expected[i], tolerance);
        } else if constexpr (std::is_same_v<T, bf16>) {
            EXPECT_NEAR(static_cast<double>(__bfloat162float(actual[i])), expected[i], tolerance);
        } else {
            EXPECT_NEAR(static_cast<double>(actual[i]), expected[i], tolerance);
        }
#else
        EXPECT_NEAR(static_cast<double>(actual[i]), expected[i], tolerance);
#endif
    }
}

}  // namespace

TEST(TensorRopeTest, AppliesFloat32RopeAcrossLeadingDimensions) {
    const std::vector<i32> dims{2, 3, 6};
    constexpr i32 position_offset = 5;
    constexpr f32 theta = 10000.0f;
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 3101);
    assert_tensor_created(input);
    auto expected = rope_expected_as_double(input.values, dims, position_offset, theta, -1);

    auto rope_result = input.tensor.rope(position_offset, theta);

    ASSERT_TRUE(rope_result.is_ok()) << rope_result.error().message();
    Tensor output = std::move(rope_result.value());

    EXPECT_EQ(output.dims(), dims);
    expect_values_near(output.ptr<f32>(), expected, 1e-5);
}

TEST(TensorRopeTest, SupportsPartialRotaryDimAndPreservesTail) {
    const std::vector<i32> dims{2, 4, 8};
    constexpr i32 rotary_dim = 4;
    constexpr i32 position_offset = 2;
    constexpr f32 theta = 5000.0f;
    auto input = make_random_tensor_or_fail<f64>(dims, DeviceType::CPU, 3201);
    assert_tensor_created(input);
    auto expected = rope_expected_as_double(input.values, dims, position_offset, theta, rotary_dim);

    auto rope_result = input.tensor.rope(position_offset, theta, rotary_dim);

    ASSERT_TRUE(rope_result.is_ok()) << rope_result.error().message();
    Tensor output = std::move(rope_result.value());

    EXPECT_EQ(output.dims(), dims);
    expect_values_near(output.ptr<f64>(), expected, 1e-10);
}

TEST(TensorRopeTest, OpsRopeRejectsOddRotaryDim) {
    const std::vector<i32> dims{2, 3, 6};
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 3301);
    assert_tensor_created(input);
    auto output_result = Tensor::create(dims, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto rope_result = ops::rope(input.tensor, output, 0, 10000.0f, 3);

    ASSERT_TRUE(rope_result.is_err());
    EXPECT_EQ(rope_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorRopeTest, OpsRopeRejectsOneDimensionalInput) {
    const std::vector<i32> dims{8};
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 3401);
    assert_tensor_created(input);
    auto output_result = Tensor::create(dims, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto rope_result = ops::rope(input.tensor, output);

    ASSERT_TRUE(rope_result.is_err());
    EXPECT_EQ(rope_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorRopeTest, OpsRopeRejectsNonPositiveTheta) {
    const std::vector<i32> dims{2, 2, 4};
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 3501);
    assert_tensor_created(input);
    auto output_result = Tensor::create(dims, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto rope_result = ops::rope(input.tensor, output, 0, 0.0f);

    ASSERT_TRUE(rope_result.is_err());
    EXPECT_EQ(rope_result.error().code(), ErrorCode::InvalidArgument);
}

#ifdef PULSE_USE_CUDA

TEST(TensorRopeTest, AppliesFloat16CudaRope) {
    const std::vector<i32> dims{2, 3, 8};
    constexpr i32 position_offset = 7;
    constexpr i32 rotary_dim = 6;
    constexpr f32 theta = 10000.0f;
    auto input = make_random_tensor_or_fail<f16>(dims, DeviceType::CUDA, 3601);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = rope_expected_as_double(input.values, dims, position_offset, theta, rotary_dim);

    auto rope_result = input.tensor.rope(position_offset, theta, rotary_dim);

    ASSERT_TRUE(rope_result.is_ok()) << rope_result.error().message();
    auto host_result = rope_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<f16>(), expected, 2e-3);
}

TEST(TensorRopeTest, AppliesFloat32CudaRope) {
    const std::vector<i32> dims{1, 5, 6};
    constexpr i32 position_offset = 3;
    constexpr f32 theta = 10000.0f;
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 3701);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = rope_expected_as_double(input.values, dims, position_offset, theta, -1);

    auto rope_result = input.tensor.rope(position_offset, theta);

    ASSERT_TRUE(rope_result.is_ok()) << rope_result.error().message();
    auto host_result = rope_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<f32>(), expected, 1e-5);
}

TEST(TensorRopeTest, AppliesBFloat16CudaRope) {
    const std::vector<i32> dims{1, 4, 8};
    constexpr i32 position_offset = 1;
    constexpr i32 rotary_dim = 8;
    constexpr f32 theta = 10000.0f;
    auto input = make_random_tensor_or_fail<bf16>(dims, DeviceType::CUDA, 3801);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto expected = rope_expected_as_double(input.values, dims, position_offset, theta, rotary_dim);

    auto rope_result = input.tensor.rope(position_offset, theta, rotary_dim);

    ASSERT_TRUE(rope_result.is_ok()) << rope_result.error().message();
    auto host_result = rope_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), dims);
    expect_values_near(host_output.ptr<bf16>(), expected, 5e-3);
}

#endif
