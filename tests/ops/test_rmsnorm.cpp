#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/ops/rmsnorm.hpp"

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
    std::uniform_real_distribution<f32> dist(-2.0f, 2.0f);
    std::vector<f32> values(count);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

template<>
std::vector<f64> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f64> dist(-2.0, 2.0);
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
    std::uniform_real_distribution<f32> dist(-2.0f, 2.0f);
    std::vector<f16> values(count);
    for (auto& value : values) {
        value = __float2half(dist(rng));
    }
    return values;
}

template<>
std::vector<bf16> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f32> dist(-2.0f, 2.0f);
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
std::vector<double> rms_norm_expected(const std::vector<T>& input,
                                      const std::vector<T>* weight,
                                      const std::vector<i32>& dims,
                                      const std::vector<i32>& normalized_shape,
                                      double eps) {
    const usize normalized_size = numel(normalized_shape);
    const usize rows = numel(dims) / normalized_size;
    std::vector<double> output(input.size(), 0.0);

    for (usize row = 0; row < rows; ++row) {
        const auto row_base = row * normalized_size;
        double sum_squares = 0.0;
        for (usize col = 0; col < normalized_size; ++col) {
            double value = 0.0;
            if constexpr (std::is_same_v<T, f16>) {
                value = static_cast<double>(__half2float(input[row_base + col]));
            } else if constexpr (std::is_same_v<T, bf16>) {
                value = static_cast<double>(__bfloat162float(input[row_base + col]));
            } else {
                value = static_cast<double>(input[row_base + col]);
            }
            sum_squares += value * value;
        }

        const double inv_rms = 1.0 / std::sqrt(sum_squares / static_cast<double>(normalized_size) + eps);
        for (usize col = 0; col < normalized_size; ++col) {
            double value = 0.0;
            if constexpr (std::is_same_v<T, f16>) {
                value = static_cast<double>(__half2float(input[row_base + col]));
            } else if constexpr (std::is_same_v<T, bf16>) {
                value = static_cast<double>(__bfloat162float(input[row_base + col]));
            } else {
                value = static_cast<double>(input[row_base + col]);
            }

            double scale = 1.0;
            if (weight != nullptr) {
                if constexpr (std::is_same_v<T, f16>) {
                    scale = static_cast<double>(__half2float((*weight)[col]));
                } else if constexpr (std::is_same_v<T, bf16>) {
                    scale = static_cast<double>(__bfloat162float((*weight)[col]));
                } else {
                    scale = static_cast<double>((*weight)[col]);
                }
            }
            output[row_base + col] = quantize_output_to_double<T>(value * inv_rms * scale);
        }
    }

    return output;
}

template<typename T>
void expect_values_near(const T* actual, const std::vector<double>& expected, double tolerance) {
    ASSERT_NE(actual, nullptr);
    for (usize i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_same_v<T, f16>) {
            EXPECT_NEAR(static_cast<double>(__half2float(actual[i])), expected[i], tolerance);
        } else if constexpr (std::is_same_v<T, bf16>) {
            EXPECT_NEAR(static_cast<double>(__bfloat162float(actual[i])), expected[i], tolerance);
        } else {
            EXPECT_NEAR(static_cast<double>(actual[i]), expected[i], tolerance);
        }
    }
}

}  // namespace

TEST(RMSNormOpTest, AppliesFloat32CpuRmsNormWithWeight) {
    const std::vector<i32> dims{2, 3, 4};
    const std::vector<i32> normalized_shape{4};
    constexpr f64 eps = 1.0e-6;
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CPU, 4101);
    auto weight = make_random_tensor_or_fail<f32>(normalized_shape, DeviceType::CPU, 4102);
    assert_tensor_created(input);
    assert_tensor_created(weight);
    auto output_result = Tensor::create(dims, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));
    auto expected = rms_norm_expected(input.values, &weight.values, dims, normalized_shape, eps);

    auto norm_result = ops::rms_norm(input.tensor, normalized_shape, &weight.tensor, output, eps);

    ASSERT_TRUE(norm_result.is_ok()) << norm_result.error().message();
    expect_values_near(output.ptr<f32>(), expected, 1e-5);
}

TEST(RMSNormOpTest, AppliesFloat64CpuRmsNormWithoutWeightAcrossTrailingShape) {
    const std::vector<i32> dims{2, 3, 4};
    const std::vector<i32> normalized_shape{3, 4};
    constexpr f64 eps = 1.0e-12;
    auto input = make_random_tensor_or_fail<f64>(dims, DeviceType::CPU, 4201);
    assert_tensor_created(input);
    auto output_result = Tensor::create(dims, DataType::Float64, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));
    auto expected = rms_norm_expected(input.values, static_cast<const std::vector<f64>*>(nullptr), dims, normalized_shape, eps);

    auto norm_result = ops::rms_norm(
        input.tensor, normalized_shape, static_cast<const Tensor*>(nullptr), output, eps);

    ASSERT_TRUE(norm_result.is_ok()) << norm_result.error().message();
    expect_values_near(output.ptr<f64>(), expected, 1e-12);
}

TEST(RMSNormOpTest, RejectsNormalizedShapeThatDoesNotMatchInputSuffix) {
    auto input = make_random_tensor_or_fail<f32>({2, 3, 4}, DeviceType::CPU, 4301);
    assert_tensor_created(input);
    auto output_result = Tensor::create({2, 3, 4}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto norm_result = ops::rms_norm(
        input.tensor, std::vector<i32>{2, 4}, static_cast<const Tensor*>(nullptr), output);

    ASSERT_TRUE(norm_result.is_err());
    EXPECT_EQ(norm_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(RMSNormOpTest, RejectsWeightShapeMismatch) {
    auto input = make_random_tensor_or_fail<f32>({2, 3, 4}, DeviceType::CPU, 4401);
    auto weight = make_random_tensor_or_fail<f32>({3}, DeviceType::CPU, 4402);
    assert_tensor_created(input);
    assert_tensor_created(weight);
    auto output_result = Tensor::create({2, 3, 4}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto norm_result = ops::rms_norm(input.tensor, std::vector<i32>{4}, &weight.tensor, output);

    ASSERT_TRUE(norm_result.is_err());
    EXPECT_EQ(norm_result.error().code(), ErrorCode::ShapeMismatch);
}

#ifdef PULSE_USE_CUDA

TEST(RMSNormOpTest, AppliesFloat32CudaRmsNormWithWeight) {
    const std::vector<i32> dims{2, 3, 8};
    const std::vector<i32> normalized_shape{8};
    constexpr f64 eps = 1.0e-6;
    auto input = make_random_tensor_or_fail<f32>(dims, DeviceType::CUDA, 4501);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto weight = make_random_tensor_or_fail<f32>(normalized_shape, DeviceType::CUDA, 4502);
    ASSERT_FALSE(weight.tensor.empty());
    auto output_result = Tensor::create(dims, DataType::Float32, DeviceType::CUDA);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));
    auto expected = rms_norm_expected(input.values, &weight.values, dims, normalized_shape, eps);

    auto norm_result = ops::rms_norm(input.tensor, normalized_shape, &weight.tensor, output, eps);

    ASSERT_TRUE(norm_result.is_ok()) << norm_result.error().message();
    auto host_result = output.to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());
    expect_values_near(host_output.ptr<f32>(), expected, 1e-4);
}

TEST(RMSNormOpTest, AppliesBFloat16CudaRmsNormWithWeight) {
    const std::vector<i32> dims{2, 2, 8};
    const std::vector<i32> normalized_shape{8};
    constexpr f64 eps = 7.8125e-3;
    auto input = make_random_tensor_or_fail<bf16>(dims, DeviceType::CUDA, 4601);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto weight = make_random_tensor_or_fail<bf16>(normalized_shape, DeviceType::CUDA, 4602);
    ASSERT_FALSE(weight.tensor.empty());
    auto output_result = Tensor::create(dims, DataType::BFloat16, DeviceType::CUDA);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));
    auto expected = rms_norm_expected(input.values, &weight.values, dims, normalized_shape, eps);

    auto norm_result = ops::rms_norm(input.tensor, normalized_shape, &weight.tensor, output, eps);

    ASSERT_TRUE(norm_result.is_ok()) << norm_result.error().message();
    auto host_result = output.to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());
    expect_values_near(host_output.ptr<bf16>(), expected, 5e-3);
}

#endif
