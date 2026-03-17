#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/core/var.hpp"
#include "pulse/layer/rmsnorm.hpp"

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

std::string scoped_name(std::string_view prefix, std::string_view name) {
    if (prefix.empty()) {
        return std::string(name);
    }

    return std::string(prefix) + "." + std::string(name);
}

template<typename T>
void insert_tensor_or_fail(VarMap& vars,
                           std::string name,
                           const std::vector<i32>& dims,
                           DeviceType device,
                           u32 seed) {
    auto tensor_with_values = make_random_tensor_or_fail<T>(dims, device, seed);
    assert_tensor_created(tensor_with_values);
    auto insert_result = vars.insert(std::move(name), std::move(tensor_with_values.tensor));
    ASSERT_TRUE(insert_result.is_ok()) << insert_result.error().message();
}

template<typename T>
std::vector<T> tensor_values_or_fail(const Tensor& tensor) {
    return std::vector<T>(tensor.ptr<T>(), tensor.ptr<T>() + tensor.size());
}

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
            const double value = static_cast<double>(input[row_base + col]);
            sum_squares += value * value;
        }

        const double inv_rms = 1.0 / std::sqrt(sum_squares / static_cast<double>(normalized_size) + eps);
        for (usize col = 0; col < normalized_size; ++col) {
            const double scale = weight == nullptr ? 1.0 : static_cast<double>((*weight)[col]);
            output[row_base + col] = static_cast<double>(input[row_base + col]) * inv_rms * scale;
        }
    }

    return output;
}

template<typename T>
void expect_values_near(const T* actual, const std::vector<double>& expected, double tolerance) {
    ASSERT_NE(actual, nullptr);
    for (usize i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(actual[i]), expected[i], tolerance);
    }
}

}  // namespace

TEST(RMSNormLayerTest, ForwardMatchesPyTorchStyleAffineRmsNorm) {
    const std::vector<i32> input_dims{2, 3, 4};
    const std::vector<i32> normalized_shape{4};
    auto input = make_random_tensor_or_fail<f32>(input_dims, DeviceType::CPU, 5101);
    assert_tensor_created(input);

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto norm_scope_result = root.pp("norm");
    ASSERT_TRUE(norm_scope_result.is_ok()) << norm_scope_result.error().message();
    const auto norm_scope = std::move(norm_scope_result.value());
    insert_tensor_or_fail<f32>(vars, scoped_name(norm_scope.prefix(), "weight"), normalized_shape, DeviceType::CPU, 5102);

    layer::RMSNorm norm(normalized_shape, -1.0, true, DeviceType::CPU, DataType::Float32);
    auto init_result = norm.init(norm_scope);
    ASSERT_TRUE(init_result.is_ok()) << init_result.error().message();
    ASSERT_NE(norm.weight(), nullptr);
    auto weight_values = tensor_values_or_fail<f32>(*norm.weight());
    auto expected = rms_norm_expected(input.values, &weight_values, input_dims, normalized_shape, norm.eps());

    auto forward_result = norm.forward(input.tensor);

    ASSERT_TRUE(forward_result.is_ok()) << forward_result.error().message();
    Tensor output = std::move(forward_result.value());

    EXPECT_EQ(output.dims(), input_dims);
    EXPECT_TRUE(norm.elementwise_affine());
    EXPECT_DOUBLE_EQ(norm.eps(), 1.1920928955078125e-7);
    expect_values_near(output.ptr<f32>(), expected, 1e-5);
}

TEST(RMSNormLayerTest, ForwardWithoutAffineDoesNotRequireInit) {
    const std::vector<i32> input_dims{2, 3, 4};
    const std::vector<i32> normalized_shape{3, 4};
    auto input = make_random_tensor_or_fail<f64>(input_dims, DeviceType::CPU, 5201);
    assert_tensor_created(input);
    layer::RMSNorm norm(normalized_shape, 1.0e-12, false, DeviceType::CPU, DataType::Float64);
    auto expected = rms_norm_expected(input.values, static_cast<const std::vector<f64>*>(nullptr), input_dims, normalized_shape, norm.eps());

    auto forward_result = norm(input.tensor);

    ASSERT_TRUE(forward_result.is_ok()) << forward_result.error().message();
    Tensor output = std::move(forward_result.value());

    EXPECT_FALSE(norm.elementwise_affine());
    EXPECT_EQ(norm.weight(), nullptr);
    expect_values_near(output.ptr<f64>(), expected, 1e-12);
}

TEST(RMSNormLayerTest, RejectsForwardWhenAffineWeightIsNotInitialized) {
    auto input = make_random_tensor_or_fail<f32>({2, 3, 4}, DeviceType::CPU, 5301);
    assert_tensor_created(input);
    layer::RMSNorm norm({4}, -1.0, true, DeviceType::CPU, DataType::Float32);

    auto forward_result = norm.forward(input.tensor);

    ASSERT_TRUE(forward_result.is_err());
    EXPECT_EQ(forward_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(RMSNormLayerTest, InitRejectsMissingWeightParameter) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    layer::RMSNorm norm({4}, -1.0, true, DeviceType::CPU, DataType::Float32);

    auto init_result = norm.init(root);

    ASSERT_TRUE(init_result.is_err());
    EXPECT_EQ(init_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(RMSNormLayerTest, ForwardRejectsInputSuffixMismatch) {
    auto input = make_random_tensor_or_fail<f32>({2, 3, 5}, DeviceType::CPU, 5401);
    assert_tensor_created(input);
    layer::RMSNorm norm({4}, -1.0, false, DeviceType::CPU, DataType::Float32);

    auto forward_result = norm.forward(input.tensor);

    ASSERT_TRUE(forward_result.is_err());
    EXPECT_EQ(forward_result.error().code(), ErrorCode::ShapeMismatch);
}
