#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/core/var.hpp"
#include "pulse/layer/linear.hpp"

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
void insert_linear_parameters_or_fail(VarMap& vars,
                                      const VarBuilder& builder,
                                      i32 in_features,
                                      i32 out_features,
                                      bool has_bias,
                                      DeviceType device,
                                      u32 weight_seed,
                                      u32 bias_seed) {
    insert_tensor_or_fail<T>(vars,
                             scoped_name(builder.prefix(), "weight"),
                             {out_features, in_features},
                             device,
                             weight_seed);

    if (has_bias) {
        insert_tensor_or_fail<T>(vars,
                                 scoped_name(builder.prefix(), "bias"),
                                 {out_features},
                                 device,
                                 bias_seed);
    }
}

template<typename T>
std::vector<T> tensor_values_or_fail(const Tensor& tensor) {
    const Tensor* source = &tensor;
    Tensor host_tensor;

    if (tensor.device() != DeviceType::CPU) {
        auto host_result = tensor.to(DeviceType::CPU);
        EXPECT_TRUE(host_result.is_ok()) << host_result.error().message();
        host_tensor = std::move(host_result.value());
        source = &host_tensor;
    }

    return std::vector<T>(source->ptr<T>(), source->ptr<T>() + source->size());
}

template<typename T>
std::vector<T> linear_expected(const std::vector<T>& input,
                               const std::vector<T>& weight,
                               const std::vector<T>* bias,
                               i32 batch_size,
                               i32 in_features,
                               i32 out_features) {
    std::vector<T> result(static_cast<usize>(batch_size) * static_cast<usize>(out_features),
                          static_cast<T>(0));
    for (i32 row = 0; row < batch_size; ++row) {
        for (i32 col = 0; col < out_features; ++col) {
            T sum = static_cast<T>(0);
            for (i32 inner = 0; inner < in_features; ++inner) {
                sum += input[static_cast<usize>(row) * static_cast<usize>(in_features) +
                             static_cast<usize>(inner)] *
                       weight[static_cast<usize>(col) * static_cast<usize>(in_features) +
                              static_cast<usize>(inner)];
            }

            if (bias != nullptr) {
                sum += (*bias)[col];
            }

            result[static_cast<usize>(row) * static_cast<usize>(out_features) + static_cast<usize>(col)] =
                sum;
        }
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
float scalar_to_float(T value) {
    return static_cast<float>(value);
}

template<>
float scalar_to_float<f16>(f16 value) {
    return __half2float(value);
}

template<>
float scalar_to_float<bf16>(bf16 value) {
    return __bfloat162float(value);
}

template<typename T>
float quantize_float_to_output(float value);

template<>
float quantize_float_to_output<f16>(float value) {
    return __half2float(__float2half(value));
}

template<>
float quantize_float_to_output<bf16>(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

template<typename T>
std::vector<float> linear_expected_as_float(const std::vector<T>& input,
                                            const std::vector<T>& weight,
                                            const std::vector<T>* bias,
                                            i32 batch_size,
                                            i32 in_features,
                                            i32 out_features) {
    std::vector<float> result(static_cast<usize>(batch_size) * static_cast<usize>(out_features), 0.0f);
    for (i32 row = 0; row < batch_size; ++row) {
        for (i32 col = 0; col < out_features; ++col) {
            float sum = 0.0f;
            for (i32 inner = 0; inner < in_features; ++inner) {
                sum += scalar_to_float(input[static_cast<usize>(row) * static_cast<usize>(in_features) +
                                             static_cast<usize>(inner)]) *
                       scalar_to_float(weight[static_cast<usize>(col) * static_cast<usize>(in_features) +
                                              static_cast<usize>(inner)]);
            }

            sum = quantize_float_to_output<T>(sum);
            if (bias != nullptr) {
                sum = quantize_float_to_output<T>(sum + scalar_to_float((*bias)[col]));
            }

            result[static_cast<usize>(row) * static_cast<usize>(out_features) + static_cast<usize>(col)] =
                sum;
        }
    }

    return result;
}

template<typename T>
void expect_values_near(const T* actual, const std::vector<float>& expected, float tolerance = 1e-3f) {
    ASSERT_NE(actual, nullptr);

    for (usize i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(scalar_to_float(actual[i]), expected[i], tolerance);
    }
}
#endif

}  // namespace

TEST(LinearLayerTest, CreateWithVarBuilderAppliesFloat32TensorWithBiasAndReturnsNewTensor) {
    auto input = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 101);
    assert_tensor_created(input);

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f32>(vars, root, 3, 4, true, DeviceType::CPU, 1001, 1002);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    ASSERT_NE(linear.weight(), nullptr);
    auto weight_values = tensor_values_or_fail<f32>(*linear.weight());
    ASSERT_NE(linear.bias(), nullptr);
    auto bias_values = tensor_values_or_fail<f32>(*linear.bias());
    auto expected = linear_expected(input.values, weight_values, &bias_values, 2, 3, 4);

    auto forward_result = linear.forward(input.tensor);

    ASSERT_TRUE(forward_result.is_ok()) << forward_result.error().message();
    Tensor output = std::move(forward_result.value());

    EXPECT_EQ(output.device(), DeviceType::CPU);
    EXPECT_EQ(output.dtype(), DataType::Float32);
    EXPECT_EQ(output.dims(), (std::vector<i32>{2, 4}));
    EXPECT_TRUE(linear.has_bias());
    ASSERT_NE(linear.weight(), nullptr);
    EXPECT_EQ(linear.weight()->dims(), (std::vector<i32>{4, 3}));
    EXPECT_EQ(linear.bias()->dims(), (std::vector<i32>{4}));
    expect_values_eq(output.ptr<f32>(), expected);
}

TEST(LinearLayerTest, CreateWithVarBuilderOperatorCallAppliesFloat64TensorWithoutBias) {
    auto input = make_random_tensor_or_fail<f64>({3, 2}, DeviceType::CPU, 202);
    assert_tensor_created(input);

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f64>(vars, root, 2, 5, false, DeviceType::CPU, 2001, 0);
    auto linear_result = layer::Linear::liner(2, 5, false, DeviceType::CPU, DataType::Float64, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    ASSERT_NE(linear.weight(), nullptr);
    auto weight_values = tensor_values_or_fail<f64>(*linear.weight());
    auto expected =
        linear_expected(input.values, weight_values, static_cast<const std::vector<f64>*>(nullptr), 3, 2, 5);

    auto forward_result = linear(input.tensor);

    ASSERT_TRUE(forward_result.is_ok()) << forward_result.error().message();
    Tensor output = std::move(forward_result.value());

    EXPECT_EQ(output.dims(), (std::vector<i32>{3, 5}));
    EXPECT_FALSE(linear.has_bias());
    EXPECT_EQ(linear.bias(), nullptr);
    expect_values_eq(output.ptr<f64>(), expected);
}

TEST(LinearLayerTest, LinerRejectsNonPositiveInFeatures) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_result = layer::Linear::liner(0, 4, true, DeviceType::CPU, DataType::Float32, root);

    ASSERT_TRUE(linear_result.is_err());
    EXPECT_EQ(linear_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(LinearLayerTest, LinerRejectsNonPositiveOutFeatures) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_result = layer::Linear::liner(4, 0, true, DeviceType::CPU, DataType::Float32, root);

    ASSERT_TRUE(linear_result.is_err());
    EXPECT_EQ(linear_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(LinearLayerTest, RejectsNonMatrixInputTensor) {
    auto input = make_random_tensor_or_fail<f32>({2, 2, 3}, DeviceType::CPU, 303);
    assert_tensor_created(input);

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f32>(vars, root, 3, 4, true, DeviceType::CPU, 3001, 3002);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    auto forward_result = linear.forward(input.tensor);

    ASSERT_TRUE(forward_result.is_err());
    EXPECT_EQ(forward_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(LinearLayerTest, RejectsInputFeatureMismatch) {
    auto input = make_random_tensor_or_fail<f32>({2, 4}, DeviceType::CPU, 404);
    assert_tensor_created(input);

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f32>(vars, root, 5, 3, true, DeviceType::CPU, 4001, 4002);
    auto linear_result = layer::Linear::liner(5, 3, true, DeviceType::CPU, DataType::Float32, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    auto forward_result = linear.forward(input.tensor);

    ASSERT_TRUE(forward_result.is_err());
    EXPECT_EQ(forward_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(LinearLayerTest, RejectsInputDtypeMismatch) {
    auto input = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 505);
    assert_tensor_created(input);

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f64>(vars, root, 3, 4, true, DeviceType::CPU, 5001, 5002);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float64, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    auto forward_result = linear.forward(input.tensor);

    ASSERT_TRUE(forward_result.is_err());
    EXPECT_EQ(forward_result.error().code(), ErrorCode::DtypeMismatch);
}

TEST(LinearLayerTest, RejectsEmptyInputTensor) {
    Tensor input;
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f32>(vars, root, 3, 4, true, DeviceType::CPU, 6001, 6002);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());

    auto forward_result = linear.forward(input);

    ASSERT_TRUE(forward_result.is_err());
    EXPECT_EQ(forward_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(LinearLayerTest, LinerRejectsUnsupportedCpuDtype) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Int32, root);

    ASSERT_TRUE(linear_result.is_err());
    EXPECT_EQ(linear_result.error().code(), ErrorCode::NotImplemented);
}

TEST(LinearLayerTest, LinerRejectsUnsupportedDevice) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::Unified, DataType::Float32, root);

    ASSERT_TRUE(linear_result.is_err());
    EXPECT_EQ(linear_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(LinearLayerTest, LinerRejectsMissingWeightParameter) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, root);

    ASSERT_TRUE(linear_result.is_err());
    EXPECT_EQ(linear_result.error().code(), ErrorCode::InvalidArgument);
    EXPECT_FALSE(vars.contains("weight"));
    EXPECT_FALSE(vars.contains("bias"));
}

TEST(LinearLayerTest, LinerRejectsMissingBiasParameterWhenRequested) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_tensor_or_fail<f32>(vars, "weight", {4, 3}, DeviceType::CPU, 7001);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, root);

    ASSERT_TRUE(linear_result.is_err());
    EXPECT_EQ(linear_result.error().code(), ErrorCode::InvalidArgument);
    EXPECT_TRUE(vars.contains("weight"));
    EXPECT_FALSE(vars.contains("bias"));
}

TEST(LinearLayerTest, CreateWithVarBuilderReadsNamedParameters) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto block_result = root.pp("decoder");
    ASSERT_TRUE(block_result.is_ok());
    auto linear_scope_result = block_result.value().pp("linear");
    ASSERT_TRUE(linear_scope_result.is_ok());
    const auto linear_scope = std::move(linear_scope_result.value());
    insert_linear_parameters_or_fail<f32>(vars, linear_scope, 3, 4, true, DeviceType::CPU, 8001, 8002);

    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, linear_scope);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());

    EXPECT_TRUE(vars.contains("decoder.linear.weight"));
    EXPECT_TRUE(vars.contains("decoder.linear.bias"));
    ASSERT_NE(vars.find("decoder.linear.weight"), nullptr);
    ASSERT_NE(vars.find("decoder.linear.bias"), nullptr);
    EXPECT_EQ(linear.weight(), vars.find("decoder.linear.weight"));
    EXPECT_EQ(linear.bias(), vars.find("decoder.linear.bias"));
}

TEST(LinearLayerTest, CreateWithVarBuilderReusesExistingParameters) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_scope_result = root.pp("mlp");
    ASSERT_TRUE(linear_scope_result.is_ok());
    const auto linear_scope = std::move(linear_scope_result.value());
    insert_linear_parameters_or_fail<f32>(vars, linear_scope, 3, 4, true, DeviceType::CPU, 9001, 9002);

    auto first_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, linear_scope);
    ASSERT_TRUE(first_result.is_ok()) << first_result.error().message();
    auto first = std::move(first_result.value());

    auto weight_result = linear_scope.get_or_create("weight", {4, 3}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(weight_result.is_ok()) << weight_result.error().message();
    Tensor* weight = weight_result.value();
    ASSERT_NE(weight, nullptr);
    weight->ptr<f32>()[0] = 1.25f;

    auto second_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, linear_scope);
    ASSERT_TRUE(second_result.is_ok()) << second_result.error().message();
    auto second = std::move(second_result.value());

    EXPECT_EQ(first.weight(), weight);
    EXPECT_EQ(second.weight(), weight);
    ASSERT_NE(second.weight(), nullptr);
    EXPECT_FLOAT_EQ(second.weight()->ptr<f32>()[0], 1.25f);
}

TEST(LinearLayerTest, CreateWithVarBuilderWithoutBiasDoesNotCreateBias) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_scope_result = root.pp("proj");
    ASSERT_TRUE(linear_scope_result.is_ok());
    const auto linear_scope = std::move(linear_scope_result.value());
    insert_linear_parameters_or_fail<f32>(vars, linear_scope, 3, 4, false, DeviceType::CPU, 10001, 0);

    auto linear_result = layer::Linear::liner(3, 4, false, DeviceType::CPU, DataType::Float32, linear_scope);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());

    EXPECT_TRUE(vars.contains("proj.weight"));
    EXPECT_FALSE(vars.contains("proj.bias"));
    EXPECT_FALSE(linear.has_bias());
    EXPECT_EQ(linear.bias(), nullptr);
}

TEST(LinearLayerTest, LinerRejectsExistingWeightShapeMismatch) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_scope_result = root.pp("down_proj");
    ASSERT_TRUE(linear_scope_result.is_ok());
    const auto linear_scope = std::move(linear_scope_result.value());

    auto weight_result = linear_scope.get_or_create("weight", {2, 3}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(weight_result.is_ok()) << weight_result.error().message();

    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, linear_scope);

    ASSERT_TRUE(linear_result.is_err());
    EXPECT_EQ(linear_result.error().code(), ErrorCode::InvalidArgument);
}

#ifdef PULSE_USE_CUDA

TEST(LinearLayerTest, RejectsInputDeviceMismatch) {
    auto input = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CUDA, 606);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f32>(vars, root, 3, 4, true, DeviceType::CPU, 11001, 11002);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CPU, DataType::Float32, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    auto forward_result = linear.forward(input.tensor);

    ASSERT_TRUE(forward_result.is_err());
    EXPECT_EQ(forward_result.error().code(), ErrorCode::DeviceMismatch);
}

TEST(LinearLayerTest, CreateAppliesFloat32CudaTensorWithBias) {
    auto input = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CUDA, 707);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f32>(vars, root, 3, 4, true, DeviceType::CUDA, 12001, 12002);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CUDA, DataType::Float32, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    ASSERT_NE(linear.weight(), nullptr);
    auto weight_values = tensor_values_or_fail<f32>(*linear.weight());
    ASSERT_NE(linear.bias(), nullptr);
    auto bias_values = tensor_values_or_fail<f32>(*linear.bias());
    auto expected = linear_expected(input.values, weight_values, &bias_values, 2, 3, 4);

    auto forward_result = linear.forward(input.tensor);
    ASSERT_TRUE(forward_result.is_ok()) << forward_result.error().message();

    auto host_result = forward_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{2, 4}));
    expect_values_eq(host_output.ptr<f32>(), expected);
}

TEST(LinearLayerTest, CreateAppliesFloat16CudaTensorWithBias) {
    auto input = make_random_tensor_or_fail<f16>({2, 3}, DeviceType::CUDA, 808);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<f16>(vars, root, 3, 4, true, DeviceType::CUDA, 13001, 13002);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CUDA, DataType::Float16, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    ASSERT_NE(linear.weight(), nullptr);
    auto weight_values = tensor_values_or_fail<f16>(*linear.weight());
    ASSERT_NE(linear.bias(), nullptr);
    auto bias_values = tensor_values_or_fail<f16>(*linear.bias());
    auto expected = linear_expected_as_float(input.values, weight_values, &bias_values, 2, 3, 4);

    auto forward_result = linear.forward(input.tensor);
    ASSERT_TRUE(forward_result.is_ok()) << forward_result.error().message();

    auto host_result = forward_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(host_output.ptr<f16>(), expected, 1e-2f);
}

TEST(LinearLayerTest, CreateAppliesBFloat16CudaTensorWithoutBias) {
    auto input = make_random_tensor_or_fail<bf16>({2, 3}, DeviceType::CUDA, 909);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    insert_linear_parameters_or_fail<bf16>(vars, root, 3, 4, false, DeviceType::CUDA, 14001, 0);
    auto linear_result = layer::Linear::liner(3, 4, false, DeviceType::CUDA, DataType::BFloat16, root);
    ASSERT_TRUE(linear_result.is_ok()) << linear_result.error().message();
    auto linear = std::move(linear_result.value());
    ASSERT_NE(linear.weight(), nullptr);
    auto weight_values = tensor_values_or_fail<bf16>(*linear.weight());
    auto expected = linear_expected_as_float(input.values,
                                             weight_values,
                                             static_cast<const std::vector<bf16>*>(nullptr),
                                             2,
                                             3,
                                             4);

    auto forward_result = linear.forward(input.tensor);
    ASSERT_TRUE(forward_result.is_ok()) << forward_result.error().message();

    auto host_result = forward_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(host_output.ptr<bf16>(), expected, 2e-2f);
}

TEST(LinearLayerTest, LinerRejectsUnsupportedCudaDtype) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto linear_result = layer::Linear::liner(3, 4, true, DeviceType::CUDA, DataType::Int32, root);

    ASSERT_TRUE(linear_result.is_err());
    EXPECT_EQ(linear_result.error().code(), ErrorCode::NotImplemented);
}

#endif
