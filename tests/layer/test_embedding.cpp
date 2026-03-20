#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/core/var.hpp"
#include "pulse/layer/embedding.hpp"

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
std::vector<i32> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<i32> dist(0, 15);
    std::vector<i32> values(count);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

#ifdef PULSE_USE_CUDA
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
std::vector<T> embedding_expected(const std::vector<i32>& input,
                                  const std::vector<T>& weight,
                                  i32 vocab_size,
                                  i32 embedding_dim) {
    std::vector<T> output(static_cast<usize>(input.size()) * static_cast<usize>(embedding_dim),
                          static_cast<T>(0));
    for (usize token_index = 0; token_index < input.size(); ++token_index) {
        const i32 token = input[token_index];
        if (token < 0 || token >= vocab_size) {
            continue;
        }

        const usize src_offset = static_cast<usize>(token) * static_cast<usize>(embedding_dim);
        const usize dst_offset = token_index * static_cast<usize>(embedding_dim);
        for (i32 dim = 0; dim < embedding_dim; ++dim) {
            output[dst_offset + static_cast<usize>(dim)] = weight[src_offset + static_cast<usize>(dim)];
        }
    }

    return output;
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
float scalar_to_float<bf16>(bf16 value) {
    return __bfloat162float(value);
}

template<typename T>
std::vector<float> embedding_expected_as_float(const std::vector<i32>& input,
                                               const std::vector<T>& weight,
                                               i32 vocab_size,
                                               i32 embedding_dim) {
    auto expected = embedding_expected(input, weight, vocab_size, embedding_dim);
    std::vector<float> output(expected.size());
    for (usize i = 0; i < expected.size(); ++i) {
        output[i] = scalar_to_float(expected[i]);
    }
    return output;
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

TEST(EmbeddingLayerTest, ForwardMatchesWeightRowsOnCpuFloat32) {
    constexpr i32 vocab_size = 16;
    constexpr i32 embedding_dim = 8;

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto embed_scope_result = root.pp("embed");
    ASSERT_TRUE(embed_scope_result.is_ok()) << embed_scope_result.error().message();
    const auto embed_scope = std::move(embed_scope_result.value());
    insert_tensor_or_fail<f32>(vars,
                               scoped_name(embed_scope.prefix(), "weight"),
                               {vocab_size, embedding_dim},
                               DeviceType::CPU,
                               6101);

    auto input = make_random_tensor_or_fail<i32>({4}, DeviceType::CPU, 6102);
    assert_tensor_created(input);

    auto embedding_result =
        layer::Embedding::embedding(vocab_size, embedding_dim, DeviceType::CPU, DataType::Float32, embed_scope);
    ASSERT_TRUE(embedding_result.is_ok()) << embedding_result.error().message();
    auto embedding = std::move(embedding_result.value());

    ASSERT_NE(embedding.weight(), nullptr);
    auto weight_values = tensor_values_or_fail<f32>(*embedding.weight());
    auto expected = embedding_expected(input.values, weight_values, vocab_size, embedding_dim);

    auto output_result = embedding.forward(input.tensor);

    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output = std::move(output_result.value());

    EXPECT_EQ(output.dims(), (std::vector<i32>{4, embedding_dim}));
    EXPECT_EQ(embedding.num_embeddings(), vocab_size);
    EXPECT_EQ(embedding.embedding_dim(), embedding_dim);
    expect_values_eq(output.ptr<f32>(), expected);
}

TEST(EmbeddingLayerTest, InitRejectsMissingWeightParameter) {
    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);

    auto embedding_result =
        layer::Embedding::embedding(8, 4, DeviceType::CPU, DataType::Float32, root);

    ASSERT_TRUE(embedding_result.is_err());
    EXPECT_EQ(embedding_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(EmbeddingLayerTest, ForwardRejectsNonInt32Input) {
    constexpr i32 vocab_size = 8;
    constexpr i32 embedding_dim = 4;

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto embed_scope_result = root.pp("embed");
    ASSERT_TRUE(embed_scope_result.is_ok()) << embed_scope_result.error().message();
    const auto embed_scope = std::move(embed_scope_result.value());
    insert_tensor_or_fail<f32>(vars,
                               scoped_name(embed_scope.prefix(), "weight"),
                               {vocab_size, embedding_dim},
                               DeviceType::CPU,
                               6201);

    auto embedding_result =
        layer::Embedding::embedding(vocab_size, embedding_dim, DeviceType::CPU, DataType::Float32, embed_scope);
    ASSERT_TRUE(embedding_result.is_ok()) << embedding_result.error().message();
    auto embedding = std::move(embedding_result.value());

    auto invalid_input = make_random_tensor_or_fail<f32>({3}, DeviceType::CPU, 6202);
    assert_tensor_created(invalid_input);

    auto output_result = embedding.forward(invalid_input.tensor);

    ASSERT_TRUE(output_result.is_err());
    EXPECT_EQ(output_result.error().code(), ErrorCode::DtypeMismatch);
}

TEST(EmbeddingLayerTest, ForwardRejectsNonVectorInput) {
    constexpr i32 vocab_size = 8;
    constexpr i32 embedding_dim = 4;

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto embed_scope_result = root.pp("embed");
    ASSERT_TRUE(embed_scope_result.is_ok()) << embed_scope_result.error().message();
    const auto embed_scope = std::move(embed_scope_result.value());
    insert_tensor_or_fail<f32>(vars,
                               scoped_name(embed_scope.prefix(), "weight"),
                               {vocab_size, embedding_dim},
                               DeviceType::CPU,
                               6301);

    auto embedding_result =
        layer::Embedding::embedding(vocab_size, embedding_dim, DeviceType::CPU, DataType::Float32, embed_scope);
    ASSERT_TRUE(embedding_result.is_ok()) << embedding_result.error().message();
    auto embedding = std::move(embedding_result.value());

    auto invalid_input = make_random_tensor_or_fail<i32>({2, 2}, DeviceType::CPU, 6302);
    assert_tensor_created(invalid_input);

    auto output_result = embedding.forward(invalid_input.tensor);

    ASSERT_TRUE(output_result.is_err());
    EXPECT_EQ(output_result.error().code(), ErrorCode::ShapeMismatch);
}

#ifdef PULSE_USE_CUDA
TEST(EmbeddingLayerTest, ForwardMatchesWeightRowsOnCudaBFloat16) {
    constexpr i32 vocab_size = 16;
    constexpr i32 embedding_dim = 8;

    auto weight = make_random_tensor_or_fail<bf16>({vocab_size, embedding_dim}, DeviceType::CUDA, 6401);
    if (weight.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    auto input = make_random_tensor_or_fail<i32>({4}, DeviceType::CUDA, 6402);
    if (input.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    VarMap vars;
    const auto root = VarBuilder::from_var_map(vars);
    auto embed_scope_result = root.pp("embed");
    ASSERT_TRUE(embed_scope_result.is_ok()) << embed_scope_result.error().message();
    const auto embed_scope = std::move(embed_scope_result.value());
    auto insert_result = vars.insert(scoped_name(embed_scope.prefix(), "weight"), std::move(weight.tensor));
    ASSERT_TRUE(insert_result.is_ok()) << insert_result.error().message();

    auto embedding_result =
        layer::Embedding::embedding(vocab_size, embedding_dim, DeviceType::CUDA, DataType::BFloat16, embed_scope);
    ASSERT_TRUE(embedding_result.is_ok()) << embedding_result.error().message();
    auto embedding = std::move(embedding_result.value());

    ASSERT_NE(embedding.weight(), nullptr);
    auto weight_values = tensor_values_or_fail<bf16>(*embedding.weight());
    auto input_values = tensor_values_or_fail<i32>(input.tensor);
    auto expected = embedding_expected_as_float(input_values, weight_values, vocab_size, embedding_dim);

    auto output_result = embedding.forward(input.tensor);

    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    auto output_host_result = output_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(output_host_result.is_ok()) << output_host_result.error().message();
    Tensor output_host = std::move(output_host_result.value());

    expect_values_near(output_host.ptr<bf16>(), expected);
}
#endif
