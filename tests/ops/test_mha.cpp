#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/ops/mha.hpp"

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
TensorWithValues<T> make_tensor_or_fail(const std::vector<i32>& dims,
                                        const std::vector<T>& values,
                                        DeviceType device) {
    if (values.size() != numel(dims)) {
        return {};
    }

    auto tensor_result = Tensor::from_vector(values, device);
    if (!tensor_result.is_ok()) {
        return {};
    }

    Tensor tensor(std::move(tensor_result.value()));
    auto reshape_result = tensor.reshape(dims);
    if (!reshape_result.is_ok()) {
        return {};
    }

    return {std::move(tensor), values};
}

template<typename T>
void assert_tensor_created(const TensorWithValues<T>& tensor_with_values) {
    ASSERT_FALSE(tensor_with_values.tensor.empty());
}

template<typename T>
std::vector<double> mha_expected(const std::vector<T>& query,
                                 const std::vector<T>& key_cache,
                                 const std::vector<T>& value_cache,
                                 i32 pos,
                                 i32 kv_dim,
                                 i32 head_num,
                                 i32 head_size) {
    const i32 kv_head_num = kv_dim / head_size;
    const i32 kv_mul = head_num / kv_head_num;
    std::vector<double> output(static_cast<usize>(head_num) * static_cast<usize>(head_size), 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

    for (i32 head = 0; head < head_num; ++head) {
        const i32 q_offset = head * head_size;
        const i32 kv_offset = (head / kv_mul) * head_size;
        std::vector<double> scores(static_cast<usize>(pos + 1), 0.0);

        for (i32 token = 0; token <= pos; ++token) {
            double dot = 0.0;
            for (i32 dim = 0; dim < head_size; ++dim) {
                const auto q_index = static_cast<usize>(q_offset + dim);
                const auto kv_index =
                    static_cast<usize>(token) * static_cast<usize>(kv_dim) + static_cast<usize>(kv_offset + dim);
                dot += static_cast<double>(query[q_index]) * static_cast<double>(key_cache[kv_index]);
            }
            scores[static_cast<usize>(token)] = dot * scale;
        }

        const double max_score = *std::max_element(scores.begin(), scores.end());
        double sum_exp = 0.0;
        for (double& score : scores) {
            score = std::exp(score - max_score);
            sum_exp += score;
        }
        for (double& score : scores) {
            score /= sum_exp;
        }

        for (i32 token = 0; token <= pos; ++token) {
            const double weight = scores[static_cast<usize>(token)];
            for (i32 dim = 0; dim < head_size; ++dim) {
                const auto out_index = static_cast<usize>(q_offset + dim);
                const auto value_index =
                    static_cast<usize>(token) * static_cast<usize>(kv_dim) + static_cast<usize>(kv_offset + dim);
                output[out_index] += weight * static_cast<double>(value_cache[value_index]);
            }
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

TEST(MHAOpTest, OpsMhaAppliesSingleHeadCpuAtPositionZero) {
    constexpr i32 head_num = 1;
    constexpr i32 head_size = 4;
    constexpr i32 dim = head_num * head_size;
    constexpr i32 kv_dim = dim;
    constexpr i32 seq_len = 4;
    constexpr i32 pos = 0;

    auto query = make_tensor_or_fail<f32>({dim}, {1.0f, 0.0f, 0.0f, 0.0f}, DeviceType::CPU);
    auto key_cache =
        make_tensor_or_fail<f32>({seq_len, kv_dim},
                                 {1.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f},
                                 DeviceType::CPU);
    auto value_cache =
        make_tensor_or_fail<f32>({seq_len, kv_dim},
                                 {2.0f, 3.0f, 4.0f, 5.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f},
                                 DeviceType::CPU);
    assert_tensor_created(query);
    assert_tensor_created(key_cache);
    assert_tensor_created(value_cache);
    auto output_result = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto mha_result = ops::mha(query.tensor, key_cache.tensor, value_cache.tensor, output, pos, head_num, head_size);

    ASSERT_TRUE(mha_result.is_ok()) << mha_result.error().message();
    EXPECT_EQ(output.dims(), (std::vector<i32>{dim}));
    EXPECT_NEAR(output.ptr<f32>()[0], 2.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[1], 3.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[2], 4.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[3], 5.0f, 1e-5f);
}

TEST(MHAOpTest, OpsMhaAppliesTwoHeadsCpu) {
    constexpr i32 head_num = 2;
    constexpr i32 head_size = 2;
    constexpr i32 dim = head_num * head_size;
    constexpr i32 kv_dim = dim;
    constexpr i32 seq_len = 3;
    constexpr i32 pos = 1;

    const std::vector<f32> query_values{1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<f32> key_values{
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const std::vector<f32> value_values{
        2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };

    auto query = make_tensor_or_fail<f32>({dim}, query_values, DeviceType::CPU);
    auto key_cache = make_tensor_or_fail<f32>({seq_len, kv_dim}, key_values, DeviceType::CPU);
    auto value_cache = make_tensor_or_fail<f32>({seq_len, kv_dim}, value_values, DeviceType::CPU);
    assert_tensor_created(query);
    assert_tensor_created(key_cache);
    assert_tensor_created(value_cache);
    auto output_result = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));
    const auto expected =
        mha_expected(query_values, key_values, value_values, pos, kv_dim, head_num, head_size);

    auto mha_result = ops::mha(query.tensor, key_cache.tensor, value_cache.tensor, output, pos, head_num, head_size);

    ASSERT_TRUE(mha_result.is_ok()) << mha_result.error().message();
    expect_values_near(output.ptr<f32>(), expected, 1e-5);
}

TEST(MHAOpTest, OpsMhaSupportsGqaCpu) {
    constexpr i32 head_num = 4;
    constexpr i32 head_size = 2;
    constexpr i32 dim = head_num * head_size;
    constexpr i32 kv_dim = 4;
    constexpr i32 seq_len = 2;
    constexpr i32 pos = 0;

    const std::vector<f32> query_values{
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        -1.0f, 1.0f,
    };
    const std::vector<f32> key_values{
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
    };
    const std::vector<f32> value_values{
        10.0f, 20.0f, 30.0f, 40.0f,
        50.0f, 60.0f, 70.0f, 80.0f,
    };

    auto query = make_tensor_or_fail<f32>({dim}, query_values, DeviceType::CPU);
    auto key_cache = make_tensor_or_fail<f32>({seq_len, kv_dim}, key_values, DeviceType::CPU);
    auto value_cache = make_tensor_or_fail<f32>({seq_len, kv_dim}, value_values, DeviceType::CPU);
    assert_tensor_created(query);
    assert_tensor_created(key_cache);
    assert_tensor_created(value_cache);
    auto output_result = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto mha_result = ops::mha(query.tensor, key_cache.tensor, value_cache.tensor, output, pos, head_num, head_size);

    ASSERT_TRUE(mha_result.is_ok()) << mha_result.error().message();
    EXPECT_NEAR(output.ptr<f32>()[0], 10.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[1], 20.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[2], 10.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[3], 20.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[4], 30.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[5], 40.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[6], 30.0f, 1e-5f);
    EXPECT_NEAR(output.ptr<f32>()[7], 40.0f, 1e-5f);
}

TEST(MHAOpTest, RejectsQuerySizeThatDoesNotMatchHeadLayout) {
    auto query = make_tensor_or_fail<f32>({3}, {1.0f, 2.0f, 3.0f}, DeviceType::CPU);
    auto key_cache =
        make_tensor_or_fail<f32>({2, 4}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}, DeviceType::CPU);
    auto value_cache =
        make_tensor_or_fail<f32>({2, 4}, {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, DeviceType::CPU);
    assert_tensor_created(query);
    assert_tensor_created(key_cache);
    assert_tensor_created(value_cache);
    auto output_result = Tensor::create({3}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));

    auto mha_result = ops::mha(query.tensor, key_cache.tensor, value_cache.tensor, output, 0, 2, 2);

    ASSERT_TRUE(mha_result.is_err());
    EXPECT_EQ(mha_result.error().code(), ErrorCode::ShapeMismatch);
}

#ifdef PULSE_USE_CUDA

template<typename T>
std::vector<T> cuda_float_vector(const std::vector<f32>& values);

template<>
std::vector<f32> cuda_float_vector<f32>(const std::vector<f32>& values) {
    return values;
}

template<>
std::vector<f16> cuda_float_vector<f16>(const std::vector<f32>& values) {
    std::vector<f16> converted;
    converted.reserve(values.size());
    for (f32 value : values) {
        converted.push_back(__float2half(value));
    }
    return converted;
}

template<>
std::vector<bf16> cuda_float_vector<bf16>(const std::vector<f32>& values) {
    std::vector<bf16> converted;
    converted.reserve(values.size());
    for (f32 value : values) {
        converted.push_back(__float2bfloat16(value));
    }
    return converted;
}

template<typename T>
void expect_cuda_host_values_near(const T* actual, const std::vector<double>& expected, double tolerance) {
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

TEST(MHAOpTest, OpsMhaAppliesFloat32Cuda) {
    constexpr i32 head_num = 2;
    constexpr i32 head_size = 2;
    constexpr i32 dim = head_num * head_size;
    constexpr i32 kv_dim = dim;
    constexpr i32 seq_len = 3;
    constexpr i32 pos = 1;

    const std::vector<f32> query_values{1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<f32> key_values{
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const std::vector<f32> value_values{
        2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };

    auto query = make_tensor_or_fail<f32>({dim}, query_values, DeviceType::CUDA);
    if (query.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto key_cache = make_tensor_or_fail<f32>({seq_len, kv_dim}, key_values, DeviceType::CUDA);
    ASSERT_FALSE(key_cache.tensor.empty());
    auto value_cache = make_tensor_or_fail<f32>({seq_len, kv_dim}, value_values, DeviceType::CUDA);
    ASSERT_FALSE(value_cache.tensor.empty());
    auto output_result = Tensor::create({dim}, DataType::Float32, DeviceType::CUDA);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));
    const auto expected =
        mha_expected(query_values, key_values, value_values, pos, kv_dim, head_num, head_size);

    auto mha_result = ops::mha(query.tensor, key_cache.tensor, value_cache.tensor, output, pos, head_num, head_size);

    ASSERT_TRUE(mha_result.is_ok()) << mha_result.error().message();
    auto host_result = output.to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());
    expect_values_near(host_output.ptr<f32>(), expected, 1e-4);
}

TEST(MHAOpTest, OpsMhaAppliesFloat16Cuda) {
    constexpr i32 head_num = 2;
    constexpr i32 head_size = 2;
    constexpr i32 dim = head_num * head_size;
    constexpr i32 kv_dim = dim;
    constexpr i32 seq_len = 3;
    constexpr i32 pos = 1;

    const std::vector<f32> query_values{1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<f32> key_values{
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const std::vector<f32> value_values{
        2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };

    auto query = make_tensor_or_fail<f16>({dim}, cuda_float_vector<f16>(query_values), DeviceType::CUDA);
    if (query.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto key_cache =
        make_tensor_or_fail<f16>({seq_len, kv_dim}, cuda_float_vector<f16>(key_values), DeviceType::CUDA);
    ASSERT_FALSE(key_cache.tensor.empty());
    auto value_cache =
        make_tensor_or_fail<f16>({seq_len, kv_dim}, cuda_float_vector<f16>(value_values), DeviceType::CUDA);
    ASSERT_FALSE(value_cache.tensor.empty());
    auto output_result = Tensor::create({dim}, DataType::Float16, DeviceType::CUDA);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));
    const auto expected =
        mha_expected(query_values, key_values, value_values, pos, kv_dim, head_num, head_size);

    auto mha_result = ops::mha(query.tensor, key_cache.tensor, value_cache.tensor, output, pos, head_num, head_size);

    ASSERT_TRUE(mha_result.is_ok()) << mha_result.error().message();
    auto host_result = output.to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());
    expect_cuda_host_values_near(host_output.ptr<f16>(), expected, 2e-2);
}

TEST(MHAOpTest, OpsMhaAppliesBFloat16Cuda) {
    constexpr i32 head_num = 2;
    constexpr i32 head_size = 2;
    constexpr i32 dim = head_num * head_size;
    constexpr i32 kv_dim = dim;
    constexpr i32 seq_len = 3;
    constexpr i32 pos = 1;

    const std::vector<f32> query_values{1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<f32> key_values{
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const std::vector<f32> value_values{
        2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };

    auto query = make_tensor_or_fail<bf16>({dim}, cuda_float_vector<bf16>(query_values), DeviceType::CUDA);
    if (query.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto key_cache =
        make_tensor_or_fail<bf16>({seq_len, kv_dim}, cuda_float_vector<bf16>(key_values), DeviceType::CUDA);
    ASSERT_FALSE(key_cache.tensor.empty());
    auto value_cache =
        make_tensor_or_fail<bf16>({seq_len, kv_dim}, cuda_float_vector<bf16>(value_values), DeviceType::CUDA);
    ASSERT_FALSE(value_cache.tensor.empty());
    auto output_result = Tensor::create({dim}, DataType::BFloat16, DeviceType::CUDA);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output(std::move(output_result.value()));
    const auto expected =
        mha_expected(query_values, key_values, value_values, pos, kv_dim, head_num, head_size);

    auto mha_result = ops::mha(query.tensor, key_cache.tensor, value_cache.tensor, output, pos, head_num, head_size);

    ASSERT_TRUE(mha_result.is_ok()) << mha_result.error().message();
    auto host_result = output.to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());
    expect_cuda_host_values_near(host_output.ptr<bf16>(), expected, 5e-2);
}

#endif
