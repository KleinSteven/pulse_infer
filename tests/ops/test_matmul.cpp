#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "pulse/core/tensor.hpp"
#include "pulse/ops/matmul.hpp"

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

template<>
std::vector<i32> make_random_values(usize count, u32 seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<i32> dist(-10, 10);
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
std::vector<T> matmul_expected(const T* lhs,
                               const T* rhs,
                               i32 m,
                               i32 n,
                               i32 k,
                               bool transpose_lhs = false,
                               bool transpose_rhs = false);

template<typename T>
std::vector<T> matmul_expected(const std::vector<T>& lhs,
                               const std::vector<T>& rhs,
                               i32 m,
                               i32 n,
                               i32 k,
                               bool transpose_lhs = false,
                               bool transpose_rhs = false);

template<typename T>
std::vector<T> matmul_expected(const std::vector<T>& lhs,
                               const std::vector<T>& rhs,
                               i32 m,
                               i32 n,
                               i32 k,
                               bool transpose_lhs,
                               bool transpose_rhs) {
    return matmul_expected(lhs.data(), rhs.data(), m, n, k, transpose_lhs, transpose_rhs);
}

template<typename T>
std::vector<T> matmul_expected(const T* lhs,
                               const T* rhs,
                               i32 m,
                               i32 n,
                               i32 k,
                               bool transpose_lhs,
                               bool transpose_rhs) {
    const i32 lhs_stride = transpose_lhs ? m : k;
    const i32 rhs_stride = transpose_rhs ? k : n;
    std::vector<T> result(static_cast<usize>(m) * static_cast<usize>(n), static_cast<T>(0));
    for (i32 row = 0; row < m; ++row) {
        for (i32 col = 0; col < n; ++col) {
            T sum = static_cast<T>(0);
            for (i32 inner = 0; inner < k; ++inner) {
                const i32 lhs_row = transpose_lhs ? inner : row;
                const i32 lhs_col = transpose_lhs ? row : inner;
                const i32 rhs_row = transpose_rhs ? col : inner;
                const i32 rhs_col = transpose_rhs ? inner : col;

                sum += lhs[static_cast<usize>(lhs_row) * static_cast<usize>(lhs_stride) +
                           static_cast<usize>(lhs_col)] *
                       rhs[static_cast<usize>(rhs_row) * static_cast<usize>(rhs_stride) +
                           static_cast<usize>(rhs_col)];
            }
            result[static_cast<usize>(row) * static_cast<usize>(n) + static_cast<usize>(col)] = sum;
        }
    }
    return result;
}

template<typename T>
void expect_values_near(const T* actual, const std::vector<T>& expected, double tolerance) {
    ASSERT_NE(actual, nullptr);

    for (usize i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_same_v<T, f32>) {
            EXPECT_NEAR(actual[i], expected[i], tolerance);
        } else if constexpr (std::is_same_v<T, f64>) {
            EXPECT_NEAR(actual[i], expected[i], tolerance);
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

template<typename T>
std::vector<float> matmul_expected_as_float(const T* lhs,
                                            const T* rhs,
                                            i32 m,
                                            i32 n,
                                            i32 k,
                                            bool transpose_lhs = false,
                                            bool transpose_rhs = false);

template<typename T>
std::vector<float> matmul_expected_as_float(const std::vector<T>& lhs,
                                            const std::vector<T>& rhs,
                                            i32 m,
                                            i32 n,
                                            i32 k,
                                            bool transpose_lhs = false,
                                            bool transpose_rhs = false);

template<>
float quantize_float_to_output<f16>(float value) {
    return __half2float(__float2half(value));
}

template<>
float quantize_float_to_output<bf16>(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

template<typename T>
std::vector<float> matmul_expected_as_float(const std::vector<T>& lhs,
                                            const std::vector<T>& rhs,
                                            i32 m,
                                            i32 n,
                                            i32 k,
                                            bool transpose_lhs,
                                            bool transpose_rhs) {
    return matmul_expected_as_float(lhs.data(), rhs.data(), m, n, k, transpose_lhs, transpose_rhs);
}

template<typename T>
std::vector<float> matmul_expected_as_float(const T* lhs,
                                            const T* rhs,
                                            i32 m,
                                            i32 n,
                                            i32 k,
                                            bool transpose_lhs,
                                            bool transpose_rhs) {
    const i32 lhs_stride = transpose_lhs ? m : k;
    const i32 rhs_stride = transpose_rhs ? k : n;
    std::vector<float> result(static_cast<usize>(m) * static_cast<usize>(n), 0.0f);
    for (i32 row = 0; row < m; ++row) {
        for (i32 col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (i32 inner = 0; inner < k; ++inner) {
                const i32 lhs_row = transpose_lhs ? inner : row;
                const i32 lhs_col = transpose_lhs ? row : inner;
                const i32 rhs_row = transpose_rhs ? col : inner;
                const i32 rhs_col = transpose_rhs ? inner : col;

                sum += scalar_to_float(lhs[static_cast<usize>(lhs_row) * static_cast<usize>(lhs_stride) +
                                           static_cast<usize>(lhs_col)]) *
                       scalar_to_float(rhs[static_cast<usize>(rhs_row) * static_cast<usize>(rhs_stride) +
                                           static_cast<usize>(rhs_col)]);
            }
            result[static_cast<usize>(row) * static_cast<usize>(n) + static_cast<usize>(col)] =
                quantize_float_to_output<T>(sum);
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

TEST(TensorMatmulTest, MultipliesFloat32MatricesAndReturnsNewTensor) {
    const std::vector<i32> lhs_dims{2, 3};
    const std::vector<i32> rhs_dims{3, 4};
    auto input1 = make_random_tensor_or_fail<f32>(lhs_dims, DeviceType::CPU, 101);
    auto input2 = make_random_tensor_or_fail<f32>(rhs_dims, DeviceType::CPU, 202);
    assert_tensor_created(input1);
    assert_tensor_created(input2);
    auto expected = matmul_expected(input1.values, input2.values, 2, 4, 3);

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();
    Tensor output = std::move(matmul_result.value());

    EXPECT_EQ(output.device(), DeviceType::CPU);
    EXPECT_EQ(output.dtype(), DataType::Float32);
    EXPECT_EQ(output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(output.ptr<f32>(), expected, 1e-5);
}

TEST(TensorMatmulTest, MultipliesFloat64MatricesAndReturnsNewTensor) {
    const std::vector<i32> lhs_dims{3, 2};
    const std::vector<i32> rhs_dims{2, 5};
    auto input1 = make_random_tensor_or_fail<f64>(lhs_dims, DeviceType::CPU, 303);
    auto input2 = make_random_tensor_or_fail<f64>(rhs_dims, DeviceType::CPU, 404);
    assert_tensor_created(input1);
    assert_tensor_created(input2);
    auto expected = matmul_expected(input1.values, input2.values, 3, 5, 2);

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();
    Tensor output = std::move(matmul_result.value());

    EXPECT_EQ(output.dims(), (std::vector<i32>{3, 5}));
    expect_values_near(output.ptr<f64>(), expected, 1e-10);
}

TEST(TensorMatmulTest, MultipliesLargeFloat32MatricesAndReturnsNewTensor) {
    const std::vector<i32> lhs_dims{64, 128};
    const std::vector<i32> rhs_dims{128, 96};
    auto input1 = make_random_tensor_or_fail<f32>(lhs_dims, DeviceType::CPU, 313);
    auto input2 = make_random_tensor_or_fail<f32>(rhs_dims, DeviceType::CPU, 414);
    assert_tensor_created(input1);
    assert_tensor_created(input2);
    auto expected = matmul_expected(input1.values, input2.values, 64, 96, 128);

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();
    Tensor output = std::move(matmul_result.value());

    EXPECT_EQ(output.dims(), (std::vector<i32>{64, 96}));
    expect_values_near(output.ptr<f32>(), expected, 1e-4);
}

TEST(TensorMatmulTest, OpsMatmulSupportsTransposeLeftOnCpu) {
    const std::vector<i32> lhs_dims{3, 2};
    const std::vector<i32> rhs_dims{3, 4};
    auto input1 = make_random_tensor_or_fail<f32>(lhs_dims, DeviceType::CPU, 451);
    auto input2 = make_random_tensor_or_fail<f32>(rhs_dims, DeviceType::CPU, 452);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto output_result = Tensor::create({2, 4}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output = std::move(output_result.value());
    auto expected = matmul_expected(input1.values, input2.values, 2, 4, 3, true, false);

    auto matmul_result = ops::matmul(input1.tensor, input2.tensor, output, true, false);

    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();
    EXPECT_EQ(output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(output.ptr<f32>(), expected, 1e-5);
}

TEST(TensorMatmulTest, OpsMatmulSupportsTransposeRightOnCpu) {
    const std::vector<i32> lhs_dims{2, 3};
    const std::vector<i32> rhs_dims{4, 3};
    auto input1 = make_random_tensor_or_fail<f64>(lhs_dims, DeviceType::CPU, 461);
    auto input2 = make_random_tensor_or_fail<f64>(rhs_dims, DeviceType::CPU, 462);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto output_result = Tensor::create({2, 4}, DataType::Float64, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output = std::move(output_result.value());
    auto expected = matmul_expected(input1.values, input2.values, 2, 4, 3, false, true);

    auto matmul_result = ops::matmul(input1.tensor, input2.tensor, output, false, true);

    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();
    EXPECT_EQ(output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(output.ptr<f64>(), expected, 1e-10);
}

TEST(TensorMatmulTest, OpsMatmulSupportsTransposeBothOnCpu) {
    const std::vector<i32> lhs_dims{3, 2};
    const std::vector<i32> rhs_dims{4, 3};
    auto input1 = make_random_tensor_or_fail<f32>(lhs_dims, DeviceType::CPU, 471);
    auto input2 = make_random_tensor_or_fail<f32>(rhs_dims, DeviceType::CPU, 472);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto output_result = Tensor::create({2, 4}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output = std::move(output_result.value());
    auto expected = matmul_expected(input1.values, input2.values, 2, 4, 3, true, true);

    auto matmul_result = ops::matmul(input1.tensor, input2.tensor, output, true, true);

    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();
    EXPECT_EQ(output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(output.ptr<f32>(), expected, 1e-5);
}

TEST(TensorMatmulTest, RejectsIncompatibleInnerDimensions) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 505);
    auto input2 = make_random_tensor_or_fail<f32>({4, 2}, DeviceType::CPU, 606);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(TensorMatmulTest, RejectsNonMatrixInput) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 2, 3}, DeviceType::CPU, 707);
    auto input2 = make_random_tensor_or_fail<f32>({3, 4}, DeviceType::CPU, 808);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorMatmulTest, OpsMatmulRejectsMismatchedOutputShape) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 909);
    auto input2 = make_random_tensor_or_fail<f32>({3, 4}, DeviceType::CPU, 1001);
    auto output = make_random_tensor_or_fail<f32>({4, 2}, DeviceType::CPU, 1101);
    assert_tensor_created(input1);
    assert_tensor_created(input2);
    assert_tensor_created(output);

    auto matmul_result = ops::matmul(input1.tensor, input2.tensor, output.tensor);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(TensorMatmulTest, OpsMatmulRejectsIncompatibleInnerDimensionsAfterTranspose) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 1121);
    auto input2 = make_random_tensor_or_fail<f32>({5, 4}, DeviceType::CPU, 1122);
    auto output_result = Tensor::create({3, 5}, DataType::Float32, DeviceType::CPU);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output = std::move(output_result.value());
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto matmul_result = ops::matmul(input1.tensor, input2.tensor, output, true, false);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(TensorMatmulTest, RejectsMismatchedDtypes) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 1201);
    auto input2 = make_random_tensor_or_fail<i32>({3, 4}, DeviceType::CPU, 1301);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::DtypeMismatch);
}

TEST(TensorMatmulTest, RejectsEmptySelfTensor) {
    Tensor input1;
    auto input2 = make_random_tensor_or_fail<f32>({3, 4}, DeviceType::CPU, 1401);
    assert_tensor_created(input2);

    auto matmul_result = input1.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorMatmulTest, RejectsEmptyOtherTensor) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 1501);
    assert_tensor_created(input1);
    Tensor input2;

    auto matmul_result = input1.tensor.matmul(input2);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorMatmulTest, RejectsUnsupportedCpuDtype) {
    auto input1 = make_random_tensor_or_fail<i32>({2, 3}, DeviceType::CPU, 1601);
    auto input2 = make_random_tensor_or_fail<i32>({3, 4}, DeviceType::CPU, 1701);
    assert_tensor_created(input1);
    assert_tensor_created(input2);

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::NotImplemented);
}

#ifdef PULSE_USE_CUDA

TEST(TensorMatmulTest, RejectsDeviceMismatch) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CPU, 1801);
    assert_tensor_created(input1);
    auto input2 = make_random_tensor_or_fail<f32>({3, 4}, DeviceType::CUDA, 1901);
    if (input2.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::DeviceMismatch);
}

TEST(TensorMatmulTest, MultipliesFloat32CudaMatricesUsingCublas) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CUDA, 2001);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f32>({3, 4}, DeviceType::CUDA, 2101);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = matmul_expected(input1.values, input2.values, 2, 4, 3);

    auto matmul_result = input1.tensor.matmul(input2.tensor);
    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();

    auto host_result = matmul_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(host_output.ptr<f32>(), expected, 1e-4);
}

TEST(TensorMatmulTest, OpsMatmulSupportsTransposeRightOnCuda) {
    auto input1 = make_random_tensor_or_fail<f32>({2, 3}, DeviceType::CUDA, 2021);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f32>({4, 3}, DeviceType::CUDA, 2022);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = matmul_expected(input1.values, input2.values, 2, 4, 3, false, true);

    auto output_result = Tensor::create({2, 4}, DataType::Float32, DeviceType::CUDA);
    ASSERT_TRUE(output_result.is_ok()) << output_result.error().message();
    Tensor output = std::move(output_result.value());

    auto matmul_result = ops::matmul(input1.tensor, input2.tensor, output, false, true);
    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();

    auto host_result = output.to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(host_output.ptr<f32>(), expected, 1e-4);
}

TEST(TensorMatmulTest, MultipliesLargeFloat32CudaMatricesUsingCublas) {
    auto input1 = make_random_tensor_or_fail<f32>({128, 192}, DeviceType::CUDA, 2051);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f32>({192, 64}, DeviceType::CUDA, 2151);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = matmul_expected(input1.values, input2.values, 128, 64, 192);

    auto matmul_result = input1.tensor.matmul(input2.tensor);
    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();

    auto host_result = matmul_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{128, 64}));
    expect_values_near(host_output.ptr<f32>(), expected, 2e-3);
}

TEST(TensorMatmulTest, MultipliesFloat64CudaMatricesUsingCublas) {
    auto input1 = make_random_tensor_or_fail<f64>({3, 2}, DeviceType::CUDA, 2201);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f64>({2, 4}, DeviceType::CUDA, 2301);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = matmul_expected(input1.values, input2.values, 3, 4, 2);

    auto matmul_result = input1.tensor.matmul(input2.tensor);
    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();

    auto host_result = matmul_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{3, 4}));
    expect_values_near(host_output.ptr<f64>(), expected, 1e-9);
}

TEST(TensorMatmulTest, MultipliesFloat16CudaMatricesUsingCublas) {
    auto input1 = make_random_tensor_or_fail<f16>({2, 3}, DeviceType::CUDA, 2401);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<f16>({3, 4}, DeviceType::CUDA, 2501);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = matmul_expected_as_float(input1.values, input2.values, 2, 4, 3);

    auto matmul_result = input1.tensor.matmul(input2.tensor);
    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();

    auto host_result = matmul_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(host_output.ptr<f16>(), expected, 1e-2f);
}

TEST(TensorMatmulTest, MultipliesBFloat16CudaMatricesUsingCublas) {
    auto input1 = make_random_tensor_or_fail<bf16>({2, 3}, DeviceType::CUDA, 2601);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<bf16>({3, 4}, DeviceType::CUDA, 2701);
    ASSERT_FALSE(input2.tensor.empty());
    auto expected = matmul_expected_as_float(input1.values, input2.values, 2, 4, 3);

    auto matmul_result = input1.tensor.matmul(input2.tensor);
    ASSERT_TRUE(matmul_result.is_ok()) << matmul_result.error().message();

    auto host_result = matmul_result.value().to(DeviceType::CPU);
    ASSERT_TRUE(host_result.is_ok()) << host_result.error().message();
    Tensor host_output = std::move(host_result.value());

    EXPECT_EQ(host_output.dims(), (std::vector<i32>{2, 4}));
    expect_values_near(host_output.ptr<bf16>(), expected, 2e-2f);
}

TEST(TensorMatmulTest, RejectsUnsupportedCudaDtype) {
    auto input1 = make_random_tensor_or_fail<i32>({2, 3}, DeviceType::CUDA, 2801);
    if (input1.tensor.empty()) {
        GTEST_SKIP() << "CUDA tensor creation failed";
    }
    auto input2 = make_random_tensor_or_fail<i32>({3, 4}, DeviceType::CUDA, 2901);
    ASSERT_FALSE(input2.tensor.empty());

    auto matmul_result = input1.tensor.matmul(input2.tensor);

    ASSERT_TRUE(matmul_result.is_err());
    EXPECT_EQ(matmul_result.error().code(), ErrorCode::NotImplemented);
}

#endif
