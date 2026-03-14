#include <vector>

#include <gtest/gtest.h>

#include "pulse/core/var.hpp"

TEST(VarBuilderTest, NamedTensorCreation) {
    pulse::VarMap vars;
    const auto root = pulse::VarBuilder::from_var_map(vars);
    auto decoder_result = root.pp("decoder");
    ASSERT_TRUE(decoder_result.is_ok());
    auto block_result = decoder_result.value().pp("layer0");
    ASSERT_TRUE(block_result.is_ok());
    const auto block = std::move(block_result.value());

    auto weight_result =
        block.get_or_create("weight", {2, 3}, pulse::DataType::Float32, pulse::DeviceType::CPU);
    ASSERT_TRUE(weight_result.is_ok());

    auto* weight = weight_result.value();
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->dims(), std::vector<pulse::i32>({2, 3}));
    EXPECT_EQ(weight->dtype(), pulse::DataType::Float32);
    EXPECT_TRUE(vars.contains("decoder.layer0.weight"));

    auto reuse_result =
        block.get_or_create("weight", {2, 3}, pulse::DataType::Float32, pulse::DeviceType::CPU);
    ASSERT_TRUE(reuse_result.is_ok());
    EXPECT_EQ(reuse_result.value(), weight);

    auto const_result = block.get("weight", std::vector<pulse::i32>{2, 3}, pulse::DataType::Float32);
    ASSERT_TRUE(const_result.is_ok());
    EXPECT_EQ(const_result.value(), weight);
}

TEST(VarBuilderTest, ShapeMismatchIsRejected) {
    pulse::VarMap vars;
    auto builder_result = pulse::VarBuilder::from_var_map(vars).pp("encoder");
    ASSERT_TRUE(builder_result.is_ok());
    const auto builder = std::move(builder_result.value());

    auto first_result =
        builder.get_or_create("bias", {4}, pulse::DataType::Float32, pulse::DeviceType::CPU);
    ASSERT_TRUE(first_result.is_ok());

    auto mismatch_result =
        builder.get_or_create("bias", {8}, pulse::DataType::Float32, pulse::DeviceType::CPU);
    EXPECT_TRUE(mismatch_result.is_err());
}

TEST(VarBuilderTest, EmptyNamesAreRejectedBeforeJoining) {
    pulse::VarMap vars;
    const auto root = pulse::VarBuilder::from_var_map(vars);

    auto pp_result = root.pp("");
    EXPECT_TRUE(pp_result.is_err());

    auto get_result = root.pp("encoder");
    ASSERT_TRUE(get_result.is_ok());
    const auto builder = std::move(get_result.value());

    EXPECT_TRUE(builder.get("").is_err());
    EXPECT_TRUE(builder.get_or_create("", {1}, pulse::DataType::Float32, pulse::DeviceType::CPU).is_err());
}

TEST(VarBuilderTest, PpRejectsSegmentContainingDot) {
    pulse::VarMap vars;
    const auto root = pulse::VarBuilder::from_var_map(vars);

    auto pp_result = root.pp("encoder.layer0");
    EXPECT_TRUE(pp_result.is_err());
}

TEST(VarMapTest, FromSafeTensorsRejectsUnsupportedDeviceBeforeLoad) {
    const auto result =
        pulse::VarMap::from_safetensors("this_path_should_not_be_opened.safetensors", pulse::DeviceType::Unified);

    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), pulse::ErrorCode::InvalidArgument);
}
