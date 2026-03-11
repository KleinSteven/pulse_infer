#include <gtest/gtest.h>

#include "pulse/core/error.hpp"

using namespace pulse;

TEST(ErrorTest, DefaultConstruction) {
    Error err;
    EXPECT_TRUE(err.is_ok());
    EXPECT_FALSE(err.is_err());
    EXPECT_EQ(err.code(), ErrorCode::Ok);
    EXPECT_TRUE(err.message().empty());
}

TEST(ErrorTest, ErrorCodeConstruction) {
    Error err(ErrorCode::Unknown);
    EXPECT_FALSE(err.is_ok());
    EXPECT_TRUE(err.is_err());
    EXPECT_EQ(err.code(), ErrorCode::Unknown);
}

TEST(ErrorTest, ErrorWithMessage) {
    Error err(ErrorCode::Unknown, "unknown error");
    EXPECT_TRUE(err.is_err());
    EXPECT_EQ(err.code(), ErrorCode::Unknown);
    EXPECT_EQ(err.message(), "unknown error");
}

TEST(ErrorTest, BoolConversion) {
    Error ok;
    Error err(ErrorCode::Unknown);

    EXPECT_TRUE(ok);
    EXPECT_FALSE(err);
}

TEST(ResultTest, OkValue) {
    Result<int> result = Ok(42);

    EXPECT_TRUE(result.is_ok());
    EXPECT_FALSE(result.is_err());
    EXPECT_TRUE(result);
    EXPECT_EQ(result.value(), 42);
}

TEST(ResultTest, ErrValue) {
    Result<int> result = Err<int>(ErrorCode::Unknown);

    EXPECT_FALSE(result.is_ok());
    EXPECT_TRUE(result.is_err());
    EXPECT_FALSE(result);
    EXPECT_EQ(result.error().code(), ErrorCode::Unknown);
}

TEST(ResultTest, ErrWithMessage) {
    Result<int> result = Err<int>(ErrorCode::Unknown, "unknown error");

    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.error().code(), ErrorCode::Unknown);
    EXPECT_EQ(result.error().message(), "unknown error");
}

TEST(ResultTest, ValueOr) {
    Result<int> ok_result = Ok(42);
    Result<int> err_result = Err<int>(ErrorCode::Unknown);

    EXPECT_EQ(ok_result.value_or(0), 42);
    EXPECT_EQ(err_result.value_or(0), 0);
}

TEST(ResultTest, MoveSemantics) {
    Result<std::string> result = Ok(std::string("hello"));

    EXPECT_TRUE(result.is_ok());
    std::string value = std::move(result).value();
    EXPECT_EQ(value, "hello");
}

TEST(ResultTest, VoidResult) {
    Result<void> ok_result = Ok();
    Result<void> err_result = Err<void>(ErrorCode::Unknown);

    EXPECT_TRUE(ok_result.is_ok());
    EXPECT_FALSE(ok_result.is_err());
    EXPECT_TRUE(ok_result);

    EXPECT_FALSE(err_result.is_ok());
    EXPECT_TRUE(err_result.is_err());
    EXPECT_FALSE(err_result);
    EXPECT_EQ(err_result.error().code(), ErrorCode::Unknown);
}
