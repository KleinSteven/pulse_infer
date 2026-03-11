#include <gtest/gtest.h>

#include "pulse/core/types.hpp"

using namespace pulse;

TEST(TypesTest, IntegerTypeSize) {
    EXPECT_EQ(sizeof(i8), 1);
    EXPECT_EQ(sizeof(i16), 2);
    EXPECT_EQ(sizeof(i32), 4);
    EXPECT_EQ(sizeof(i64), 8);

    EXPECT_EQ(sizeof(u8), 1);
    EXPECT_EQ(sizeof(u16), 2);
    EXPECT_EQ(sizeof(u32), 4);
    EXPECT_EQ(sizeof(u64), 8);
}

TEST(TypesTest, FloatTypeSize) {
    EXPECT_EQ(sizeof(f32), 4);
    EXPECT_EQ(sizeof(f64), 8);
}
