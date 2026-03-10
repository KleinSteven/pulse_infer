/**
 * @file test_main.cpp
 * @brief Main entry point for Google Test
 */

#include <gtest/gtest.h>

#include "pulse_infer/core/config.h"


TEST(UseTest, TestAdd) {
    EXPECT_EQ(1 + 2, 3);
}

TEST(UseTest, TestFmt) {
    pulse_infer::test_config tmp{};
    tmp.test_config_print();
    EXPECT_EQ(1 + 2, 3);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
