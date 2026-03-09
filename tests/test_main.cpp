/**
 * @file test_main.cpp
 * @brief Main entry point for Google Test
 */

#include <gtest/gtest.h>

#include "fmt/core.h"

#include "pulse_infer/core/config.h"

int test_fmt(int in) {
  fmt::println("in = {}", in);
  return in;
}

TEST(UseTest, TestAdd) { EXPECT_EQ(1 + 2, 3); }

TEST(UseTest, TestFmt) {
  EXPECT_EQ(1 + 2, 3);
  pulse_infer::test_config tmp{};
  tmp.test_config_print();
  EXPECT_EQ(3, test_fmt(3));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
