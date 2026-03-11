/**
 * @file test_main.cpp
 * @brief Main entry point for Google Test
 */

#include <gtest/gtest.h>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
