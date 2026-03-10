#pragma once

#include <string>

namespace pulse_infer {

struct test_config {
    std::string version = "0.1";
    bool test = true;

    void test_config_print();
};

}  // namespace pulse_infer
