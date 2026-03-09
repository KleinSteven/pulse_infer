#include "pulse_infer/core/config.h"

namespace pulse_infer {
void test_config::test_config_print() {
  fmt::println("version = {}, test = {}", version, test);
}
} // namespace pulse_infer