#include "pulse_infer/core/config.h"

#include "pulse_infer/logging.h"

namespace pulse_infer {

void test_config::test_config_print() {
    constexpr static auto pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] [%s:%#] %v";
    pulse_infer::logging::set_level(Level::trace);
    pulse_infer::logging::set_pattern(pattern);
    pulse_infer::info("version = {}, test = {}", version, test);
    pulse_infer::trace("version = {}, test = {}", version, test);
    pulse_infer::debug("version = {}, test = {}", version, test);
    pulse_infer::warn("version = {}, test = {}", version, test);
    pulse_infer::error("version = {}, test = {}", version, test);
    pulse_infer::fatal("version = {}, test = {}", version, test);
}

}  // namespace pulse_infer
