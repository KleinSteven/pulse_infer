#include "pulse/logging.hpp"

#include <memory>

#include "spdlog/logger.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace pulse::logging::detail {
namespace {

constexpr auto kLoggerName = "pulse_infer";

std::shared_ptr<spdlog::logger> make_logger() {
    if (auto existing = spdlog::get(kLoggerName)) {
        return existing;
    }

    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>(kLoggerName, std::move(sink));
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::err);
    spdlog::register_logger(logger);
    return logger;
}

}  // namespace

spdlog::logger& logger() {
    static std::shared_ptr<spdlog::logger> pulse_logger = make_logger();
    return *pulse_logger;
}

}  // namespace pulse::logging::detail
