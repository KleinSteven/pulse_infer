#pragma once

#include <concepts>
#include <cstdint>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "spdlog/spdlog.h"

namespace pulse::logging {

namespace detail {

spdlog::logger& logger();

}  // namespace detail

enum class Level : std::int8_t {
    trace,
    debug,
    info,
    warn,
    err,
    critical,
    off,
};

constexpr spdlog::level::level_enum to_spdlog_level(Level level) noexcept {
    switch (level) {
        case Level::trace:
            return spdlog::level::trace;
        case Level::debug:
            return spdlog::level::debug;
        case Level::info:
            return spdlog::level::info;
        case Level::warn:
            return spdlog::level::warn;
        case Level::err:
            return spdlog::level::err;
        case Level::critical:
            return spdlog::level::critical;
        case Level::off:
            return spdlog::level::off;
    }
    return spdlog::level::off;
}

template<typename... Args>
struct logging_rformat {
    template<std::convertible_to<std::string_view> StrLike>
    consteval logging_rformat(const StrLike& string,
                              std::source_location loc = std::source_location::current())
        : str(string), location(loc) {}

    std::format_string<Args...> str;
    std::source_location location;
};

template<typename... Args>
void log(Level level, std::source_location location, std::format_string<Args...> fmt, Args&&... args) {
    const auto spd_level = to_spdlog_level(level);
    auto& pulse_logger = detail::logger();
    if (!pulse_logger.should_log(spd_level))
        return;

    spdlog::source_loc loc{
        location.file_name(),
        static_cast<int>(location.line()),
        location.function_name(),
    };

    using spdlog_fmt = spdlog::format_string_t<Args...>;

    if constexpr (std::same_as<spdlog_fmt, std::string_view>) {
        pulse_logger.log(loc, spd_level, fmt.get(), std::forward<Args>(args)...);
    } else {
        pulse_logger.log(loc, spd_level, fmt, std::forward<Args>(args)...);
    }
}

inline void set_level(Level level) {
    detail::logger().set_level(to_spdlog_level(level));
}

inline void set_pattern(std::string pattern) {
    detail::logger().set_pattern(std::move(pattern));
}

}  // namespace pulse::logging

namespace pulse {

template<typename... Args>
using logging_format = logging::logging_rformat<std::type_identity_t<Args>...>;

using Level = logging::Level;

template<typename... Args>
void info(logging_format<Args...> fmt, Args&&... args) {
    pulse::logging::log(logging::Level::info, fmt.location, fmt.str, std::forward<Args>(args)...);
}

template<typename... Args>
void trace(logging_format<Args...> fmt, Args&&... args) {
    pulse::logging::log(logging::Level::trace, fmt.location, fmt.str, std::forward<Args>(args)...);
}

template<typename... Args>
void debug(logging_format<Args...> fmt, Args&&... args) {
    pulse::logging::log(logging::Level::debug, fmt.location, fmt.str, std::forward<Args>(args)...);
}

template<typename... Args>
void warn(logging_format<Args...> fmt, Args&&... args) {
    pulse::logging::log(logging::Level::warn, fmt.location, fmt.str, std::forward<Args>(args)...);
}

template<typename... Args>
void error(logging_format<Args...> fmt, Args&&... args) {
    pulse::logging::log(logging::Level::err, fmt.location, fmt.str, std::forward<Args>(args)...);
}

template<typename... Args>
void fatal(logging_format<Args...> fmt, Args&&... args) {
    pulse::logging::log(logging::Level::critical, fmt.location, fmt.str, std::forward<Args>(args)...);
}

}  // namespace pulse
