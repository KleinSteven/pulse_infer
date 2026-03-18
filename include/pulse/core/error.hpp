#pragma once

#include <exception>
#include <string>
#include <variant>

#include "pulse/core/types.hpp"

namespace pulse {

enum class ErrorCode : u32 {
    // Success
    Ok = 0,

    // Memory errors (1-99)
    OutOfMemory = 1,
    NullPointer = 2,

    // I/O errors (100-199)
    OpenFileError = 100,
    GetFileSizeError = 101,
    MmapError = 102,

    // Tensor errors (200-299)
    ShapeMismatch = 200,
    DeviceMismatch = 201,
    DtypeMismatch = 202,

    // Model errors (300-399)


    // Operator errors (400-499)
    InvalidOperator = 400,

    // CUDA errors (500-599)
    CudaError = 500,
    CudaOutOfMemory = 501,


    // Generic errors (900-999)
    Unknown = 900,
    InvalidArgument = 901,
    NotImplemented = 902,

};

class Error {
public:
    Error() noexcept = default;

    explicit Error(ErrorCode code) noexcept : code_(code) {}

    Error(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

    [[nodiscard]] constexpr ErrorCode code() const noexcept {
        return code_;
    }

    [[nodiscard]] const std::string& message() const noexcept {
        return message_;
    }

    [[nodiscard]] constexpr bool is_ok() const noexcept {
        return code_ == ErrorCode::Ok;
    }

    [[nodiscard]] constexpr bool is_err() const noexcept {
        return code_ != ErrorCode::Ok;
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return is_ok();
    }

private:
    ErrorCode code_{ErrorCode::Ok};
    std::string message_;
};

template<typename T, typename E = Error>
class [[nodiscard]] Result {
public:
    constexpr Result(T value) : storage_(std::move(value)) {}

    constexpr Result(E error) : storage_(std::move(error)) {}

    [[nodiscard]] constexpr bool is_ok() const noexcept {
        return std::holds_alternative<T>(storage_);
    }

    [[nodiscard]] constexpr bool is_err() const noexcept {
        return std::holds_alternative<E>(storage_);
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return is_ok();
    }

    [[nodiscard]] constexpr T& value() & {
        return std::get<T>(storage_);
    }

    [[nodiscard]] constexpr const T& value() const& {
        return std::get<T>(storage_);
    }

    [[nodiscard]] constexpr T&& value() && {
        return std::get<T>(std::move(storage_));
    }

    [[nodiscard]] constexpr E& error() & {
        return std::get<E>(storage_);
    }

    [[nodiscard]] constexpr const E& error() const& {
        return std::get<E>(storage_);
    }

    [[nodiscard]] constexpr E&& error() && {
        return std::get<E>(std::move(storage_));
    }

    template<typename U>
    [[nodiscard]] constexpr T value_or(U&& defalut_value) const& {
        return is_ok() ? value() : static_cast<T>(std::forward<U>(defalut_value));
    }

    template<typename U>
    [[nodiscard]] constexpr T value_or(U&& defalut_value) && {
        return is_ok() ? std::move(value()) : static_cast<T>(std::forward<U>(defalut_value));
    }

    [[nodiscard]] constexpr T unwrap() && {
        if (is_err()) {
            std::terminate();
        }

        return std::move(value());
    }

    [[nodiscard]] constexpr E unwrap_err() && {
        if (is_ok()) {
            std::terminate();
        }

        return std::move(error());
    }

private:
    std::variant<T, E> storage_;
};

template<typename E>
class [[nodiscard]] Result<void, E> {
public:
    constexpr Result() noexcept : has_value_(true), error_() {}

    constexpr Result(E error) : has_value_(false), error_(std::move(error)) {}

    [[nodiscard]] constexpr bool is_ok() const noexcept {
        return has_value_;
    }

    [[nodiscard]] constexpr bool is_err() const noexcept {
        return !has_value_;
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return is_ok();
    }

    [[nodiscard]] constexpr E& error() & {
        return error_;
    }

    [[nodiscard]] constexpr const E& error() const& {
        return error_;
    }

    [[nodiscard]] constexpr E&& error() && {
        return std::move(error_);
    }

    constexpr void unwrap() const {
        if (is_err()) {
            std::terminate();
        }
    }

private:
    bool has_value_;
    E error_;
};

/// Helper Functions
template<typename T>
[[nodiscard]] inline Result<T> Ok(T value) {
    return Result<T>(std::move(value));
}

[[nodiscard]] inline Result<void> Ok() {
    return Result<void>();
}

template<typename T>
[[nodiscard]] inline Result<T> Err(Error error) {
    return Result<T>(std::move(error));
}

template<typename T>
[[nodiscard]] inline Result<T> Err(ErrorCode code) {
    return Result<T>(Error(code));
}

template<typename T>
[[nodiscard]] inline Result<T> Err(ErrorCode code, std::string message) {
    return Result<T>(Error(code, std::move(message)));
}

}  // namespace pulse
