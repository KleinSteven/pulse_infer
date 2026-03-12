#pragma once

#include <cstddef>
#include <cstdint>

namespace pulse {

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using usize = std::size_t;
using isize = std::ptrdiff_t;

using f32 = float;
using f64 = double;

enum class DeviceType : u8 {
    CPU = 0,
    CUDA = 1,
    Unified = 3,  // CPU and CUDA UnifiedMemory
    Mmap = 4,
};

constexpr const char* device_type_str(DeviceType type) noexcept {
    switch (type) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::CUDA:
            return "CUDA";
        case DeviceType::Unified:
            return "Unified";
        case DeviceType::Mmap:
            return "Mmap";
        default:
            return "Unknown";
    }
}

enum class DataType : u8 {
    Float32 = 0,
    Float64 = 1,
    Int8 = 2,
    Int16 = 3,
    Int32 = 4,
    Int64 = 5,
    UInt8 = 6,
    UInt16 = 7,
    UInt32 = 8,
    UInt64 = 9,
};

constexpr usize data_type_size(DataType type) noexcept {
    switch (type) {
        case DataType::Float32:
        case DataType::Int32:
        case DataType::UInt32:
            return 4;
        case DataType::Float64:
        case DataType::Int64:
        case DataType::UInt64:
            return 8;
        case DataType::Int8:
        case DataType::UInt8:
            return 1;
        case DataType::Int16:
        case DataType::UInt16:
            return 2;
        default:
            return 0;
    }
}

constexpr const char* data_type_str(DataType type) noexcept {
    switch (type) {
        case DataType::Float32:
            return "float32";
        case DataType::Float64:
            return "float64";
        case DataType::Int8:
            return "int8";
        case DataType::Int16:
            return "int16";
        case DataType::Int32:
            return "int32";
        case DataType::Int64:
            return "int64";
        case DataType::UInt8:
            return "uint8";
        case DataType::UInt16:
            return "uint16";
        case DataType::UInt32:
            return "uint32";
        case DataType::UInt64:
            return "uint64";
        default:
            return "unknown";
    }
}

template<typename T>
struct CppTypeToDataType;

template<>
struct CppTypeToDataType<i8> {
    static constexpr DataType value = DataType::Int8;
};

template<>
struct CppTypeToDataType<i16> {
    static constexpr DataType value = DataType::Int16;
};

template<>
struct CppTypeToDataType<i32> {
    static constexpr DataType value = DataType::Int32;
};

template<>
struct CppTypeToDataType<i64> {
    static constexpr DataType value = DataType::Int64;
};

template<>
struct CppTypeToDataType<u8> {
    static constexpr DataType value = DataType::UInt8;
};

template<>
struct CppTypeToDataType<u16> {
    static constexpr DataType value = DataType::UInt16;
};

template<>
struct CppTypeToDataType<u32> {
    static constexpr DataType value = DataType::UInt32;
};

template<>
struct CppTypeToDataType<u64> {
    static constexpr DataType value = DataType::UInt64;
};

template<>
struct CppTypeToDataType<f32> {
    static constexpr DataType value = DataType::Float32;
};

template<>
struct CppTypeToDataType<f64> {
    static constexpr DataType value = DataType::Float64;
};

template<typename T>
constexpr DataType cpp_type_to_data_type_v = CppTypeToDataType<T>::value;

}  // namespace pulse
