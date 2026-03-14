#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "pulse/core/safetensor.hpp"

namespace {

std::filesystem::path write_safetensor_file(std::string_view file_name, std::string_view header_json) {
    namespace fs = std::filesystem;

    const auto temp_dir = fs::temp_directory_path() / "pulse_safetensor_tests";
    fs::create_directories(temp_dir);

    const auto path = temp_dir / file_name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    EXPECT_TRUE(out.is_open());

    const auto header_size = static_cast<std::uint64_t>(header_json.size());
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
    out.flush();
    EXPECT_TRUE(out.good());

    return path;
}

}  // namespace

TEST(SafeTensorLoaderTest, RejectsZeroElementTensorAtLoadTime) {
    const auto path = write_safetensor_file(
        "zero_element.safetensors",
        R"({"zero":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}})");

    const auto load_result = pulse::SafeTensorLoader::load(path);
    EXPECT_TRUE(load_result.is_err());
}

TEST(SafeTensorLoaderTest, RejectsByteSizeOverflowAtLoadTime) {
    const auto path = write_safetensor_file(
        "overflow_bytes.safetensors",
        R"({"overflow":{"dtype":"F64","shape":[2305843009213693952],"data_offsets":[0,0]}})");

    const auto load_result = pulse::SafeTensorLoader::load(path);
    EXPECT_TRUE(load_result.is_err());
}
