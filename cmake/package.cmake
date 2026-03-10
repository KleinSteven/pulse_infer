include_guard()

include(FetchContent)

# spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.3
    GIT_SHALLOW    TRUE
)

set(CPPTRACE_DISABLE_CXX_20_MODULES ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(spdlog)

target_compile_definitions(spdlog PUBLIC
    SPDLOG_USE_STD_FORMAT=1
    SPDLOG_NO_EXCEPTIONS=1
)

# Google Test (for unit tests)
if(PULSE_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.16.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    include(GoogleTest)
endif()
