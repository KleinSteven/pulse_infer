include_guard()

include(FetchContent)

set(SPDLOG_USE_STD_FORMAT OFF CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)

# fmt
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        12.0.0
    GIT_SHALLOW    TRUE
)

# spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.3
    GIT_SHALLOW    TRUE
)

# cpptrace
FetchContent_Declare(
    cpptrace
    GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG        v1.0.4
    GIT_SHALLOW    TRUE
)

set(CPPTRACE_DISABLE_CXX_20_MODULES ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(fmt spdlog cpptrace)

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
