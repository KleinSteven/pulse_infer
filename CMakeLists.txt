cmake_minimum_required(VERSION 3.18)
project(PulseInfer VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

# ============================================================================
# C++ Standard and Compiler Requirements
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for clangd/IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PULSE_BUILD_TESTS "Build unit tests" ON)
option(PULSE_BUILD_EXAMPLES "Build examples" ON)
option(PULSE_BUILD_CUDA "Build with CUDA support" ON)
option(PULSE_BUILD_SHARED "Build shared libraries" OFF)
option(PULSE_ENABLE_ASAN "Enable AddressSanitizer" ON)
option(PULSE_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" ON)
option(PULSE_DISABLE_EXCEPTIONS "Disable C++ exceptions for project targets" OFF)

set(BUILD_TESTING ${PULSE_BUILD_TESTS} CACHE BOOL "Enable CTest support" FORCE)
include(CTest)

# ============================================================================
# Output Directories
# ============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================================
# Compiler Flags
# ============================================================================
set(PULSE_COMPILE_OPTIONS)
set(PULSE_LINK_OPTIONS)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  list(APPEND PULSE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
    $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>
    $<$<COMPILE_LANGUAGE:CXX>:-Wold-style-cast>
    $<$<COMPILE_LANGUAGE:CXX>:-Wcast-align>
    $<$<COMPILE_LANGUAGE:CXX>:-Wunused>
    $<$<COMPILE_LANGUAGE:CXX>:-Woverloaded-virtual>
    $<$<COMPILE_LANGUAGE:CXX>:-Wformat=2>
  )

  # Release optimizations
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG)
  endif()

  # Debug symbols
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
  endif()
endif()

if(PULSE_DISABLE_EXCEPTIONS)
  if(MSVC)
    list(APPEND PULSE_COMPILE_OPTIONS
      $<$<COMPILE_LANGUAGE:CXX>:/EHs-c->
    )
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND PULSE_COMPILE_OPTIONS
      $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    )
  endif()
endif()

# AddressSanitizer
if(PULSE_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# UndefinedBehaviorSanitizer
if(PULSE_ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=undefined)
endif()

# ============================================================================
# CUDA Configuration
# ============================================================================
if(PULSE_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # CUDA architectures
  # Common architectures:
  # - 70: V100
  # - 75: T4
  # - 80: A100
  # - 86: RTX 30xx
  # - 89: RTX 40xx
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA architectures")

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
  endif()

  message(STATUS "CUDA Enabled: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

include("${PROJECT_SOURCE_DIR}/cmake/package.cmake")

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(src)

if(PULSE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Examples directory removed as per user request
# if(PHOTON_BUILD_EXAMPLES)
#   add_subdirectory(examples)
# endif()
# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  PulseInfer Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Tests: ${PULSE_BUILD_TESTS}")
message(STATUS "  Build Examples: ${PULSE_BUILD_EXAMPLES}")
message(STATUS "  Build CUDA: ${PULSE_BUILD_CUDA}")
message(STATUS "  Build Shared: ${PULSE_BUILD_SHARED}")
message(STATUS "  Enable ASAN: ${PULSE_ENABLE_ASAN}")
message(STATUS "  Enable UBSAN: ${PULSE_ENABLE_UBSAN}")
message(STATUS "  Disable Exceptions: ${PULSE_DISABLE_EXCEPTIONS}")
message(STATUS "========================================")
message(STATUS "")
